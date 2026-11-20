"""
run_libero_interactive_with_gpt_ref.py

Interactive interface with GPT reference suggestions.

Like run_libero_interactive_commands.py, but optionally queries GPT at each step
to show what GPT would suggest, allowing you to compare your command choice against
GPT's reasoning.

Modes:
    user-only         Manual steering (no GPT) — just like interactive_commands.py
    gpt-reference     Show GPT suggestions as a reference before each command
    compare           Side-by-side: execute both user and GPT commands to compare

Usage:
    cd /proj/inf-scaling/zwhong/idan/vla-rl/openvla-oft

    export OPENAI_API_KEY='sk-...'

    python experiments/robot/libero/run_libero_interactive_with_gpt_ref.py \\
        --pretrained_checkpoint runs/<checkpoint> \\
        --unnorm_key libero_subgoal_multilevel \\
        --task_suite_name libero_goal \\
        --task_id 0 \\
        --gpt_mode gpt-reference \\
        --gpt_model gpt-4o \\
        --device cuda:0
"""

import base64
import io
import json
import logging
import os
import re
import sys
import time
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict

import draccus
import imageio
import numpy as np
import torch
from libero.libero import benchmark
from PIL import Image

sys.path.append("../..")
from experiments.robot.libero.libero_utils import (
    get_libero_dummy_action,
    get_libero_env,
    get_libero_image,
    get_libero_wrist_image,
    quat2axisangle,
)
from experiments.robot.libero.run_libero_eval import (
    GenerateConfig as BaseGenerateConfig,
    TASK_MAX_STEPS,
    validate_config,
    initialize_model,
    prepare_observation,
    process_action,
    log_message,
)
from experiments.robot.robot_utils import (
    DATE_TIME,
    get_action,
    get_image_resize_size,
    set_seed_everywhere,
)
from prismatic.vla.constants import NUM_ACTIONS_CHUNK


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


# =============================================================================
# Config
# =============================================================================

@dataclass
class InteractiveGptRefConfig(BaseGenerateConfig):
    # fmt: off
    initial_task: str = "open the drawer"

    # ── GPT settings ──────────────────────────────────────────────────────────
    # "user-only" = no GPT, just manual; "gpt-reference" = show GPT suggestions;
    # "compare" = execute both user and GPT commands
    gpt_mode: str = "gpt-reference"  # user-only | gpt-reference | compare
    gpt_model: str = "gpt-4o"
    # How many past entries to show GPT as history
    gpt_history_length: int = 3
    # Directory with in-context learning examples
    prompt_examples_dir: str = "/proj/inf-scaling/zwhong/idan/vla-rl/outputs/prompt_example_frames"
    rlds_results_dir: str = "/proj/inf-scaling/zwhong/idan/vla-rl/outputs/subgoal_results_rlds"

    # ── Output ────────────────────────────────────────────────────────────────
    video_dir: str = "./rollout_videos_interactive"
    local_log_dir: str = "./experiments/logs"
    device: str = "cuda:0"
    # fmt: on


# =============================================================================
# Image utilities
# =============================================================================

def encode_image_to_base64(img) -> str:
    """Encode a numpy array or PIL image to base64 PNG string."""
    if isinstance(img, np.ndarray):
        img = Image.fromarray(img)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def load_image_b64(path: str) -> Optional[str]:
    """Load an image file and encode to base64. Returns None if not found."""
    if not os.path.exists(path):
        return None
    return encode_image_to_base64(Image.open(path))


def format_proprio(obs: dict) -> str:
    """Format proprioceptive state as readable string."""
    eef_pos = obs["robot0_eef_pos"]
    grip = obs["robot0_gripper_qpos"]
    grip_state = "open" if grip[0] > 0.03 else "closed"
    return (
        f"  EEF: x={eef_pos[0]:7.3f} y={eef_pos[1]:7.3f} z={eef_pos[2]:7.3f}  |  "
        f"Gripper: {grip_state}"
    )


# =============================================================================
# GPT Reference Mode
# =============================================================================

def build_system_prompt() -> str:
    """Build the system prompt for GPT."""
    COMMAND_TYPES_DOC = """\
## Command types the VLA understands

1. **task** — Full task description
2. **subgoal** — Semantic phase description
3. **spatial_command** — Directional / relational goal
4. **object_affordance** — Grasp/contact style hint
5. **low_level_motion** — Motion direction words only
6. **low_level_motion_with_numbers** — Exact displacements (cm-scale, labeled "m")
"""

    AXIS_CONVENTION_DOC = """\
## ⚠ Axis convention note (training data bug)

| Physical motion | Say in command |
|---|---|
| Forward | move right |
| Backward | move left |
| Physical left | move forward |
| Physical right | move backward |
| Up | move up |
| Down | move down |
"""

    return (
        "You are a robot planner observing a tabletop manipulation task in progress.\n\n"
        "At this step, observe the current scene and suggest the NEXT command to\n"
        "steer the robot forward.\n\n"
        + COMMAND_TYPES_DOC + "\n\n"
        + AXIS_CONVENTION_DOC + "\n\n"
        "Output as JSON:\n"
        "{\n"
        '  "observation": "brief description of current state",\n'
        '  "reasoning": "why this command",\n'
        '  "command_type": "one of the 6 types",\n'
        '  "command": "the command string"\n'
        "}\n"
    )


def query_gpt_for_suggestion(
    task: str,
    obs_img: np.ndarray,
    obs_raw: dict,
    history: List[Dict],
    gpt_model: str = "gpt-4o",
    max_retries: int = 3,
) -> Optional[Dict]:
    """Query GPT to get a command suggestion for the current state.

    Returns a dict with keys: command_type, command, reasoning, observation
    or None if GPT fails.
    """
    try:
        from openai import OpenAI
    except ImportError:
        logger.error("OpenAI SDK not installed. Install with: pip install openai")
        return None

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        logger.error("OPENAI_API_KEY not set")
        return None

    client = OpenAI(api_key=api_key)

    # Build user message with image and history
    obs_b64 = encode_image_to_base64(obs_img)

    user_content = [
        {"type": "text", "text": f"Task: {task}\n\nCurrent observation:"},
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{obs_b64}", "detail": "low"}},
    ]

    if history:
        user_content.append({
            "type": "text",
            "text": f"\nRecent history ({len(history)} prior steps):\n"
        })
        for i, h in enumerate(history[-3:]):  # Show last 3
            user_content.append({
                "type": "text",
                "text": f"  {i+1}. [{h.get('command_type', 'unknown')}] {h.get('command', '(empty)')}"
            })

    user_content.append({
        "type": "text",
        "text": "\nGenerate the NEXT command. Respond with JSON only."
    })

    messages = [
        {"role": "system", "content": build_system_prompt()},
        {"role": "user", "content": user_content},
    ]

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=gpt_model,
                messages=messages,
                temperature=0.7,
                max_completion_tokens=256,
            )
            raw = response.choices[0].message.content.strip()

            # Parse JSON
            json_match = re.search(r"```(?:json)?\s*\n?([\s\S]*?)```", raw)
            if json_match:
                parsed = json.loads(json_match.group(1).strip())
            else:
                parsed = json.loads(re.search(r"(\{[\s\S]*\})", raw).group(1).strip())

            return {
                "command_type": parsed.get("command_type", "task").strip(),
                "command": parsed.get("command", "").strip(),
                "reasoning": parsed.get("reasoning", "").strip(),
                "observation": parsed.get("observation", "").strip(),
            }

        except Exception as e:
            if attempt < max_retries - 1:
                logger.debug(f"GPT attempt {attempt+1} failed: {e}. Retrying...")
                time.sleep(1)
            else:
                logger.error(f"GPT failed after {max_retries} retries: {e}")
                return None

    return None


# =============================================================================
# Main interactive loop (with GPT reference)
# =============================================================================

COMMAND_HELP = """\
╔════════════════════════════════════════════════════════════════════════════╗
║ Command Format: <TYPE> | <COMMAND_STRING>                                 ║
╚════════════════════════════════════════════════════════════════════════════╝

COMMAND TYPES (choose one, followed by pipe character |):

  task | <full task description>
  subgoal | <semantic phase description>
  spatial_command | <directional / relational goal>
  object_affordance | <grasp/contact style hint>
  low_level_motion | <motion words without numbers>
  low_level_motion_with_numbers | <exact displacements in cm-scale>

SPECIAL DIRECTIVES:

  quit, q, exit      → Exit cleanly
  reset              → Reset environment
  repeat             → Repeat last command
  gpt_suggest        → Show GPT suggestion (if in gpt-reference mode)
  help               → Show this help

Type 'help' for more information.
"""


class InteractiveSessionWithGpt:
    def __init__(self, config: InteractiveGptRefConfig, model, image_resize_size, env, task_suite):
        self.config = config
        self.model = model
        self.image_resize_size = image_resize_size
        self.env = env
        self.task_suite = task_suite

        self.step = 0
        self.action_chunk_step = 0
        self.frames = []
        self.action_history = []
        self.current_task = config.initial_task
        self.last_command = None
        self.last_command_type = None
        self.last_gpt_suggestion = None

        self.reset_obs()

    def reset_obs(self):
        """Reset and get initial observation."""
        obs = self.env.reset()
        self.step = 0
        self.action_chunk_step = 0
        self.frames = []
        self.action_history = []
        self.last_obs = obs
        return obs

    def get_current_image(self) -> np.ndarray:
        """Get current RGB image from environment."""
        obs = self.last_obs
        image = get_libero_image(obs)
        return image

    def display_state(self):
        """Display current state to user."""
        print("\n" + "=" * 80)
        print(f"STEP: {self.step:4d} | CHUNK: {self.action_chunk_step:2d}")
        print(f"TASK: {self.current_task}")
        print(f"PROP: {format_proprio(self.last_obs)}")
        print("=" * 80)

        img = self.get_current_image()
        img_rgb = img.astype(np.uint8) if img.dtype == np.float32 else img
        print(f"[Image: {img_rgb.shape[0]}×{img_rgb.shape[1]}]")

        if len(self.action_history) > 0:
            print(f"Last: {self.last_command_type} | {self.last_command}")

        print()

    def show_gpt_suggestion(self):
        """Query GPT and display its suggestion."""
        if self.config.gpt_mode == "user-only":
            print("  ⚠ GPT mode disabled (gpt_mode=user-only)")
            return

        print("  Querying GPT for suggestion...")
        img = self.get_current_image()

        suggestion = query_gpt_for_suggestion(
            self.current_task,
            img,
            self.last_obs,
            self.action_history,
            gpt_model=self.config.gpt_model,
        )

        if not suggestion:
            print("  ✗ GPT query failed")
            return

        self.last_gpt_suggestion = suggestion

        print("\n" + "─" * 80)
        print("GPT SUGGESTION:")
        print("─" * 80)
        print(f"Observation: {suggestion['observation']}")
        print(f"Reasoning:   {suggestion['reasoning']}")
        print(f"Type:        {suggestion['command_type']}")
        print(f"Command:     {suggestion['command']}")
        print("─" * 80)
        print("(You can still type your own command, or press Enter to use GPT's suggestion)")
        print()

    def parse_user_input(self, user_input: str) -> Tuple[Optional[str], Optional[str]]:
        """Parse user input."""
        user_input = user_input.strip()

        if not user_input:
            # Empty input: use GPT suggestion if available
            if self.last_gpt_suggestion:
                return self.last_gpt_suggestion["command_type"], self.last_gpt_suggestion["command"]
            return None, None

        # Special directives
        if user_input.lower() in ["quit", "q", "exit"]:
            return "QUIT", None
        if user_input.lower() == "reset":
            return "RESET", None
        if user_input.lower() == "repeat":
            if self.last_command is None:
                print("  ⚠ No previous command to repeat.")
                return None, None
            return self.last_command_type, self.last_command
        if user_input.lower() == "gpt_suggest":
            return "GPT_SUGGEST", None
        if user_input.lower() == "help":
            return "HELP", None

        # Parse command
        if "|" not in user_input:
            print("  ⚠ Invalid format. Use: <TYPE> | <COMMAND>")
            return None, None

        parts = user_input.split("|", 1)
        cmd_type = parts[0].strip().lower()
        cmd_string = parts[1].strip()

        valid_types = {
            "task", "subgoal", "spatial_command",
            "object_affordance", "low_level_motion",
            "low_level_motion_with_numbers"
        }

        if cmd_type not in valid_types:
            print(f"  ⚠ Unknown command type: {cmd_type}")
            return None, None

        if not cmd_string:
            print("  ⚠ Command string cannot be empty.")
            return None, None

        return cmd_type, cmd_string

    def execute_command(self, command_type: str, command_string: str) -> bool:
        """Execute a command via the VLA model."""
        try:
            obs = self.last_obs
            image = get_libero_image(obs)
            wrist_image = get_libero_wrist_image(obs)
            state_dict = prepare_observation(obs, self.config)

            prompt = f"{command_type}: {command_string}"

            action = get_action(
                self.model,
                image,
                wrist_image,
                prompt,
                state_dict,
                self.image_resize_size,
                config=self.config,
            )

            # Execute action
            for step_in_chunk in range(NUM_ACTIONS_CHUNK):
                action_to_take = action[step_in_chunk]
                obs, reward, done, info = self.env.step(action_to_take)
                frame = get_libero_image(obs)
                self.frames.append(frame)
                self.step += 1

            self.last_obs = obs
            self.action_chunk_step += 1

            self.action_history.append({
                "command_type": command_type,
                "command": command_string,
                "step": self.step - NUM_ACTIONS_CHUNK,
            })
            self.last_command = command_string
            self.last_command_type = command_type
            self.last_gpt_suggestion = None  # Clear suggestion after execution

            return True

        except Exception as e:
            print(f"  ✗ Error executing command: {e}")
            return False

    def save_video(self, suffix: str = ""):
        """Save recorded frames to video file."""
        if not self.frames:
            print("  ⚠ No frames recorded yet.")
            return

        os.makedirs(self.config.video_dir, exist_ok=True)

        task_name = self.current_task[:30].replace(" ", "_")
        if suffix:
            video_path = os.path.join(self.config.video_dir, f"interactive_gpt_{task_name}_{suffix}.mp4")
        else:
            video_path = os.path.join(self.config.video_dir, f"interactive_gpt_{task_name}_{DATE_TIME}.mp4")

        try:
            imageio.mimwrite(video_path, self.frames, fps=10)
            print(f"  ✓ Saved video: {video_path}")
        except Exception as e:
            print(f"  ✗ Error saving video: {e}")

    def run(self):
        """Main interactive loop."""
        print(COMMAND_HELP)
        print(f"\nMode: {self.config.gpt_mode}")
        print(f"Task: {self.current_task}\n")

        while True:
            self.display_state()

            try:
                user_input = input("Enter command (or 'help'): ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n\nInterrupted, saving video...")
                self.save_video("final")
                break

            cmd_type, cmd_string = self.parse_user_input(user_input)

            if cmd_type is None and cmd_string is None:
                continue

            if cmd_type == "QUIT":
                print("\nSaving video before exit...")
                self.save_video("final")
                print("Done. Goodbye!")
                break

            if cmd_type == "RESET":
                print("Resetting environment...")
                self.reset_obs()
                print("✓ Reset complete.")
                continue

            if cmd_type == "GPT_SUGGEST":
                self.show_gpt_suggestion()
                continue

            if cmd_type == "HELP":
                print(COMMAND_HELP)
                continue

            if cmd_type and cmd_string:
                # Show GPT suggestion if in gpt-reference mode
                if self.config.gpt_mode == "gpt-reference" and not user_input.strip():
                    pass  # Already shown GPT suggestion in parse_user_input
                elif self.config.gpt_mode == "gpt-reference":
                    self.show_gpt_suggestion()

                print(f"\nExecuting: {cmd_type} | {cmd_string}")
                if self.execute_command(cmd_type, cmd_string):
                    print(f"✓ Command executed (step {self.step})")
                else:
                    print(f"✗ Command failed")


# =============================================================================
# Main
# =============================================================================

def main(config: InteractiveGptRefConfig):
    """Initialize and run interactive session with GPT reference."""

    validate_config(config)
    set_seed_everywhere(config.seed)

    logger.info(f"Loading environment: {config.task_suite_name}...")
    task_suite = benchmark.get_benchmark_dict()[config.task_suite_name]()
    env = get_libero_env(config)

    logger.info(f"Loading model: {config.pretrained_checkpoint}...")
    model, image_resize_size = initialize_model(config)
    model = model.to(config.device)
    model.eval()

    logger.info("=" * 80)
    logger.info("Interactive Command Interface (with GPT Reference) Ready")
    logger.info("=" * 80)
    logger.info(f"Mode: {config.gpt_mode}")
    logger.info(f"Device: {config.device}")
    logger.info("=" * 80)

    session = InteractiveSessionWithGpt(config, model, image_resize_size, env, task_suite)
    session.run()


if __name__ == "__main__":
    config = draccus.parse(InteractiveGptRefConfig)
    main(config)
