"""
run_libero_interactive_commands.py

Interactive interface for manually steering the robot via typed commands.
No GPT — just observe the current state and type what you want the robot to do.

At each action-chunk boundary:
    Display → observe image + proprioceptive state
    User input → command type + command string (or special directives)
    → VLA executes 8-step chunk → repeat

Command types (same 6 the model was trained on):
    task                      full task description
    subgoal                   semantic phase description
    spatial_command           directional / relational goal
    object_affordance         grasp / contact style hint
    low_level_motion          motion-direction words only
    low_level_motion_with_numbers  exact displacements with magnitudes

Special directives (user input commands):
    quit / q                  exit cleanly
    reset                     reset environment to initial state
    save_video                save video of this episode
    set_task <description>    set/change task description
    repeat                    repeat the last command

Usage:
    cd /proj/inf-scaling/zwhong/idan/vla-rl/openvla-oft

    python experiments/robot/libero/run_libero_interactive_commands.py \\
        --pretrained_checkpoint runs/<checkpoint> \\
        --unnorm_key libero_subgoal_multilevel \\
        --task_suite_name libero_goal \\
        --task_id 0 \\
        --use_l1_regression True \\
        --use_lora True \\
        --video_dir ./rollout_videos_interactive \\
        --device cuda:0
"""

import base64
import io
import json
import logging
import os
import sys
import time
from dataclasses import dataclass
from typing import Optional, Tuple

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
class InteractiveCommandsConfig(BaseGenerateConfig):
    # fmt: off
    # Initial task (can be changed interactively)
    initial_task: str = "open the drawer"

    # ── Output / logging ──────────────────────────────────────────────────────
    video_dir: str = "./rollout_videos_interactive"
    local_log_dir: str = "./experiments/logs"

    # Device for inference
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


def format_proprio(obs: dict) -> str:
    """Format proprioceptive state as readable string."""
    eef_pos = obs["robot0_eef_pos"]
    grip = obs["robot0_gripper_qpos"]
    grip_state = "open" if grip[0] > 0.03 else "closed"
    return (
        f"  EEF: x={eef_pos[0]:7.3f} y={eef_pos[1]:7.3f} z={eef_pos[2]:7.3f}  |  "
        f"Gripper: {grip_state}"
    )


def save_image(img, path: str):
    """Save image to file."""
    if isinstance(img, np.ndarray):
        img = Image.fromarray(img)
    img.save(path)


# =============================================================================
# Command type help
# =============================================================================

COMMAND_HELP = """\
╔════════════════════════════════════════════════════════════════════════════╗
║ Command Format: <TYPE> | <COMMAND_STRING>                                 ║
╚════════════════════════════════════════════════════════════════════════════╝

COMMAND TYPES (choose one, followed by pipe character |, then the command):

  1. task           → Full task description
                      Example: task | put the bowl on the plate

  2. subgoal        → Semantic phase description
                      Example: subgoal | move gripper above the drawer

  3. spatial_command → Directional / relational goal
                      Example: spatial_command | move down and to the left

  4. object_affordance → Grasp/contact style hint
                      Example: object_affordance | grip the handle firmly

  5. low_level_motion → Motion words (no numbers)
                      Example: low_level_motion | move up, forward, right

  6. low_level_motion_with_numbers → Exact displacements (cm-scale)
                      Example: low_level_motion_with_numbers | move down 15m, right 10m

⚠️  AXIS CONVENTION NOTE (training data bug):
    Physical forward  → say "move right"
    Physical backward → say "move left"
    Physical left     → say "move forward"
    Physical right    → say "move backward"
    Up/Down are correct

SPECIAL DIRECTIVES:

  quit, q, exit      → Exit cleanly (saves video)
  reset              → Reset environment to initial state
  save_video         → Force save the current rollout video
  repeat             → Repeat the last command
  help               → Show this help text
  ────────────────────────────────────────────────────────────────────────────
"""


# =============================================================================
# Main interactive loop
# =============================================================================

class InteractiveSession:
    def __init__(self, config: InteractiveCommandsConfig, model, image_resize_size, env, task_suite):
        self.config = config
        self.model = model
        self.image_resize_size = image_resize_size
        self.env = env
        self.task_suite = task_suite

        # State tracking
        self.step = 0
        self.action_chunk_step = 0
        self.frames = []
        self.action_history = []
        self.current_task = config.initial_task
        self.last_command = None
        self.last_command_type = None

        # Initial observation
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

        # Display current image
        img = self.get_current_image()
        img_rgb = img.astype(np.uint8) if img.dtype == np.float32 else img
        pil_img = Image.fromarray(img_rgb)
        print(f"\n[Image: {img_rgb.shape[0]}×{img_rgb.shape[1]}]")

        if len(self.action_history) > 0:
            print(f"Last command: {self.last_command_type} | {self.last_command}")

        print()

    def parse_user_input(self, user_input: str) -> Tuple[Optional[str], Optional[str]]:
        """
        Parse user input and return (command_type, command_string).
        Handles special directives and command type parsing.

        Returns (None, None) for special directives.
        Raises ValueError for invalid input.
        """
        user_input = user_input.strip()

        if not user_input:
            return None, None

        # Special directives
        if user_input.lower() in ["quit", "q", "exit"]:
            return "QUIT", None
        if user_input.lower() == "reset":
            return "RESET", None
        if user_input.lower() == "save_video":
            return "SAVE_VIDEO", None
        if user_input.lower() == "repeat":
            if self.last_command is None:
                print("  ⚠ No previous command to repeat.")
                return None, None
            return self.last_command_type, self.last_command
        if user_input.lower() == "help":
            return "HELP", None
        if user_input.lower().startswith("set_task "):
            task_desc = user_input[9:].strip()
            return "SET_TASK", task_desc

        # Parse command: "TYPE | COMMAND"
        if "|" not in user_input:
            print("  ⚠ Invalid format. Use: <TYPE> | <COMMAND>")
            print("     (Type 'help' for examples)")
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
            print(f"     Valid types: {', '.join(sorted(valid_types))}")
            return None, None

        if not cmd_string:
            print("  ⚠ Command string cannot be empty.")
            return None, None

        return cmd_type, cmd_string

    def execute_command(self, command_type: str, command_string: str) -> bool:
        """
        Execute a command via the VLA model.
        Returns True if execution was successful, False otherwise.
        """
        try:
            # Prepare observation for VLA
            obs = self.last_obs
            image = get_libero_image(obs)
            wrist_image = get_libero_wrist_image(obs)
            state_dict = prepare_observation(obs, self.config)

            # Construct prompt
            # The model expects specific fields based on command_type
            prompt = f"{command_type}: {command_string}"

            # Get action from model
            action = get_action(
                self.model,
                image,
                wrist_image,
                prompt,
                state_dict,
                self.image_resize_size,
                config=self.config,
            )

            # Execute action (NUM_ACTIONS_CHUNK steps = 8 steps)
            for step_in_chunk in range(NUM_ACTIONS_CHUNK):
                action_to_take = action[step_in_chunk]
                obs, reward, done, info = self.env.step(action_to_take)

                # Capture frame
                frame = get_libero_image(obs)
                self.frames.append(frame)

                self.step += 1

            # Update last observation
            self.last_obs = obs
            self.action_chunk_step += 1

            # Record command
            self.action_history.append({
                "chunk": self.action_chunk_step - 1,
                "command_type": command_type,
                "command": command_string,
                "step": self.step - NUM_ACTIONS_CHUNK,
            })
            self.last_command = command_string
            self.last_command_type = command_type

            return True

        except Exception as e:
            print(f"  ✗ Error executing command: {e}")
            import traceback
            traceback.print_exc()
            return False

    def save_video(self, suffix: str = ""):
        """Save recorded frames to video file."""
        if not self.frames:
            print("  ⚠ No frames recorded yet.")
            return

        # Create output directory
        os.makedirs(self.config.video_dir, exist_ok=True)

        # Video filename
        task_name = self.current_task[:30].replace(" ", "_")
        if suffix:
            video_path = os.path.join(
                self.config.video_dir,
                f"interactive_{task_name}_{suffix}.mp4"
            )
        else:
            video_path = os.path.join(
                self.config.video_dir,
                f"interactive_{task_name}_{DATE_TIME}.mp4"
            )

        # Save video
        try:
            imageio.mimwrite(video_path, self.frames, fps=10)
            print(f"  ✓ Saved video: {video_path}")
        except Exception as e:
            print(f"  ✗ Error saving video: {e}")

    def run(self):
        """Main interactive loop."""
        print("\n" + COMMAND_HELP)
        print(f"\nInitializing with task: '{self.current_task}'")
        print("Waiting for your first command...\n")

        while True:
            self.display_state()

            # Get user input
            try:
                user_input = input("Enter command (or 'help' for options): ").strip()
            except EOFError:
                print("\nEnd of input, exiting...")
                break
            except KeyboardInterrupt:
                print("\n\nInterrupted by user, exiting...")
                break

            # Parse input
            cmd_type, cmd_string = self.parse_user_input(user_input)

            if cmd_type is None and cmd_string is None:
                continue

            # Handle special directives
            if cmd_type == "QUIT":
                print("\n  Saving video before exit...")
                self.save_video("final")
                print("  Done. Goodbye!")
                break

            if cmd_type == "RESET":
                print("\n  Resetting environment...")
                self.reset_obs()
                print("  ✓ Reset complete.")
                continue

            if cmd_type == "SAVE_VIDEO":
                print("\n  Saving video...")
                self.save_video()
                continue

            if cmd_type == "HELP":
                print(COMMAND_HELP)
                continue

            if cmd_type == "SET_TASK":
                self.current_task = cmd_string
                print(f"  ✓ Task set to: '{self.current_task}'")
                continue

            # Execute command
            if cmd_type and cmd_string:
                print(f"\n  Executing: {cmd_type} | {cmd_string}")
                if self.execute_command(cmd_type, cmd_string):
                    print(f"  ✓ Command executed (step {self.step})")
                else:
                    print(f"  ✗ Command failed")


# =============================================================================
# Main entry point
# =============================================================================

def main(config: InteractiveCommandsConfig):
    """Initialize and run interactive session."""

    # Validate config
    validate_config(config)

    # Set seed
    set_seed_everywhere(config.seed)

    # Get environment
    logger.info(f"Loading environment: {config.task_suite_name}...")
    task_suite = benchmark.get_benchmark_dict()[config.task_suite_name]()
    env = get_libero_env(config)

    # Initialize model
    logger.info(f"Loading model: {config.pretrained_checkpoint}...")
    model, image_resize_size = initialize_model(config)

    # Move model to device
    model = model.to(config.device)
    model.eval()

    logger.info("=" * 80)
    logger.info("Interactive Command Interface Ready")
    logger.info("=" * 80)
    logger.info(f"Task Suite: {config.task_suite_name}")
    logger.info(f"Model: {config.pretrained_checkpoint}")
    logger.info(f"Device: {config.device}")
    logger.info("=" * 80)

    # Create and run session
    session = InteractiveSession(config, model, image_resize_size, env, task_suite)
    session.run()


if __name__ == "__main__":
    config = draccus.parse(InteractiveCommandsConfig)
    main(config)
