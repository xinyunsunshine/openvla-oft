"""
run_libero_eval_gpt_commands.py

Closed-loop evaluation: GPT generates steering commands that drive a
multilevel-finetuned OpenVLA-OFT model on LIBERO-Goal tasks.

At each action-chunk boundary:
    observe → GPT picks command type + generates command string
    → VLA executes 8-step chunk → observe result → repeat

Command types (same 6 keys the model was trained on):
    task                      full task description
    subgoal                   semantic phase description
    spatial_command           directional / relational goal
    object_affordance         grasp / contact style hint
    low_level_motion          motion-direction words only
    low_level_motion_with_numbers  exact displacements with magnitudes

Modes:
    gpt_commands   GPT picks command type and generates command at each chunk
    openvla_only   Baseline: VLA with plain task description (no GPT)

Usage:
    cd /proj/inf-scaling/zwhong/idan/vla-rl/openvla-oft

    export OPENAI_API_KEY='sk-...'

    python experiments/robot/libero/run_libero_eval_gpt_commands.py \\
        --pretrained_checkpoint runs/<multilevel6_checkpoint> \\
        --unnorm_key libero_subgoal_multilevel \\
        --task_suite_name libero_goal \\
        --task_id_start 0 --task_id_end -1 \\
        --num_trials_per_task 3 \\
        --eval_mode gpt_commands \\
        --gpt_model gpt-4o \\
        --rlds_results_dir ../outputs/subgoal_results_rlds \\
        --prompt_examples_dir ../outputs/prompt_example_frames \\
        --video_dir ./rollout_videos_gpt_commands
"""

import base64
import io
import json
import logging
import os
import re
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import draccus
import imageio
import numpy as np
import torch
import tqdm
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
class GptCommandsEvalConfig(BaseGenerateConfig):
    # fmt: off
    eval_mode: str = "gpt_commands"   # gpt_commands | openvla_only

    # ── GPT settings ──────────────────────────────────────────────────────────
    gpt_model: str = "gpt-5.4"
    # Re-query GPT every N action chunks (1 = every 8 steps)
    gpt_replan_every_n_chunks: int = 3
    # How many past (command, image) entries to include in history
    gpt_history_length: int = 3
    # Allow GPT to freely choose the command type, or pin to a specific one.
    # "auto" = GPT decides; or one of: task, subgoal, spatial_command,
    #          object_affordance, low_level_motion, low_level_motion_with_numbers
    gpt_command_type: str = "auto"
    # Comma-separated list of allowed command types (restricts GPT's choices).
    # "all" = no restriction.
    # Example: "task,subgoal,low_level_motion,low_level_motion_with_numbers"
    allowed_command_types: str = "all"

    # ── GPT strategy ──────────────────────────────────────────────────────────
    # "monitoring" : GPT tracks subgoal progress and execution quality each query,
    #                decides whether to repeat / refine / advance / switch type.
    # "simple"     : original version — GPT just picks a command type each query.
    gpt_strategy: str = "monitoring"
    # Max chunks to stay on the same subgoal before forcing advance (monitoring only)
    max_chunks_per_subgoal: int = 10
    # Min consecutive chunks before allowing a command-type switch (monitoring only)
    min_chunks_before_type_switch: int = 2

    # ── RLDS episode map for building in-context examples ─────────────────────
    rlds_results_dir: str = "/proj/inf-scaling/zwhong/idan/vla-rl/outputs/subgoal_results_rlds"
    # Directory containing extracted per-subgoal PNG frames (for system prompt ICL)
    prompt_examples_dir: str = "/proj/inf-scaling/zwhong/idan/vla-rl/outputs/prompt_example_frames"

    # ── Output / logging ──────────────────────────────────────────────────────
    video_dir: str = "./rollout_videos_gpt_commands"
    local_log_dir: str = "./experiments/logs"
    # fmt: on


# =============================================================================
# RLDS episode map
# =============================================================================

def build_episode_map(rlds_results_dir: str) -> Dict[str, List[dict]]:
    """Scan rlds_results_dir and build task description → episode list mapping."""
    ep_map: Dict[str, list] = defaultdict(list)
    for ep_name in sorted(os.listdir(rlds_results_dir)):
        ep_dir = os.path.join(rlds_results_dir, ep_name)
        results_path = os.path.join(ep_dir, "results.json")
        if not os.path.exists(results_path):
            continue
        with open(results_path) as f:
            data = json.load(f)
        steering_path = os.path.join(ep_dir, "steering_commands.json")
        if os.path.exists(steering_path):
            with open(steering_path) as f:
                data["steering_commands"] = json.load(f).get("steering_commands", [])
        task_str = data.get("task", "")
        ep_map[task_str].append({
            "episode_index": data.get("episode_index", -1),
            "ep_dir": ep_dir,
            "subgoal_data": data,
        })
    for key in ep_map:
        ep_map[key].sort(key=lambda x: x["episode_index"])
    return dict(ep_map)


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
    """Load an image file and encode to base64.  Returns None if not found."""
    if not os.path.exists(path):
        return None
    return encode_image_to_base64(Image.open(path))


def format_proprio(obs: dict) -> str:
    eef_pos = obs["robot0_eef_pos"]
    grip = obs["robot0_gripper_qpos"]
    grip_state = "open" if grip[0] > 0.03 else "closed"
    return (
        f"EEF position (m): x={eef_pos[0]:.3f}, y={eef_pos[1]:.3f}, z={eef_pos[2]:.3f}\n"
        f"Gripper: {grip_state}"
    )


# =============================================================================
# System prompt construction
# =============================================================================

# Cached system prompt content (text + image references)
_system_prompt_cache: Optional[Tuple[str, list]] = None


COMMAND_TYPES_DOC = """\
## Command types the VLA understands

The fine-tuned OpenVLA-OFT model was trained with 6 types of language instructions.
At each query you choose the most appropriate type for the current situation:

### 1. task
The full task description.  Use at the very start or when the robot is lost.
Example:  "put the bowl on the plate"
Example:  "open the middle drawer of the cabinet"

### 2. subgoal
A short semantic description of the current phase.  Best for phase transitions.
Example:  "move the gripper from above the table toward the bowl, then lower and align it around the bowl"
Example:  "grasp the wine bottle, lift it off the table, and carry it rightward toward the wooden rack"
Example:  "lower the bowl onto the plate, then release it"

### 3. spatial_command
A directional / positional description of the goal state.  Best for mid-phase guidance.
Example:  "Move the gripper down, forward, and to the right to align around the bowl."
Example:  "Move the gripper down, backward, and to the right towards the cabinet on the right."
Example:  "lower the bowl onto the plate, then release it"

### 4. object_affordance
A grasp or contact style hint — how to interact with the object.
Example:  "Align the gripper around the bowl without grasping."
Example:  "hook fingers around the drawer handle from the front"
Example:  "release the wine bottle into the rack"

### 5. low_level_motion
Concise motion direction words, no numbers.  Best for the model to follow when
the direction is clear but magnitude matters less.
Example:  "move down, forward, and right"
Example:  "move up, backward, and right, then release"
Example:  "move down and release"

### 6. low_level_motion_with_numbers
Exact displacements computed from the demo trajectory.  Best for precise motions.
IMPORTANT: magnitudes use the label "m" but are centimeter-scale values (e.g. "16m" ≈ 16 cm).
Example:  "move down 16m, move forward 1m, move right 11m"
Example:  "move up 9m, move backward 1m, move right 8m, open gripper"
Example:  "move down 18m, move forward 21m, move left 8m"
"""


AXIS_CONVENTION_DOC = """\
## ⚠ Axis convention note (training data has a labeling bug)

The training data has swapped dx/dy labels.  Use this table when generating
`spatial_command`, `low_level_motion`, or `low_level_motion_with_numbers`:

| Physical robot motion | Say in the command |
|---|---|
| Forward (away from robot base) | `move right`     |
| Backward (toward robot base)   | `move left`      |
| Physical left                  | `move forward`   |
| Physical right                 | `move backward`  |
| Up                             | `move up`    ← correct |
| Down                           | `move down`  ← correct |

The `subgoal`, `task`, and `object_affordance` commands are not affected —
use natural English for those (the model learned them without axis confusion).
Only apply the swap for direction-word commands.
"""


def build_icl_content(prompt_examples_dir: str) -> list:
    """Build in-context learning content blocks (text + images) for system prompt.

    Returns a list of OpenAI content blocks ready to be appended to a
    system message.  Each episode is shown as: task, then for each subgoal —
    the observation image at subgoal start, the commands for that subgoal.

    Episodes used:
        ep00000  put the bowl on the plate           (3 subgoals)
        ep00300  open the middle drawer of cabinet   (4 subgoals)
        ep00100  put the wine bottle on the rack     (4 subgoals)
    """
    # Per-episode data aligned with the extracted PNGs in prompt_examples_dir
    episodes = [
        {
            "task": "put the bowl on the plate",
            "ep_dir": "ep00000",
            "subgoals": [
                {
                    "img": "ep000_sg0_start.png",
                    "frame": 0,
                    "subgoal": "move the gripper from above the table toward the bowl, then lower and align it around the bowl",
                    "spatial_command": "Move the gripper down, forward, and to the right to align around the bowl.",
                    "object_affordance": "Align the gripper around the bowl without grasping.",
                    "low_level_motion": "move down, forward, and right",
                    "low_level_motion_with_numbers": "move down 16m, move forward 1m, move right 11m",
                },
                {
                    "img": "ep000_sg1_start.png",
                    "frame": 42,
                    "subgoal": "grasp and lift the bowl off the table, then carry it rightward toward the plate",
                    "spatial_command": "lift the bowl up, move it backward and to the right above the plate, then release it",
                    "object_affordance": "release the bowl above the plate from directly above",
                    "low_level_motion": "move up, backward, and right, then release",
                    "low_level_motion_with_numbers": "move up 9m, move backward 1m, move right 8m, open gripper",
                },
                {
                    "img": "ep000_sg2_start.png",
                    "frame": 75,
                    "subgoal": "position the bowl above the plate, then lower it onto the plate and release it",
                    "spatial_command": "lower the bowl onto the plate, then release it",
                    "object_affordance": "release the bowl onto the plate from above",
                    "low_level_motion": "move down and release",
                    "low_level_motion_with_numbers": "move down 9m, move left 1m, close gripper",
                },
            ],
        },
        {
            "task": "open the middle drawer of the cabinet",
            "ep_dir": "ep00300",
            "subgoals": [
                {
                    "img": "ep300_sg0_start.png",
                    "frame": 0,
                    "subgoal": "move the gripper from above the table toward the cabinet on the right",
                    "spatial_command": "Move the gripper down, backward, and to the right towards the cabinet on the right.",
                    "object_affordance": "move the gripper toward the cabinet on the right",
                    "low_level_motion": "move down, backward, and right",
                    "low_level_motion_with_numbers": "move down 8m, move backward 6m, move right 17m",
                },
                {
                    "img": "ep300_sg1_start.png",
                    "frame": 58,
                    "subgoal": "lower and align the gripper in front of the middle drawer handle",
                    "spatial_command": "Move the gripper down, backward, and to the left to align with the drawer handle.",
                    "object_affordance": "Hold the gripper open and align it in front of the drawer handle.",
                    "low_level_motion": "move down, backward, and left",
                    "low_level_motion_with_numbers": "move down 3m, move backward 7m, move left 4m",
                },
                {
                    "img": "ep300_sg2_start.png",
                    "frame": 79,
                    "subgoal": "make contact with and grasp the middle drawer handle",
                    "spatial_command": "move the gripper down, backward, and to the left to align with the drawer handle, then grasp it",
                    "object_affordance": "hook fingers around the drawer handle from the front",
                    "low_level_motion": "move down, backward, and left, then grasp",
                    "low_level_motion_with_numbers": "move down 16m, move backward 7m, move left 5m",
                },
                {
                    "img": "ep300_sg3_start.png",
                    "frame": 123,
                    "subgoal": "pull the middle drawer outward until it is fully open",
                    "spatial_command": "Move the gripper down, forward, and to the left while pulling the middle drawer outward.",
                    "object_affordance": "hook fingers around the drawer handle and pull it backward",
                    "low_level_motion": "move down, forward, and left",
                    "low_level_motion_with_numbers": "move down 18m, move forward 21m, move left 8m",
                },
            ],
        },
        {
            "task": "put the wine bottle on the rack",
            "ep_dir": "ep00100",
            "subgoals": [
                {
                    "img": "ep100_sg0_start.png",
                    "frame": 0,
                    "subgoal": "move the gripper from above the table toward the wine bottle",
                    "spatial_command": "Move the gripper down and backward toward the wine bottle, keeping it open.",
                    "object_affordance": "Hold the gripper open while moving toward the wine bottle",
                    "low_level_motion": "move down and backward",
                    "low_level_motion_with_numbers": "move down 12m, move backward 4m",
                },
                {
                    "img": "ep100_sg1_start.png",
                    "frame": 62,
                    "subgoal": "lower and align the gripper around the wine bottle",
                    "spatial_command": "move down and backward to align the gripper around the wine bottle, then open the gripper",
                    "object_affordance": "open gripper to surround the neck of the wine bottle",
                    "low_level_motion": "move down and backward, open gripper",
                    "low_level_motion_with_numbers": "move down 2m, move backward 3m, open gripper",
                },
                {
                    "img": "ep100_sg2_start.png",
                    "frame": 75,
                    "subgoal": "grasp the wine bottle, lift it off the table, and carry it rightward toward the wooden rack",
                    "spatial_command": "move the gripper up, backward, and left to align with the wine bottle, then grasp it",
                    "object_affordance": "grasp the wine bottle from above with a secure grip",
                    "low_level_motion": "move up, backward, and left, then grasp",
                    "low_level_motion_with_numbers": "move up 13m, move backward 4m, move left 2m",
                },
                {
                    "img": "ep100_sg3_start.png",
                    "frame": 92,
                    "subgoal": "align the bottle with an open slot in the rack, insert it into the rack, and release it",
                    "spatial_command": "move the wine bottle down, backward, and to the right, then release it into the rack",
                    "object_affordance": "release the wine bottle into the rack",
                    "low_level_motion": "move down, backward, and right, then release",
                    "low_level_motion_with_numbers": "move down 1m, move backward 6m, move right 1m, close gripper",
                },
            ],
        },
    ]

    content = [{"type": "text", "text": "## In-context examples from real training episodes\n\n"
                "Below are 3 full episodes showing the complete sequence of commands used to\n"
                "complete each task.  For each subgoal you see the observation at the start\n"
                "of that phase and all 6 command types computed from the demo trajectory.\n"
                "Study these to understand what commands look like at each phase.\n"}]

    for ep in episodes:
        content.append({
            "type": "text",
            "text": f"\n### Episode: \"{ep['task']}\"\n"
        })
        for sg_idx, sg in enumerate(ep["subgoals"]):
            img_path = os.path.join(prompt_examples_dir, sg["img"])
            b64 = load_image_b64(img_path)

            content.append({
                "type": "text",
                "text": (
                    f"\n**Subgoal {sg_idx}** (frame {sg['frame']}):\n"
                    f"  task:                         \"{ep['task']}\"\n"
                    f"  subgoal:                      \"{sg['subgoal']}\"\n"
                    f"  spatial_command:              \"{sg['spatial_command']}\"\n"
                    f"  object_affordance:            \"{sg['object_affordance']}\"\n"
                    f"  low_level_motion:             \"{sg['low_level_motion']}\"\n"
                    f"  low_level_motion_with_numbers:\"{sg['low_level_motion_with_numbers']}\"\n"
                    f"\nObservation at subgoal {sg_idx} start:"
                )
            })
            if b64:
                content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{b64}", "detail": "low"},
                })
            else:
                content.append({
                    "type": "text",
                    "text": f"  [image not available: {sg['img']}]"
                })

    return content


def build_system_prompt(prompt_examples_dir: str, allowed_types: Optional[List[str]] = None) -> Tuple[str, list]:
    """Build the system prompt for GPT command generation.

    Args:
        allowed_types: If provided, restricts which command types GPT may use.

    Returns:
        (system_text, icl_content_blocks)
    """
    ALL_TYPES = ["task", "subgoal", "spatial_command", "object_affordance",
                 "low_level_motion", "low_level_motion_with_numbers"]
    if not allowed_types:
        allowed_types = ALL_TYPES
    allowed_str = "|".join(allowed_types)

    # Strategy hint based on allowed types
    if set(allowed_types) == {"subgoal"}:
        strategy = (
        "  3. **PICK COMMAND TYPE**: You must always use `subgoal`.\n"
        "     Write a short semantic description of the current phase.\n"
        )
    elif len(allowed_types) <= 4 and "spatial_command" not in allowed_types:
        strategy = (
        "  3. **PICK COMMAND TYPE**: Always start with `task` or `subgoal` to orient\n"
        "     the robot at a high level. Only switch to `low_level_motion` or\n"
        "     `low_level_motion_with_numbers` if the robot is stuck or needs precise\n"
        "     correction after multiple high-level attempts have failed.\n"
        )
    else:
        strategy = (
        "  3. **PICK COMMAND TYPE**: Start with `task` or `subgoal` to orient the robot.\n"
        "     Use `spatial_command` or `object_affordance` for phase-level guidance.\n"
        "     Only fall back to `low_level_motion` / `low_level_motion_with_numbers`\n"
        "     if the robot is stuck after multiple higher-level attempts.\n"
        )

    system_text = (
        "You are a high-level robot planner for a tabletop manipulation task.\n\n"
        "You observe the current scene through an RGB image and generate a language\n"
        "command that steers a low-level visuomotor policy (OpenVLA-OFT) to execute\n"
        "the next motion chunk (8 simulation steps).\n\n"
        + COMMAND_TYPES_DOC + "\n\n"
        + AXIS_CONVENTION_DOC + "\n\n"
        "## Your job at each step\n\n"
        "You receive:\n"
        "  1. The task description\n"
        "  2. The current observation image\n"
        "  3. Recent history: previous (command_type, command, observation) entries\n\n"
        "Steps:\n"
        "  1. **OBSERVE**: Where is the gripper?  Where are the target objects?\n"
        "  2. **ASSESS PROGRESS**: What phase is the robot in (approach / grasp / lift / transport / place)?\n"
        + strategy +
        "  4. **GENERATE COMMAND**: Write the command string exactly as the model expects.\n"
        "     Remember the axis swap for direction-word commands (see above).\n"
        f"  NOTE: You are ONLY allowed to use these command types: {allowed_str}\n\n"
        "  5. **OUTPUT** as JSON:\n\n"
        "```json\n"
        "{\n"
        '  "observation": "brief description of what you see",\n'
        '  "phase": "approach|grasp|lift|transport|place|done",\n'
        '  "reasoning": "why this command",\n'
        f'  "command_type": "one of: {allowed_str}",\n'
        '  "command": "the command string",\n'
        '  "task_complete": false\n'
        "}\n"
        "```\n\n"
        "Set `task_complete` to true ONLY when you can unambiguously confirm the task is "
        "fully done — e.g. the object is clearly resting in its target location and the "
        "gripper has released it. Be conservative: if there is any doubt (object partially "
        "placed, gripper still touching, target not visible), set `task_complete` to false "
        "and continue. It is much better to over-execute than to stop early.\n"
    )

    icl_content = build_icl_content(prompt_examples_dir)
    return system_text, icl_content


# =============================================================================
# GPT command generation
# =============================================================================

def query_gpt_command(
    client,
    gpt_model: str,
    system_text: str,
    icl_content: list,
    task_description: str,
    obs_img: np.ndarray,
    obs_raw: dict,
    history: List[dict],
    pinned_command_type: Optional[str] = None,
    chunk_idx: int = 0,
    allowed_types: Optional[List[str]] = None,
    max_retries: int = 3,
) -> dict:
    """Query GPT to get the next command.

    Args:
        history: List of {"command_type": str, "command": str, "obs_b64": str} entries.
        pinned_command_type: If not None/"auto", override the command type choice.
        chunk_idx: Current chunk index; chunk 0 is forced to task or subgoal.

    Returns:
        Parsed JSON dict with keys: command_type, command, reasoning, task_complete, ...
    """
    # ── Build the messages ────────────────────────────────────────────────────
    messages = [{"role": "system", "content": system_text}]

    # ICL examples as first user turn (cached images)
    if icl_content:
        messages.append({"role": "user", "content": icl_content})
        messages.append({"role": "assistant", "content":
                         "Understood. I have studied the in-context examples and the command conventions."})

    # Current query
    user_content: list = [
        {"type": "text", "text": f"Task: {task_description}"},
    ]

    # History of recent chunks
    if history:
        hist_text = "Recent history (oldest first):\n"
        for i, h in enumerate(history):
            hist_text += (
                f"  Chunk -{len(history)-i}: "
                f"[{h['command_type']}] \"{h['command']}\"\n"
            )
        user_content.append({"type": "text", "text": hist_text})
        # Show last observation image from history
        if history[-1].get("obs_b64"):
            user_content.append({"type": "text", "text": "Previous observation:"})
            user_content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{history[-1]['obs_b64']}",
                    "detail": "low",
                },
            })

    user_content.append({"type": "text", "text": "Current observation:"})
    user_content.append({
        "type": "image_url",
        "image_url": {
            "url": f"data:image/png;base64,{encode_image_to_base64(obs_img)}",
            "detail": "low",
        },
    })
    user_content.append({
        "type": "text",
        "text": f"Proprioceptive state:\n{format_proprio(obs_raw)}",
    })

    if pinned_command_type and pinned_command_type != "auto":
        user_content.append({
            "type": "text",
            "text": f"Use command_type: \"{pinned_command_type}\" for this query.",
        })
    elif chunk_idx == 0:
        user_content.append({
            "type": "text",
            "text": (
                "This is the FIRST chunk of the episode. "
                "You MUST use command_type \"task\" or \"subgoal\" to orient the robot. "
                "Do NOT use spatial_command, low_level_motion, or low_level_motion_with_numbers yet."
            ),
        })

    user_content.append({
        "type": "text",
        "text": "Generate the next command. Respond with a JSON code block.",
    })
    messages.append({"role": "user", "content": user_content})

    # ── Call GPT ──────────────────────────────────────────────────────────────
    ALL_TYPES = {"task", "subgoal", "spatial_command",
                 "object_affordance", "low_level_motion", "low_level_motion_with_numbers"}
    VALID_TYPES = set(allowed_types) if allowed_types else ALL_TYPES
    fallback = {
        "command_type": "task",
        "command": task_description,
        "reasoning": "GPT failed",
        "task_complete": False,
        "error": "max retries exceeded",
    }

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=gpt_model,
                messages=messages,
                temperature=0.0,
                max_completion_tokens=512,
            )
            raw = response.choices[0].message.content.strip()
            json_match = re.search(r"```(?:json)?\s*\n?([\s\S]*?)```", raw)
            if json_match:
                parsed = json.loads(json_match.group(1).strip())
            else:
                parsed = json.loads(re.search(r"(\{[\s\S]*\})", raw).group(1).strip())

            cmd_type = parsed.get("command_type", "").strip()
            command  = parsed.get("command", "").strip()
            if cmd_type not in VALID_TYPES:
                raise ValueError(f"Unknown command_type: {cmd_type!r}")
            if not command:
                raise ValueError("Empty command string")
            if pinned_command_type and pinned_command_type != "auto":
                parsed["command_type"] = pinned_command_type
            elif cmd_type not in VALID_TYPES:
                # GPT used a disallowed type — fall back to highest-level allowed
                fallback_type = next(
                    (t for t in ["task", "subgoal", "low_level_motion"] if t in VALID_TYPES),
                    list(VALID_TYPES)[0]
                )
                parsed["command_type"] = fallback_type
                if fallback_type == "task":
                    parsed["command"] = task_description
            elif chunk_idx == 0 and cmd_type not in ("task", "subgoal"):
                # Force high-level command on first chunk
                first_allowed = next(
                    (t for t in ["task", "subgoal"] if t in VALID_TYPES), None
                )
                if first_allowed:
                    parsed["command_type"] = first_allowed
                    if first_allowed == "task":
                        parsed["command"] = task_description

            return parsed

        except Exception as e:
            if attempt < max_retries - 1:
                logger.warning(f"GPT command attempt {attempt+1} failed: {e}. Retrying...")
                time.sleep(2 ** attempt)
            else:
                logger.error(f"GPT command failed after {max_retries} retries: {e}")
                return fallback

    return fallback


# =============================================================================
# Frame annotation
# =============================================================================

# Command type → color (BGR-style but we use RGB for PIL)
CMD_TYPE_COLORS = {
    "task":                          (100, 180, 255),   # blue
    "subgoal":                       (100, 220, 100),   # green
    "spatial_command":               (255, 200,  80),   # yellow
    "object_affordance":             (255, 140,  60),   # orange
    "low_level_motion":              (220, 100, 220),   # purple
    "low_level_motion_with_numbers": (255,  80,  80),   # red
}


def _wrap_text(text: str, max_chars: int) -> List[str]:
    """Wrap text to max_chars per line."""
    words = text.split()
    lines, cur = [], ""
    for w in words:
        if len(cur) + len(w) + 1 > max_chars:
            if cur:
                lines.append(cur)
            cur = w
        else:
            cur = (cur + " " + w).strip()
    if cur:
        lines.append(cur)
    return lines or [""]


def annotate_frame(
    img: np.ndarray,
    gpt_info: dict,
    step: int,
    chunk: int,
    panel_width: int = 320,
) -> np.ndarray:
    """Create a side-by-side frame: robot view on left, GPT info panel on right.

    Args:
        img: HxWx3 uint8 robot observation.
        gpt_info: dict with keys command_type, command, reasoning, phase.
        step: current simulation step.
        chunk: current GPT chunk index.
        panel_width: width of the text panel in pixels.

    Returns:
        HxW2x3 composite frame.
    """
    from PIL import Image as PILImage, ImageDraw, ImageFont

    h, w = img.shape[:2]
    robot_pil = PILImage.fromarray(img)

    # ── Text panel ────────────────────────────────────────────────────────────
    panel = PILImage.new("RGB", (panel_width, h), color=(20, 20, 20))
    draw = ImageDraw.Draw(panel)

    cmd_type  = gpt_info.get("command_type", "")
    command   = gpt_info.get("command", "")
    reasoning = gpt_info.get("reasoning", "")
    phase     = gpt_info.get("phase", "")
    color     = CMD_TYPE_COLORS.get(cmd_type, (200, 200, 200))

    margin = 8
    line_h = 14
    y = margin

    def draw_label(label: str, value: str, label_color=(180, 180, 180), value_color=(255, 255, 255), wrap=True):
        nonlocal y
        draw.text((margin, y), label, fill=label_color)
        y += line_h
        lines = _wrap_text(value, (panel_width - 2 * margin) // 7) if wrap else [value]
        for line in lines:
            draw.text((margin + 4, y), line, fill=value_color)
            y += line_h
        y += 4

    # Header
    draw.text((margin, y), f"Step {step}  |  Chunk {chunk}", fill=(150, 150, 150))
    y += line_h + 6

    # Command type badge
    draw.rectangle([margin, y, panel_width - margin, y + line_h + 4], fill=color)
    draw.text((margin + 4, y + 2), cmd_type.upper(), fill=(20, 20, 20))
    y += line_h + 10

    # Phase
    if phase:
        draw_label("PHASE", phase, value_color=(220, 220, 100))

    # Command
    draw_label("COMMAND", command, value_color=(255, 255, 255))

    # Separator
    draw.line([(margin, y), (panel_width - margin, y)], fill=(60, 60, 60), width=1)
    y += 6

    # Monitoring fields (if present)
    exec_assessment = gpt_info.get("execution_assessment", "")
    action = gpt_info.get("action", "")
    if exec_assessment:
        assess_color = {
            "following": (100, 220, 100),
            "partial": (255, 200, 80),
            "not_following": (255, 80, 80),
        }.get(exec_assessment, (200, 200, 200))
        draw_label("EXECUTION", exec_assessment, value_color=assess_color)
    if action:
        action_color = {
            "repeat": (150, 150, 220),
            "advance": (100, 220, 100),
            "switch_type": (255, 200, 80),
            "refine": (255, 160, 60),
        }.get(action, (200, 200, 200))
        draw_label("ACTION", action, value_color=action_color)

    # Reasoning
    draw_label("REASONING", reasoning, value_color=(180, 200, 180))

    # ── Composite ─────────────────────────────────────────────────────────────
    composite = PILImage.new("RGB", (w + panel_width, h))
    composite.paste(robot_pil, (0, 0))
    composite.paste(panel, (w, 0))
    return np.asarray(composite)


# =============================================================================
# Monitoring strategy: system prompt + query function
# =============================================================================

MONITORING_EXTRA_DOC = """\
## Execution monitoring (3-step decision process)

At each query you MUST follow these three steps in order:

### Step 1 — Monitor execution
Compare the previous observation (if any) with the current one.
Assess how well the robot followed the last command:
  - "following"     : robot moved clearly in the commanded direction / made progress
  - "partial"       : some movement but less than expected or slightly off-axis
  - "not_following" : robot barely moved, moved the wrong way, or got stuck

### Step 2 — Assess subgoal completion
Based on the current observation, decide if the CURRENT subgoal has been reached:
  - A subgoal is reached when the physical goal state it describes is visible.
  - Be conservative: only mark reached if clearly visible (same rule as task_complete).

### Step 3 — Choose action
Based on steps 1 and 2:

| execution_assessment | subgoal_reached | → action        | what to do                                      |
|---|---|---|---|
| following / partial  | false           | repeat          | Keep the same command (exact same string)        |
| following / partial  | true            | advance         | Reason about next subgoal, generate new command  |
| not_following        | false           | switch_type     | Try a different command_type for the same goal   |
| not_following        | true            | advance         | Subgoal done despite poor following — move on    |

**Important rules:**
- On `repeat`: return the EXACT same command string and command_type as before.
- On `switch_type`: pick a different command_type from the allowed list than what was just used.
  Do NOT switch again until you have tried the new type for at least 2 queries.
- On `advance`: set `next_subgoal` to the next step from the plan, or override it with a
  different description if the situation has changed unexpectedly. The plan is a reference —
  you may skip, repeat, or reword a subgoal if the physical state requires it.
- If you have been on the same subgoal for many queries without progress, choose `advance`
  and move to the next step even if uncertain.
"""


def build_monitoring_system_prompt(prompt_examples_dir: str, allowed_types: Optional[List[str]] = None) -> Tuple[str, list]:
    """System prompt for the monitoring strategy."""
    ALL_TYPES = ["task", "subgoal", "spatial_command", "object_affordance",
                 "low_level_motion", "low_level_motion_with_numbers"]
    if not allowed_types:
        allowed_types = ALL_TYPES
    allowed_str = "|".join(allowed_types)

    if set(allowed_types) == {"subgoal"}:
        strategy_note = "You must always use command_type `subgoal`.\n"
    elif len(allowed_types) <= 4 and "spatial_command" not in allowed_types:
        strategy_note = (
            "Prefer `task` or `subgoal` for high-level orientation. "
            "Only use `low_level_motion` / `low_level_motion_with_numbers` when switching type after failure.\n"
        )
    else:
        strategy_note = (
            "Prefer `task` or `subgoal` first. Use `spatial_command` / `object_affordance` for mid-phase guidance. "
            "Reserve `low_level_motion` / `low_level_motion_with_numbers` for correction after failure.\n"
        )

    system_text = (
        "You are a high-level robot planner with execution monitoring for a tabletop manipulation task.\n\n"
        "You observe the scene through RGB images and generate language commands that steer a "
        "low-level visuomotor policy (OpenVLA-OFT) to execute the next motion chunk.\n\n"
        + COMMAND_TYPES_DOC + "\n\n"
        + AXIS_CONVENTION_DOC + "\n\n"
        + MONITORING_EXTRA_DOC + "\n\n"
        f"Allowed command types for this model: {allowed_str}\n"
        f"{strategy_note}\n"
        f"NOTE: You are ONLY allowed to use these command types: {allowed_str}\n\n"
        "## Output format\n\n"
        "Respond with a JSON code block:\n"
        "```json\n"
        "{\n"
        '  "observation": "what you see in the current frame",\n'
        '  "execution_assessment": "following|partial|not_following",\n'
        '  "execution_reasoning": "why you assessed execution this way",\n'
        '  "subgoal_reached": false,\n'
        '  "current_subgoal": "the subgoal you are currently working on (from the plan)",\n'
        '  "action": "repeat|advance|switch_type",\n'
        '  "next_subgoal": null,\n'
        '  "reasoning": "why you chose this action and command",\n'
        f'  "command_type": "one of: {allowed_str}",\n'
        '  "command": "the command string",\n'
        '  "task_complete": false\n'
        "}\n"
        "```\n\n"
        "Set `task_complete` to true when you can confirm the full task is done.\n"
        "`next_subgoal`: only fill this when action is `advance`. Use the next step from the "
        "plan, or override with a different description if the situation has changed.\n"
    )

    icl_content = build_icl_content(prompt_examples_dir)
    return system_text, icl_content


def query_gpt_command_monitoring(
    client,
    gpt_model: str,
    system_text: str,
    icl_content: list,
    task_description: str,
    obs_img: np.ndarray,
    obs_raw: dict,
    history: List[dict],
    current_subgoal: str,
    subgoal_plan: Optional[List[str]] = None,
    current_subgoal_idx: int = 0,
    chunk_idx: int = 0,
    allowed_types: Optional[List[str]] = None,
    pinned_command_type: Optional[str] = None,
    max_retries: int = 3,
) -> dict:
    """Query GPT using the monitoring strategy.

    History entries expected to have keys:
        command_type, command, subgoal, execution_assessment, subgoal_reached, obs_b64
    """
    ALL_TYPES = {"task", "subgoal", "spatial_command",
                 "object_affordance", "low_level_motion", "low_level_motion_with_numbers"}
    VALID_TYPES = set(allowed_types) if allowed_types else ALL_TYPES

    messages = [{"role": "system", "content": system_text}]
    if icl_content:
        messages.append({"role": "user", "content": icl_content})
        messages.append({"role": "assistant", "content":
                         "Understood. I have studied the in-context examples and the monitoring protocol."})

    user_content: list = [{"type": "text", "text": f"Task: {task_description}"}]

    # History with execution trace
    if history:
        hist_text = "Execution history (oldest first):\n"
        for i, h in enumerate(history):
            hist_text += (
                f"  Chunk -{len(history)-i}: "
                f"[{h['command_type']}] \"{h['command'][:60]}\" | "
                f"subgoal: \"{h.get('subgoal','?')[:50]}\" | "
                f"execution: {h.get('execution_assessment','?')} | "
                f"subgoal_reached: {h.get('subgoal_reached','?')}\n"
            )
        user_content.append({"type": "text", "text": hist_text})
        if history[-1].get("obs_b64"):
            user_content.append({"type": "text", "text": "Previous observation:"})
            user_content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{history[-1]['obs_b64']}", "detail": "low"},
            })

    # Subgoal plan with current position marked
    if subgoal_plan:
        plan_lines = []
        for i, sg in enumerate(subgoal_plan):
            if i < current_subgoal_idx:
                plan_lines.append(f"  [done] {i}. {sg}")
            elif i == current_subgoal_idx:
                plan_lines.append(f"  → [current] {i}. {sg}")
            else:
                plan_lines.append(f"  [ ] {i}. {sg}")
        plan_text = "Subgoal plan:\n" + "\n".join(plan_lines)
        user_content.append({"type": "text", "text": plan_text})

    user_content.append({"type": "text", "text": f"Current subgoal: {current_subgoal}"})
    user_content.append({"type": "text", "text": "Current observation:"})
    user_content.append({
        "type": "image_url",
        "image_url": {"url": f"data:image/png;base64,{encode_image_to_base64(obs_img)}", "detail": "low"},
    })
    user_content.append({
        "type": "text",
        "text": f"Proprioceptive state:\n{format_proprio(obs_raw)}",
    })

    if chunk_idx == 0:
        user_content.append({
            "type": "text",
            "text": (
                "This is the FIRST chunk. No execution to monitor yet. "
                "Set execution_assessment to \"following\", action to \"advance\", "
                "and use command_type \"task\" or \"subgoal\" to orient the robot."
            ),
        })
    elif pinned_command_type and pinned_command_type != "auto":
        user_content.append({
            "type": "text",
            "text": f"Use command_type: \"{pinned_command_type}\" for this query.",
        })

    user_content.append({
        "type": "text",
        "text": "Follow the 3-step monitoring protocol and respond with a JSON code block.",
    })
    messages.append({"role": "user", "content": user_content})

    fallback = {
        "command_type": "task", "command": task_description,
        "execution_assessment": "not_following", "execution_reasoning": "GPT failed",
        "subgoal_reached": False, "current_subgoal": current_subgoal,
        "action": "repeat", "next_subgoal": None,
        "reasoning": "GPT failed", "task_complete": False, "error": "max retries exceeded",
    }

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=gpt_model, messages=messages, temperature=0.0, max_completion_tokens=700,
            )
            raw = response.choices[0].message.content.strip()
            json_match = re.search(r"```(?:json)?\s*\n?([\s\S]*?)```", raw)
            if json_match:
                parsed = json.loads(json_match.group(1).strip())
            else:
                parsed = json.loads(re.search(r"(\{[\s\S]*\})", raw).group(1).strip())

            cmd_type = parsed.get("command_type", "").strip()
            command  = parsed.get("command", "").strip()
            if not command:
                raise ValueError("Empty command string")

            # Enforce allowed types
            if cmd_type not in VALID_TYPES:
                fallback_type = next(
                    (t for t in ["task", "subgoal", "low_level_motion"] if t in VALID_TYPES),
                    list(VALID_TYPES)[0]
                )
                parsed["command_type"] = fallback_type
                if fallback_type == "task":
                    parsed["command"] = task_description

            # Force high-level on first chunk
            if chunk_idx == 0:
                first_allowed = next(
                    (t for t in ["task", "subgoal"] if t in VALID_TYPES), None
                )
                if first_allowed and parsed.get("command_type") not in ("task", "subgoal"):
                    parsed["command_type"] = first_allowed
                    if first_allowed == "task":
                        parsed["command"] = task_description

            return parsed

        except Exception as e:
            if attempt < max_retries - 1:
                logger.warning(f"GPT monitoring attempt {attempt+1} failed: {e}. Retrying...")
                time.sleep(2 ** attempt)
            else:
                logger.error(f"GPT monitoring failed after {max_retries} retries: {e}")
                return fallback

    return fallback


# =============================================================================
# GPT-commanded closed-loop episode runner
# =============================================================================

def run_episode_gpt_commands(
    cfg: GptCommandsEvalConfig,
    env,
    task_description: str,
    model,
    resize_size: int,
    gpt_client,
    system_text: str,
    icl_content: list,
    allowed_types: Optional[List[str]] = None,
    initial_state=None,
    processor=None,
    action_head=None,
    proprio_projector=None,
    noisy_action_projector=None,
) -> Tuple[bool, List[np.ndarray], List[dict]]:
    """Run a closed-loop episode where GPT generates steering commands."""
    env.reset()
    if initial_state is not None:
        obs = env.set_init_state(initial_state)
    else:
        obs = env.get_observation()

    max_steps = TASK_MAX_STEPS[cfg.task_suite_name]
    t = 0
    replay_images = []
    gpt_logs = []
    history: List[dict] = []  # recent (command_type, command, obs_b64) entries
    current_command_type = "task"
    current_command = task_description
    current_gpt_info: dict = {"command_type": "task", "command": task_description,
                               "reasoning": "initializing", "phase": "start"}
    chunk_idx = 0
    success = False

    # Stabilization steps
    for _ in range(cfg.num_steps_wait):
        obs, _, _, _ = env.step(get_libero_dummy_action(cfg.model_family))
        t += 1

    try:
        while t < max_steps + cfg.num_steps_wait:
            observation, img = prepare_observation(obs, resize_size)
            replay_images.append(annotate_frame(img, current_gpt_info,
                                                step=t - cfg.num_steps_wait,
                                                chunk=chunk_idx))

            # ── Re-query GPT every N chunks ──────────────────────────────────
            if chunk_idx % cfg.gpt_replan_every_n_chunks == 0:
                result = query_gpt_command(
                    client=gpt_client,
                    gpt_model=cfg.gpt_model,
                    system_text=system_text,
                    icl_content=icl_content,
                    task_description=task_description,
                    obs_img=img,
                    obs_raw=obs,
                    history=history[-cfg.gpt_history_length:],
                    pinned_command_type=(
                        cfg.gpt_command_type
                        if cfg.gpt_command_type != "auto" else None
                    ),
                    chunk_idx=chunk_idx,
                    allowed_types=allowed_types,
                )
                current_command_type = result.get("command_type", "task")
                current_command      = result.get("command", task_description)
                current_gpt_info = {
                    "command_type": current_command_type,
                    "command":      current_command,
                    "reasoning":    result.get("reasoning", ""),
                    "phase":        result.get("phase", ""),
                }

                gpt_logs.append({
                    "chunk": chunk_idx,
                    "step": t - cfg.num_steps_wait,
                    "command_type": current_command_type,
                    "command": current_command,
                    "reasoning": result.get("reasoning", ""),
                    "phase": result.get("phase", ""),
                    "task_complete": result.get("task_complete", False),
                })
                logger.info(
                    f"  [chunk {chunk_idx}] [{current_command_type}] \"{current_command}\""
                )

                # Update history
                history.append({
                    "command_type": current_command_type,
                    "command": current_command,
                    "obs_b64": encode_image_to_base64(img),
                })

                if result.get("task_complete", False):
                    logger.info("  GPT says task complete — stopping early.")
                    break

            # ── Execute one action chunk with the VLA ─────────────────────────
            with torch.no_grad():
                actions = get_action(
                    cfg=cfg,
                    model=model,
                    obs=observation,
                    task_label=current_command,
                    processor=processor,
                    action_head=action_head,
                    proprio_projector=proprio_projector,
                    noisy_action_projector=noisy_action_projector,
                    use_film=cfg.use_film,
                )

            chunk_idx += 1
            for action in actions:
                action = process_action(action, cfg.model_family)
                obs, _, done, _ = env.step(action.tolist())
                _, step_img = prepare_observation(obs, resize_size)
                replay_images.append(annotate_frame(step_img, current_gpt_info,
                                                    step=t - cfg.num_steps_wait,
                                                    chunk=chunk_idx))
                t += 1
                if done:
                    success = True
                    break
            if success:
                break

    except Exception as e:
        logger.error(f"GPT commands episode error: {e}", exc_info=True)

    return success, replay_images, gpt_logs


# =============================================================================
# Subgoal planning (monitoring strategy — called once before episode loop)
# =============================================================================

def plan_episode_subgoals(
    client,
    gpt_model: str,
    task_description: str,
    obs_img: np.ndarray,
    max_retries: int = 3,
) -> List[str]:
    """Ask GPT to generate the ordered subgoal plan for this episode.

    Called once before the main loop. Returns a list of subgoal strings
    (e.g. ["align gripper above bottle", "grasp bottle", "lift bottle", ...]).
    Falls back to [task_description] on failure.
    """
    system_text = (
        "You are a robot task planner for tabletop manipulation.\n\n"
        "Given a task description and an initial observation image, produce an ordered list of "
        "subgoals the robot must complete to finish the task.\n\n"
        "Each subgoal is a short phrase describing a concrete physical state to achieve "
        "(e.g. 'gripper aligned above the wine bottle', 'wine bottle grasped', "
        "'wine bottle lifted clear of the table', 'wine bottle placed on cabinet top').\n\n"
        "Rules:\n"
        "- 3 to 6 subgoals total\n"
        "- Each subgoal must be clearly visually checkable — describe the end state, not the motion\n"
        "- Cover the full task from start to finish\n\n"
        "Output ONLY a JSON code block:\n"
        "```json\n"
        "{\"subgoal_plan\": [\"subgoal 0\", \"subgoal 1\", ...]}\n"
        "```"
    )
    messages = [
        {"role": "system", "content": system_text},
        {"role": "user", "content": [
            {"type": "text", "text": f"Task: {task_description}"},
            {"type": "text", "text": "Initial observation:"},
            {"type": "image_url", "image_url": {
                "url": f"data:image/png;base64,{encode_image_to_base64(obs_img)}",
                "detail": "low",
            }},
            {"type": "text", "text": "Generate the subgoal plan."},
        ]},
    ]

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=gpt_model, messages=messages, temperature=0.0, max_completion_tokens=300,
            )
            raw = response.choices[0].message.content.strip()
            json_match = re.search(r"```(?:json)?\s*\n?([\s\S]*?)```", raw)
            if json_match:
                parsed = json.loads(json_match.group(1).strip())
            else:
                parsed = json.loads(re.search(r"(\{[\s\S]*\})", raw).group(1).strip())
            plan = parsed.get("subgoal_plan", [])
            if plan and isinstance(plan, list) and len(plan) > 0:
                return [str(s).strip() for s in plan]
            raise ValueError("Empty or invalid subgoal_plan")
        except Exception as e:
            if attempt < max_retries - 1:
                logger.warning(f"Subgoal planning attempt {attempt+1} failed: {e}. Retrying...")
                time.sleep(2 ** attempt)
            else:
                logger.error(f"Subgoal planning failed after {max_retries} retries: {e}")

    return [task_description]


# =============================================================================
# GPT-commanded episode runner — monitoring strategy
# =============================================================================

def run_episode_gpt_commands_monitoring(
    cfg: GptCommandsEvalConfig,
    env,
    task_description: str,
    model,
    resize_size: int,
    gpt_client,
    system_text: str,
    icl_content: list,
    allowed_types: Optional[List[str]] = None,
    initial_state=None,
    processor=None,
    action_head=None,
    proprio_projector=None,
    noisy_action_projector=None,
) -> Tuple[bool, List[np.ndarray], List[dict]]:
    """Run a closed-loop episode using the monitoring GPT strategy.

    Tracks current_subgoal and execution quality; enforces max_chunks_per_subgoal
    and min_chunks_before_type_switch limits.
    """
    env.reset()
    if initial_state is not None:
        obs = env.set_init_state(initial_state)
    else:
        obs = env.get_observation()

    max_steps = TASK_MAX_STEPS[cfg.task_suite_name]
    t = 0
    replay_images = []
    gpt_logs = []
    history: List[dict] = []
    current_command_type = "task"
    current_command = task_description
    chunks_on_subgoal = 0
    type_switch_cooldown = 0
    current_gpt_info: dict = {
        "command_type": "task", "command": task_description,
        "reasoning": "initializing", "phase": "start",
        "execution_assessment": "", "action": "advance",
    }
    chunk_idx = 0
    success = False

    # Stabilization steps
    for _ in range(cfg.num_steps_wait):
        obs, _, _, _ = env.step(get_libero_dummy_action(cfg.model_family))
        t += 1

    # Generate subgoal plan from the initial observation before the main loop
    _, img_for_plan = prepare_observation(obs, resize_size)
    subgoal_plan = plan_episode_subgoals(
        client=gpt_client,
        gpt_model=cfg.gpt_model,
        task_description=task_description,
        obs_img=img_for_plan,
    )
    current_subgoal_idx = 0
    current_subgoal = subgoal_plan[0]
    logger.info(f"  Subgoal plan ({len(subgoal_plan)} steps):")
    for i, sg in enumerate(subgoal_plan):
        logger.info(f"    {i}. {sg}")

    try:
        while t < max_steps + cfg.num_steps_wait:
            observation, img = prepare_observation(obs, resize_size)
            replay_images.append(annotate_frame(img, current_gpt_info,
                                                step=t - cfg.num_steps_wait,
                                                chunk=chunk_idx))

            # ── Re-query GPT every N chunks ──────────────────────────────────
            if chunk_idx % cfg.gpt_replan_every_n_chunks == 0:
                # Force advance if stuck too long on one subgoal
                force_advance = chunks_on_subgoal >= cfg.max_chunks_per_subgoal

                result = query_gpt_command_monitoring(
                    client=gpt_client,
                    gpt_model=cfg.gpt_model,
                    system_text=system_text,
                    icl_content=icl_content,
                    task_description=task_description,
                    obs_img=img,
                    obs_raw=obs,
                    history=history[-cfg.gpt_history_length:],
                    current_subgoal=current_subgoal,
                    subgoal_plan=subgoal_plan,
                    current_subgoal_idx=current_subgoal_idx,
                    chunk_idx=chunk_idx,
                    allowed_types=allowed_types,
                    pinned_command_type=(
                        cfg.gpt_command_type
                        if cfg.gpt_command_type != "auto" else None
                    ),
                )

                action = result.get("action", "repeat")
                exec_assessment = result.get("execution_assessment", "following")
                subgoal_reached = result.get("subgoal_reached", False)

                # Override action if stuck
                if force_advance:
                    action = "advance"
                    logger.info(
                        f"  [chunk {chunk_idx}] FORCE ADVANCE "
                        f"(stuck {chunks_on_subgoal} chunks on subgoal)"
                    )

                # Enforce min_chunks_before_type_switch cooldown
                if action == "switch_type" and type_switch_cooldown < cfg.min_chunks_before_type_switch:
                    action = "repeat"

                # Update subgoal state tracking
                if action == "advance":
                    current_subgoal_idx = min(current_subgoal_idx + 1, len(subgoal_plan) - 1)
                    # GPT may override the plan's next subgoal if the situation changed
                    next_sg = result.get("next_subgoal")
                    if next_sg and next_sg.strip():
                        current_subgoal = next_sg.strip()
                    else:
                        current_subgoal = subgoal_plan[current_subgoal_idx]
                    chunks_on_subgoal = 0
                    type_switch_cooldown = 0
                    logger.info(f"  → Advanced to subgoal [{current_subgoal_idx}]: {current_subgoal}")
                elif action == "switch_type":
                    type_switch_cooldown = 0
                    chunks_on_subgoal += 1
                else:
                    chunks_on_subgoal += 1
                    type_switch_cooldown += 1

                current_command_type = result.get("command_type", "task")
                current_command = result.get("command", task_description)
                current_gpt_info = {
                    "command_type": current_command_type,
                    "command": current_command,
                    "reasoning": result.get("reasoning", ""),
                    "phase": result.get("current_subgoal", current_subgoal),
                    "execution_assessment": exec_assessment,
                    "action": action,
                }

                gpt_logs.append({
                    "chunk": chunk_idx,
                    "step": t - cfg.num_steps_wait,
                    "command_type": current_command_type,
                    "command": current_command,
                    "reasoning": result.get("reasoning", ""),
                    "current_subgoal": current_subgoal,
                    "current_subgoal_idx": current_subgoal_idx,
                    "execution_assessment": exec_assessment,
                    "subgoal_reached": subgoal_reached,
                    "action": action,
                    "task_complete": result.get("task_complete", False),
                })
                logger.info(
                    f"  [chunk {chunk_idx}] [{exec_assessment}] action={action} "
                    f"[{current_command_type}] \"{current_command[:60]}\""
                )

                history.append({
                    "command_type": current_command_type,
                    "command": current_command,
                    "subgoal": result.get("current_subgoal", current_subgoal),
                    "execution_assessment": exec_assessment,
                    "subgoal_reached": subgoal_reached,
                    "obs_b64": encode_image_to_base64(img),
                })

                if result.get("task_complete", False):
                    logger.info("  GPT says task complete — stopping early.")
                    break

            # ── Execute one action chunk with the VLA ─────────────────────────
            with torch.no_grad():
                actions = get_action(
                    cfg=cfg, model=model, obs=observation,
                    task_label=current_command,
                    processor=processor, action_head=action_head,
                    proprio_projector=proprio_projector,
                    noisy_action_projector=noisy_action_projector,
                    use_film=cfg.use_film,
                )

            chunk_idx += 1
            for action_step in actions:
                action_step = process_action(action_step, cfg.model_family)
                obs, _, done, _ = env.step(action_step.tolist())
                _, step_img = prepare_observation(obs, resize_size)
                replay_images.append(annotate_frame(step_img, current_gpt_info,
                                                    step=t - cfg.num_steps_wait,
                                                    chunk=chunk_idx))
                t += 1
                if done:
                    success = True
                    break
            if success:
                break

    except Exception as e:
        logger.error(f"GPT monitoring episode error: {e}", exc_info=True)

    # Prepend the subgoal plan to the log so it's visible in the JSON
    if gpt_logs:
        gpt_logs[0]["subgoal_plan"] = subgoal_plan

    return success, replay_images, gpt_logs


# =============================================================================
# Baseline: full task description, no GPT
# =============================================================================

def run_episode_openvla_only(
    cfg: GptCommandsEvalConfig,
    env,
    task_description: str,
    model,
    resize_size: int,
    initial_state=None,
    processor=None,
    action_head=None,
    proprio_projector=None,
    noisy_action_projector=None,
) -> Tuple[bool, List[np.ndarray]]:
    """Baseline: run episode with full task description, no GPT."""
    from collections import deque
    env.reset()
    if initial_state is not None:
        obs = env.set_init_state(initial_state)
    else:
        obs = env.get_observation()

    max_steps = TASK_MAX_STEPS[cfg.task_suite_name]
    action_queue = deque()
    t = 0
    replay_images = []
    success = False

    try:
        while t < max_steps + cfg.num_steps_wait:
            if t < cfg.num_steps_wait:
                obs, _, _, _ = env.step(get_libero_dummy_action(cfg.model_family))
                t += 1
                continue

            observation, img = prepare_observation(obs, resize_size)
            replay_images.append(img)

            if len(action_queue) == 0:
                with torch.no_grad():
                    actions = get_action(
                        cfg=cfg, model=model, obs=observation, task_label=task_description,
                        processor=processor, action_head=action_head,
                        proprio_projector=proprio_projector,
                        noisy_action_projector=noisy_action_projector,
                        use_film=cfg.use_film,
                    )
                action_queue.extend(actions)

            action = action_queue.popleft()
            action = process_action(action, cfg.model_family)
            obs, _, done, _ = env.step(action.tolist())
            if done:
                success = True
                break
            t += 1

    except Exception as e:
        logger.error(f"openvla_only episode error: {e}")

    return success, replay_images


# =============================================================================
# Main evaluation loop
# =============================================================================

def run_evaluation(cfg: GptCommandsEvalConfig) -> None:
    set_seed_everywhere(cfg.seed)
    validate_config(cfg)

    model, action_head, proprio_projector, noisy_action_projector, processor = initialize_model(cfg)
    resize_size = get_image_resize_size(cfg)

    gpt_client = None
    system_text = ""
    icl_content: list = []
    if cfg.eval_mode == "gpt_commands":
        from openai import OpenAI
        gpt_client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        allowed_types = (
            None if cfg.allowed_command_types == "all"
            else [t.strip() for t in cfg.allowed_command_types.split(",")]
        )
        logger.info(f"Building system prompt from {cfg.prompt_examples_dir} ...")
        logger.info(f"Allowed command types: {allowed_types or 'all'}")
        logger.info(f"GPT strategy: {cfg.gpt_strategy}")
        if cfg.gpt_strategy == "monitoring":
            system_text, icl_content = build_monitoring_system_prompt(cfg.prompt_examples_dir, allowed_types)
        else:
            system_text, icl_content = build_system_prompt(cfg.prompt_examples_dir, allowed_types)
        logger.info(f"System prompt built. ICL content blocks: {len(icl_content)}")

    os.makedirs(cfg.video_dir, exist_ok=True)
    os.makedirs(cfg.local_log_dir, exist_ok=True)

    ckpt_name = os.path.basename(cfg.pretrained_checkpoint.rstrip("/"))
    run_id = f"GPTCMD-EVAL-{cfg.task_suite_name}-{ckpt_name[:40]}-{DATE_TIME}"
    log_path = os.path.join(cfg.local_log_dir, f"{run_id}.txt")
    log_file = open(log_path, "w")
    logger.info(f"Logging to {log_path}")

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[cfg.task_suite_name]()
    num_tasks = task_suite.n_tasks
    task_id_end = cfg.task_id_end if cfg.task_id_end > 0 else num_tasks

    # ── Per-mode aggregates ───────────────────────────────────────────────────
    mode_stats = {
        "gpt_commands":  {"total": 0, "success": 0},
        "openvla_only":  {"total": 0, "success": 0},
    }
    # command_type distribution
    cmd_type_counts: Dict[str, int] = defaultdict(int)

    # Per-task per-episode JSON log
    all_episode_results = []

    # ── Task loop ─────────────────────────────────────────────────────────────
    for task_id in tqdm.tqdm(range(cfg.task_id_start, task_id_end), desc="Tasks"):
        task = task_suite.get_task(task_id)
        initial_states = task_suite.get_task_init_states(task_id)
        env, task_description = get_libero_env(task, cfg.model_family, resolution=cfg.env_img_res)
        if cfg.custom_task_description:
            task_description = cfg.custom_task_description

        # Per-task output folder: <video_dir>/task{id:02d}_<slug>/
        task_slug = re.sub(r"[^a-z0-9]+", "_", task_description.lower()).strip("_")[:40]
        task_dir = os.path.join(cfg.video_dir, f"task{task_id:02d}_{task_slug}")
        os.makedirs(task_dir, exist_ok=True)

        log_message(f"\n{'='*60}", log_file)
        log_message(f"Task {task_id}: {task_description}", log_file)
        log_message(f"  output dir: {task_dir}", log_file)
        log_message(f"{'='*60}", log_file)

        num_trials = min(cfg.num_trials_per_task, len(initial_states))

        # ── GPT commands mode ─────────────────────────────────────────────────
        if cfg.eval_mode == "gpt_commands":
            for ep_idx in range(num_trials):
                initial_state = initial_states[ep_idx]
                logger.info(f"  [gpt_commands] task={task_id}, trial={ep_idx}")

                episode_runner = (
                    run_episode_gpt_commands_monitoring
                    if cfg.gpt_strategy == "monitoring"
                    else run_episode_gpt_commands
                )
                success, replay_images, gpt_logs = episode_runner(
                    cfg=cfg,
                    env=env,
                    task_description=task_description,
                    model=model,
                    resize_size=resize_size,
                    gpt_client=gpt_client,
                    system_text=system_text,
                    icl_content=icl_content,
                    allowed_types=allowed_types,
                    initial_state=initial_state,
                    processor=processor,
                    action_head=action_head,
                    proprio_projector=proprio_projector,
                    noisy_action_projector=noisy_action_projector,
                )

                mode_stats["gpt_commands"]["total"] += 1
                if success:
                    mode_stats["gpt_commands"]["success"] += 1
                for log in gpt_logs:
                    cmd_type_counts[log["command_type"]] += 1

                log_message(
                    f"  [gpt_commands] task={task_id} ep={ep_idx}: "
                    f"success={success}  chunks={len(gpt_logs)}\n"
                    f"  commands: {[(g['command_type'], g['command'][:40]) for g in gpt_logs]}",
                    log_file,
                )

                # Save video
                suffix = "success" if success else "fail"
                vpath = os.path.join(task_dir, f"ep{ep_idx}_{suffix}.mp4")
                try:
                    imageio.mimwrite(vpath, [np.asarray(x) for x in replay_images], fps=10)
                except Exception:
                    pass

                # Save per-episode GPT log JSON
                ep_log_path = os.path.join(task_dir, f"ep{ep_idx}_gpt_log.json")
                with open(ep_log_path, "w") as f:
                    json.dump({
                        "task_id": task_id,
                        "task_description": task_description,
                        "ep_idx": ep_idx,
                        "success": success,
                        "gpt_logs": gpt_logs,
                    }, f, indent=2)

                all_episode_results.append({
                    "mode": "gpt_commands",
                    "task_id": task_id,
                    "task_description": task_description,
                    "ep_idx": ep_idx,
                    "success": success,
                    "gpt_logs": gpt_logs,
                    "video_path": vpath,
                })

        # ── Baseline mode ─────────────────────────────────────────────────────
        if cfg.eval_mode == "openvla_only":
            for ep_idx in range(num_trials):
                initial_state = initial_states[ep_idx]
                success, replay_images = run_episode_openvla_only(
                    cfg=cfg, env=env, task_description=task_description,
                    model=model, resize_size=resize_size,
                    initial_state=initial_state,
                    processor=processor, action_head=action_head,
                    proprio_projector=proprio_projector,
                    noisy_action_projector=noisy_action_projector,
                )

                mode_stats["openvla_only"]["total"] += 1
                if success:
                    mode_stats["openvla_only"]["success"] += 1

                log_message(
                    f"  [openvla_only] task={task_id} ep={ep_idx}: success={success}",
                    log_file,
                )

                suffix = "success" if success else "fail"
                vpath = os.path.join(task_dir, f"ep{ep_idx}_{suffix}.mp4")
                try:
                    imageio.mimwrite(vpath, [np.asarray(x) for x in replay_images], fps=10)
                except Exception:
                    pass

                all_episode_results.append({
                    "mode": "openvla_only",
                    "task_id": task_id,
                    "task_description": task_description,
                    "ep_idx": ep_idx,
                    "success": success,
                    "video_path": vpath,
                })

        env.close()

    # ── Final summary ─────────────────────────────────────────────────────────
    log_message(f"\n{'='*60}", log_file)
    log_message("FINAL RESULTS", log_file)
    log_message(f"{'='*60}", log_file)

    for mode, stats in mode_stats.items():
        if stats["total"] > 0:
            rate = stats["success"] / stats["total"]
            log_message(f"  [{mode}] {stats['success']}/{stats['total']} = {rate:.3f}", log_file)

    if cmd_type_counts:
        total_cmds = sum(cmd_type_counts.values())
        log_message(f"\n  Command type distribution (total={total_cmds}):", log_file)
        for ct, cnt in sorted(cmd_type_counts.items(), key=lambda x: -x[1]):
            log_message(f"    {ct}: {cnt} ({100*cnt/total_cmds:.1f}%)", log_file)

    # Save results JSON
    results_path = os.path.join(cfg.local_log_dir, f"{run_id}_results.json")
    with open(results_path, "w") as f:
        json.dump({
            "mode_stats": mode_stats,
            "cmd_type_counts": dict(cmd_type_counts),
            "episode_results": all_episode_results,
        }, f, indent=2)
    log_message(f"\nResults saved to {results_path}", log_file)
    log_file.close()
    logger.info(f"Done. Results at {results_path}")


# =============================================================================
# Entry point
# =============================================================================

@draccus.wrap()
def main(cfg: GptCommandsEvalConfig) -> None:
    run_evaluation(cfg)


if __name__ == "__main__":
    main()
