"""
run_libero_eval_with_gpt.py

Closed-loop evaluation: runs LIBERO episodes end-to-end with OpenVLA-only,
GPT normal correction, and GPT residual correction — using the SAME initial
state for each mode so videos are directly comparable.

Saves:
  rollout_videos/{suite}/{task}/ep{N}/openvla_only.mp4
  rollout_videos/{suite}/{task}/ep{N}/gpt_normal.mp4
  rollout_videos/{suite}/{task}/ep{N}/gpt_residual.mp4
  rollout_videos/{suite}/{task}/ep{N}/side_by_side.mp4   (combined)

Usage:
    cd openvla-oft

    python experiments/robot/libero/run_libero_eval_with_gpt.py \
        --pretrained_checkpoint <path_or_hf_id> \
        --task_suite_name libero_spatial \
        --num_trials_per_task 3 \
        --gpt_mode all \
        --gpt_model gpt-4o \
        --video_dir ./rollout_videos

Set OPENAI_API_KEY environment variable before running.
"""

import base64
import io
import json
import logging
import os
import sys
import time
from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Optional

import draccus
import imageio
import numpy as np
import tqdm
from libero.libero import benchmark
from PIL import Image, ImageDraw, ImageFont

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
    TaskSuite,
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


# ---------------------------------------------------------------------------
# GPT correction
# ---------------------------------------------------------------------------

ACTION_DESCRIPTION = """\
Each action is a 7-dimensional vector: [dx, dy, dz, droll, dpitch, dyaw, gripper]

Each action step represents a small delta applied to the robot's end-effector. \
An action chunk contains 8 consecutive steps executed in sequence. Values are \
per-step deltas (NOT cumulative over the chunk).

Dimension details (per-step values):
  [0] dx (delta_x):      +forward / -back.
  [1] dy (delta_y):      +left    / -right.
  [2] dz (delta_z):      +up      / -down.
  [3] droll (delta_roll):   +tilt left  / -tilt right.
  [4] dpitch (delta_pitch): +tilt back  / -tilt forward.
  [5] dyaw (delta_yaw):    +rotate CCW / -rotate CW.
  [6] gripper:           >= 0.5 = OPEN, < 0.5 = CLOSE.

Magnitude reference from LIBERO dataset (per-step values, NOT meters — these are \
the robot's native action units):
  Position dims (dx, dy, dz):
    Small motion:  0.01–0.05 per step  (gentle nudge)
    Medium motion: 0.05–0.20 per step  (typical reaching/moving)
    Large motion:  0.20–0.50 per step  (fast repositioning)
    Dataset mean:  |dx|≈0.05, |dy|≈0.03, |dz|≈0.15    std ≈ 0.35–0.49
    Hard limit:    [-0.94, +0.94]
  Rotation dims (droll, dpitch, dyaw):
    Small rotation: 0.005–0.02 per step
    Medium rotation: 0.02–0.05 per step
    Large rotation: 0.05–0.15 per step
    Dataset mean: |droll|≈0.003, |dpitch|≈0.03, |dyaw|≈0.03    std ≈ 0.06–0.10
    Hard limit: [-0.38, +0.38]
  Gripper: 0.0 (CLOSE) or 1.0 (OPEN). Binary — use exactly 0.0 or 1.0.

IMPORTANT — values that are too small (e.g., 0.001 for position) will produce \
no visible motion. Values that are too large (e.g., > 0.5 per step) are outside \
the training distribution and will cause erratic behavior.

Gripper guidelines:
  - Keep gripper OPEN (1.0) while approaching and reaching toward an object.
  - CLOSE the gripper (0.0) only when it is positioned around the object and ready to grasp.
  - Keep gripper CLOSED (0.0) while transporting/lifting/moving the grasped object.
  - OPEN the gripper (1.0) when releasing/placing the object at the target location.\
"""

ICL_EXAMPLES = """\

## In-context examples

Below are examples showing how natural-language motion plans map to action chunks. \
The action values are in the robot's native action units (NOT meters).

### Example 1 — Approach: reach forward and down toward object (gripper open)
Scene: The gripper is above and behind a bottle on the table. Move forward \
and down to get closer while staying open.
Plan:
  - Forward/Back: move forward (medium) → dx ≈ +0.15 per step
  - Left/Right: no adjustment → dy ≈ 0
  - Up/Down: move down (small) → dz ≈ -0.08 per step
  - Rotations: none needed → droll, dpitch, dyaw ≈ 0
  - Gripper: keep open → 1.0
Action chunk:
  [[+0.15, 0.0, -0.08, 0.0, 0.0, 0.0, 1.0],
   [+0.15, 0.0, -0.08, 0.0, 0.0, 0.0, 1.0],
   [+0.15, 0.0, -0.08, 0.0, 0.0, 0.0, 1.0],
   [+0.15, 0.0, -0.08, 0.0, 0.0, 0.0, 1.0],
   [+0.15, 0.0, -0.08, 0.0, 0.0, 0.0, 1.0],
   [+0.15, 0.0, -0.08, 0.0, 0.0, 0.0, 1.0],
   [+0.15, 0.0, -0.08, 0.0, 0.0, 0.0, 1.0],
   [+0.15, 0.0, -0.08, 0.0, 0.0, 0.0, 1.0]]

### Example 2 — Grasp: close gripper around object
Scene: The gripper is open and positioned around a bowl. Close it to grasp. \
Minimal position change needed.
Plan:
  - Forward/Back: tiny nudge → dx ≈ +0.02
  - Left/Right: none → dy ≈ 0
  - Up/Down: tiny down → dz ≈ -0.02
  - Rotations: none → 0
  - Gripper: close → 0.0
Action chunk:
  [[+0.02, 0.0, -0.02, 0.0, 0.0, 0.0, 0.0],
   [+0.02, 0.0, -0.02, 0.0, 0.0, 0.0, 0.0],
   [+0.02, 0.0, -0.02, 0.0, 0.0, 0.0, 0.0],
   [+0.02, 0.0, -0.02, 0.0, 0.0, 0.0, 0.0],
   [+0.02, 0.0, -0.02, 0.0, 0.0, 0.0, 0.0],
   [+0.02, 0.0, -0.02, 0.0, 0.0, 0.0, 0.0],
   [+0.02, 0.0, -0.02, 0.0, 0.0, 0.0, 0.0],
   [+0.02, 0.0, -0.02, 0.0, 0.0, 0.0, 0.0]]

### Example 3 — Lift: raise grasped object straight up
Scene: The gripper has just grasped a cup. Lift it straight up.
Plan:
  - Forward/Back: none → dx ≈ 0
  - Left/Right: none → dy ≈ 0
  - Up/Down: move up (medium) → dz ≈ +0.12 per step
  - Rotations: slight tilt back to stabilize → dpitch ≈ +0.03
  - Gripper: keep closed → 0.0
Action chunk:
  [[0.0, 0.0, +0.12, 0.0, +0.03, 0.0, 0.0],
   [0.0, 0.0, +0.12, 0.0, +0.03, 0.0, 0.0],
   [0.0, 0.0, +0.12, 0.0, +0.03, 0.0, 0.0],
   [0.0, 0.0, +0.12, 0.0, +0.03, 0.0, 0.0],
   [0.0, 0.0, +0.12, 0.0, +0.03, 0.0, 0.0],
   [0.0, 0.0, +0.12, 0.0, +0.03, 0.0, 0.0],
   [0.0, 0.0, +0.12, 0.0, +0.03, 0.0, 0.0],
   [0.0, 0.0, +0.12, 0.0, +0.03, 0.0, 0.0]]

### Example 4 — Transport: move object left and forward while holding
Scene: The robot is holding a plate and needs to move it left and forward \
toward the target location.
Plan:
  - Forward/Back: forward (medium) → dx ≈ +0.10
  - Left/Right: left (medium) → dy ≈ +0.08
  - Up/Down: maintain height → dz ≈ 0
  - Rotations: none → 0
  - Gripper: keep closed → 0.0
Action chunk:
  [[+0.10, +0.08, 0.0, 0.0, 0.0, 0.0, 0.0],
   [+0.10, +0.08, 0.0, 0.0, 0.0, 0.0, 0.0],
   [+0.10, +0.08, 0.0, 0.0, 0.0, 0.0, 0.0],
   [+0.10, +0.08, 0.0, 0.0, 0.0, 0.0, 0.0],
   [+0.10, +0.08, 0.0, 0.0, 0.0, 0.0, 0.0],
   [+0.10, +0.08, 0.0, 0.0, 0.0, 0.0, 0.0],
   [+0.10, +0.08, 0.0, 0.0, 0.0, 0.0, 0.0],
   [+0.10, +0.08, 0.0, 0.0, 0.0, 0.0, 0.0]]

### Example 5 — Place: lower object and release
Scene: The robot is holding a bowl above the target plate. Lower it, then \
open the gripper to release.
Plan:
  - Forward/Back: slight forward → dx ≈ +0.03
  - Left/Right: none → dy ≈ 0
  - Up/Down: move down (medium) → dz ≈ -0.10
  - Tilt: slight tilt back → dpitch ≈ +0.02
  - Gripper: closed during descent, then open to release at the end
Action chunk:
  [[+0.03, 0.0, -0.10, 0.0, +0.02, 0.0, 0.0],
   [+0.03, 0.0, -0.10, 0.0, +0.02, 0.0, 0.0],
   [+0.03, 0.0, -0.10, 0.0, +0.02, 0.0, 0.0],
   [+0.03, 0.0, -0.10, 0.0, +0.02, 0.0, 0.0],
   [+0.03, 0.0, -0.10, 0.0, +0.02, 0.0, 0.0],
   [+0.03, 0.0, -0.05, 0.0, 0.0, 0.0, 1.0],
   [+0.03, 0.0, -0.05, 0.0, 0.0, 0.0, 1.0],
   [+0.03, 0.0, -0.05, 0.0, 0.0, 0.0, 1.0]]\
"""

# ---------------------------------------------------------------------------
# Two variants of per-DOF planning instructions
# ---------------------------------------------------------------------------

DOF_PLAN_CATEGORICAL = """\
2. PLAN EACH DEGREE OF FREEDOM: Decide what the robot should do in just the \
NEXT 8 steps. For each degree of freedom, state the direction and approximate \
magnitude:
   - Forward (+dx) or Back (-dx)? Small (0.01–0.05), medium (0.05–0.20), or large (0.20–0.50)?
   - Left (+dy) or Right (-dy)? Small, medium, or large?
   - Up (+dz) or Down (-dz)? Small, medium, or large?
   - Tilt left (+droll) or Tilt right (-droll)? Small (0.005–0.02), medium (0.02–0.05), or large (0.05–0.15)?
   - Tilt back (+dpitch) or Tilt forward (-dpitch)? Small, medium, or large?
   - Rotate CCW (+dyaw) or Rotate CW (-dyaw)? Small, medium, or large?
   - Gripper: Open (1.0) or Close (0.0)?"""

DOF_PLAN_EXACT = """\
2. PLAN EACH DEGREE OF FREEDOM: Decide what the robot should do in just the \
NEXT 8 steps. For each degree of freedom, state the direction and choose an \
exact per-step value. Use the ranges below as a guide:
   - Forward (+dx) or Back (-dx)? Exact per-step value? (position range: 0.01–0.50)
   - Left (+dy) or Right (-dy)? Exact per-step value?
   - Up (+dz) or Down (-dz)? Exact per-step value?
   - Tilt left (+droll) or Tilt right (-droll)? Exact per-step value? (rotation range: 0.005–0.15)
   - Tilt back (+dpitch) or Tilt forward (-dpitch)? Exact per-step value?
   - Rotate CCW (+dyaw) or Rotate CW (-dyaw)? Exact per-step value?
   - Gripper: Open (1.0) or Close (0.0)?
   Write e.g. "Forward, dx = +0.15 per step" or "no motion needed, dx = 0"."""


def _build_system_prompt_normal(dof_plan: str) -> str:
    return (
        "You are a robot action correction system for a tabletop manipulation task.\n\n"
        "You receive:\n"
        "- Two RGB images from the robot's third-person camera:\n"
        '  "Previous observation" (BEFORE the last action chunk was executed, ~8 steps ago) '
        'and "Current observation" (AFTER the last action chunk — this is the result).\n'
        "- A task description (what the robot should accomplish).\n"
        "- A proposed 8-step action chunk from a learned policy (to be executed NEXT).\n\n"
        "IMPORTANT: An action chunk is only 8 small steps — it moves the gripper just a "
        "few centimeters. Think SHORT-HORIZON only. Plan for this one chunk, not the entire task.\n\n"
        + ACTION_DESCRIPTION + ICL_EXAMPLES + "\n"
        "Follow these steps:\n"
        "1. EFFECT OF LAST CHUNK: Compare the previous and current observations. "
        "What changed between the two images? Did the gripper move closer to the target "
        "object? Did it successfully grasp something? Did anything go wrong (missed the "
        "object, knocked something over, moved in the wrong direction)?\n"
        + dof_plan + "\n"
        "3. EVALUATE PROPOSED ACTIONS: A language summary of the proposed chunk is "
        "provided showing direction, magnitude category, and per-step value for each "
        'active dimension (e.g., "move forward (medium, 0.150/step), open gripper"). '
        "Compare this summary against the current observation image and your per-DOF "
        "plan. Does the described motion make sense given what you see? Check each "
        "direction, magnitude, and the gripper state. Identify any mismatches.\n"
        "4. CONVERT & OUTPUT: If the proposed actions match your plan, output them "
        "unchanged. If not, convert your per-DOF plan into corrected action values "
        "using the magnitude reference above and output the corrected actions.\n\n"
        "Output format: Write your analysis, then output the JSON inside a ```json "
        "code block. The JSON must be a flat array of 8 arrays, each with exactly 7 "
        "numbers. No comments, no objects, no extra keys. Example:\n"
        "```json\n"
        "[[0.15, -0.08, 0.0, 0.0, 0.0, 0.0, 1.0], [0.15, -0.08, 0.0, 0.0, 0.0, 0.0, 1.0], ...]\n"
        "```"
    )


def _build_system_prompt_residual(dof_plan: str) -> str:
    return (
        "You are a robot action residual correction system for a tabletop manipulation task.\n\n"
        "You receive:\n"
        "- Two RGB images from the robot's third-person camera:\n"
        '  "Previous observation" (BEFORE the last action chunk was executed, ~8 steps ago) '
        'and "Current observation" (AFTER the last action chunk — this is the result).\n'
        "- A task description (what the robot should accomplish).\n"
        "- A proposed 8-step action chunk from a learned policy (to be executed NEXT).\n\n"
        "IMPORTANT: An action chunk is only 8 small steps — it moves the gripper just a "
        "few centimeters. Think SHORT-HORIZON only. Plan for this one chunk, not the entire task.\n\n"
        + ACTION_DESCRIPTION + ICL_EXAMPLES + "\n"
        "The corrected action will be computed as:\n"
        "  corrected_action = proposed_action + your_residual\n\n"
        "Follow these steps:\n"
        "1. EFFECT OF LAST CHUNK: Compare the previous and current observations. "
        "What changed between the two images? Did the gripper move closer to the target "
        "object? Did it successfully grasp something? Did anything go wrong (missed the "
        "object, knocked something over, moved in the wrong direction)?\n"
        + dof_plan + "\n"
        "3. EVALUATE PROPOSED ACTIONS: A language summary of the proposed chunk is "
        "provided showing direction, magnitude category, and per-step value for each "
        'active dimension (e.g., "move forward (medium, 0.150/step), open gripper"). '
        "Compare this summary against the current observation image and your per-DOF "
        "plan. Does the described motion make sense given what you see? Check each "
        "direction, magnitude, and the gripper state. Identify any mismatches.\n"
        "4. COMPUTE RESIDUALS: If the proposed actions match your plan, output all zeros. "
        "If not, compute the residual correction for each mismatched dimension. "
        "Keep residuals small (position: [-0.03, 0.03], rotation: [-0.05, 0.05]).\n\n"
        "Output format: Write your analysis, then output the JSON inside a ```json "
        "code block. The JSON must be a flat array of 8 arrays, each with exactly 7 "
        "numbers representing the RESIDUAL (not the final action). No comments, no "
        "objects, no extra keys. Example:\n"
        "```json\n"
        "[[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], ...]\n"
        "```"
    )


def _build_system_prompt_direct(dof_plan: str) -> str:
    return (
        "You are a robot action generation system for a tabletop manipulation task.\n\n"
        "You receive:\n"
        "- Two RGB images from the robot's third-person camera:\n"
        '  "Previous observation" (BEFORE the last action chunk was executed, ~8 steps ago) '
        'and "Current observation" (AFTER the last action chunk — this is the result).\n'
        "- A task description (what the robot should accomplish).\n\n"
        "You must generate an 8-step action chunk from scratch based on the visual "
        "observations and the task goal. There is NO proposed action from a learned "
        "policy — you are the sole controller.\n\n"
        "IMPORTANT: An action chunk is only 8 small steps — it moves the gripper just a "
        "few centimeters. Think SHORT-HORIZON only. Plan for this one chunk, not the entire task.\n\n"
        + ACTION_DESCRIPTION + ICL_EXAMPLES + "\n"
        "Follow these steps:\n"
        "1. OBSERVE: Describe what you see in the current image. Where is the gripper? "
        "Where are the relevant objects? What is the gripper holding (if anything)?\n"
        "2. EFFECT OF LAST CHUNK: If a previous observation is provided, compare it "
        "with the current one. What changed? Did the robot make progress toward the task?\n"
        "3. " + dof_plan[3:] + "\n"  # strip leading "2. " since this is step 3 here
        "4. CONVERT & OUTPUT: Convert your per-DOF plan into action values using the "
        "magnitude reference above. Output an 8-step action chunk.\n\n"
        "Output format: Write your analysis, then output the JSON inside a ```json "
        "code block. The JSON must be a flat array of 8 arrays, each with exactly 7 "
        "numbers. No comments, no objects, no extra keys. Example:\n"
        "```json\n"
        "[[0.15, -0.08, 0.0, 0.0, 0.0, 0.0, 1.0], [0.15, -0.08, 0.0, 0.0, 0.0, 0.0, 1.0], ...]\n"
        "```"
    )


def get_system_prompt(mode: str, exact_values: bool = False) -> str:
    """Get the system prompt for a given mode and value specification style.

    Args:
        mode: "normal", "residual", or "direct"
        exact_values: If True, ask GPT for exact per-step values.
                      If False, ask for small/medium/large categories.
    """
    dof_plan = DOF_PLAN_EXACT if exact_values else DOF_PLAN_CATEGORICAL
    if mode == "normal":
        return _build_system_prompt_normal(dof_plan)
    elif mode == "residual":
        return _build_system_prompt_residual(dof_plan)
    elif mode == "direct":
        return _build_system_prompt_direct(dof_plan)
    else:
        raise ValueError(f"Unknown mode: {mode}")


def encode_image_to_base64(img: np.ndarray) -> str:
    pil_img = Image.fromarray(img)
    buffer = io.BytesIO()
    pil_img.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def _mag_label(val: float, thresholds: tuple) -> str:
    """Classify an absolute value as small/medium/large."""
    small, med = thresholds
    if val < small:
        return "small"
    elif val < med:
        return "medium"
    return "large"


def actions_to_language(actions: list) -> str:
    """Convert an 8-step action chunk to a human-readable language description.

    Uses the LIBERO dataset conventions (native action units, NOT meters):
      - Position (dims 0-2): mean per-step value, classified as small/medium/large
      - Rotation (dims 3-5): mean per-step value, classified as small/medium/large
      - Gripper (dim 6): last step value, >= 0.5 = open, < 0.5 = close
    """
    arr = np.array([a if not hasattr(a, "tolist") else a for a in actions])  # (8, 7)

    # Mean per-step values (what GPT should think about per step)
    dx = float(arr[:, 0].mean())
    dy = float(arr[:, 1].mean())
    dz = float(arr[:, 2].mean())
    droll = float(arr[:, 3].mean())
    dpitch = float(arr[:, 4].mean())
    dyaw = float(arr[:, 5].mean())
    grip = float(arr[-1, 6])  # last step

    # Position thresholds: small < 0.05, medium < 0.20, large >= 0.20
    pos_thresh = (0.05, 0.20)
    # Rotation thresholds: small < 0.02, medium < 0.05, large >= 0.05
    rot_thresh = (0.02, 0.05)

    parts = []
    if abs(dx) >= 0.01:
        parts.append(f"move {'forward' if dx > 0 else 'back'} ({_mag_label(abs(dx), pos_thresh)}, {abs(dx):.3f}/step)")
    if abs(dy) >= 0.01:
        parts.append(f"move {'left' if dy > 0 else 'right'} ({_mag_label(abs(dy), pos_thresh)}, {abs(dy):.3f}/step)")
    if abs(dz) >= 0.01:
        parts.append(f"move {'up' if dz > 0 else 'down'} ({_mag_label(abs(dz), pos_thresh)}, {abs(dz):.3f}/step)")
    if abs(droll) >= 0.005:
        parts.append(f"tilt {'left' if droll > 0 else 'right'} ({_mag_label(abs(droll), rot_thresh)}, {abs(droll):.3f}/step)")
    if abs(dpitch) >= 0.005:
        parts.append(f"tilt {'back' if dpitch > 0 else 'forward'} ({_mag_label(abs(dpitch), rot_thresh)}, {abs(dpitch):.3f}/step)")
    if abs(dyaw) >= 0.005:
        parts.append(f"rotate {'CCW' if dyaw > 0 else 'CW'} ({_mag_label(abs(dyaw), rot_thresh)}, {abs(dyaw):.3f}/step)")
    parts.append("open gripper" if grip >= 0.5 else "close gripper")

    return ", ".join(parts)


def save_obs_image(img: np.ndarray, path: str) -> None:
    """Save an observation image to disk as PNG."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    Image.fromarray(img).save(path)


ICL_EXAMPLES_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "..", "icl_examples")

# Cache for loaded ICL examples (loaded once, reused across calls)
_icl_cache: Optional[list] = None


def _load_icl_examples() -> list:
    """Load pre-extracted ICL examples (images + action chunks) from disk.

    Returns list of message pairs: [(user_content, assistant_content), ...]
    """
    global _icl_cache
    if _icl_cache is not None:
        return _icl_cache

    meta_path = os.path.join(ICL_EXAMPLES_DIR, "icl_metadata.json")
    if not os.path.exists(meta_path):
        logger.warning(f"ICL examples not found at {meta_path}. Skipping image ICL.")
        _icl_cache = []
        return _icl_cache

    with open(meta_path) as f:
        metadata = json.load(f)

    pairs = []
    for phase_name in ["approach", "descend", "lift", "transport", "place"]:
        if phase_name not in metadata:
            continue
        ex = metadata[phase_name]
        img_path = os.path.join(ICL_EXAMPLES_DIR, f"icl_{phase_name}.png")
        if not os.path.exists(img_path):
            continue

        # Encode the image
        with open(img_path, "rb") as f:
            img_b64 = base64.b64encode(f.read()).decode("utf-8")

        # Build user message: observation + action chunk
        actions_str = json.dumps(
            [{"step": i, "action": a} for i, a in enumerate(ex["action_chunk"])],
            indent=2,
        )
        lang_summary = ex["lang"]

        user_content = [
            {"type": "text", "text": f"Task: put the wine bottle on the plate"},
            {"type": "text", "text": "(No previous observation — this is a demonstration example.)"},
            {"type": "text", "text": "Current observation:"},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}},
            {"type": "text", "text": (
                f"Proposed NEXT action chunk (language summary: {lang_summary}):\n"
                f"{actions_str}\n\nOutput corrected actions as JSON:"
            )},
        ]

        # Build assistant response: reasoning + JSON (this IS the ground truth, so output unchanged)
        chunk_json = json.dumps(ex["action_chunk"])
        mean = ex["mean_per_step"]
        assistant_content = (
            f"**1. EFFECT OF LAST CHUNK:** This is a demonstration example — {ex['desc']}.\n\n"
            f"**2. PLAN EACH DEGREE OF FREEDOM:**\n"
        )
        dim_names = ["dx", "dy", "dz", "droll", "dpitch", "dyaw"]
        dir_labels = [
            ("forward", "back"), ("left", "right"), ("up", "down"),
            ("tilt left", "tilt right"), ("tilt back", "tilt forward"),
            ("rotate CCW", "rotate CW"),
        ]
        pos_thresh = (0.05, 0.20)
        rot_thresh = (0.02, 0.05)
        for i, (name, (pos_dir, neg_dir)) in enumerate(zip(dim_names, dir_labels)):
            val = mean[i]
            thresh = pos_thresh if i < 3 else rot_thresh
            mag = _mag_label(abs(val), thresh)
            direction = pos_dir if val >= 0 else neg_dir
            if abs(val) >= (0.01 if i < 3 else 0.005):
                assistant_content += f"   - {direction} ({mag}, {abs(val):.3f}/step)\n"
            else:
                assistant_content += f"   - no significant {dim_names[i]} motion\n"
        grip_str = "open gripper (1.0)" if mean[6] >= 0.5 else "close gripper (0.0)"
        assistant_content += f"   - Gripper: {grip_str}\n\n"
        assistant_content += (
            f"**3. EVALUATE PROPOSED ACTIONS:** The proposed actions match the plan — "
            f"directions and magnitudes are consistent with the scene.\n\n"
            f"**4. CONVERT & OUTPUT:** Actions are correct, output unchanged.\n\n"
            f"```json\n{chunk_json}\n```"
        )

        pairs.append((user_content, assistant_content))

    _icl_cache = pairs
    logger.info(f"Loaded {len(pairs)} ICL image examples from {ICL_EXAMPLES_DIR}")
    return _icl_cache


def query_gpt_for_correction(
    client,
    task_description: str,
    curr_obs_img: np.ndarray,
    openvla_actions: list,
    prev_obs_img: Optional[np.ndarray],
    mode: str,
    gpt_model: str = "gpt-4o",
    max_retries: int = 3,
    save_dir: Optional[str] = None,
    chunk_idx: int = 0,
    exact_values: bool = False,
) -> list:
    """Query GPT and return corrected actions as list of np arrays."""
    system_prompt = get_system_prompt(mode, exact_values=exact_values)

    # Save observation images to disk for inspection
    if save_dir is not None:
        chunk_dir = os.path.join(save_dir, f"chunk_{chunk_idx:03d}")
        os.makedirs(chunk_dir, exist_ok=True)
        save_obs_image(curr_obs_img, os.path.join(chunk_dir, "current_obs.png"))
        if prev_obs_img is not None:
            save_obs_image(prev_obs_img, os.path.join(chunk_dir, "previous_obs.png"))

    # Convert proposed actions to language summary for GPT evaluation
    lang_summary = actions_to_language(openvla_actions)

    actions_formatted = [
        {"step": i, "action": [round(float(v), 5) for v in (a.tolist() if hasattr(a, "tolist") else a)]}
        for i, a in enumerate(openvla_actions)
    ]
    actions_str = json.dumps(actions_formatted, indent=2)

    content = [{"type": "text", "text": f"Task: {task_description}"}]

    if prev_obs_img is not None:
        content.append({"type": "text", "text": "Previous observation (BEFORE last action chunk):"})
        content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encode_image_to_base64(prev_obs_img)}"}})
    else:
        content.append({"type": "text", "text": "(No previous observation — this is the very first chunk of the episode.)"})

    content.append({"type": "text", "text": "Current observation (AFTER last action chunk — current state):"})
    content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encode_image_to_base64(curr_obs_img)}"}})

    suffix = "Output corrected actions as JSON:" if mode == "normal" else "Output residual corrections as JSON:"
    content.append({"type": "text", "text": (
        f"Proposed NEXT action chunk (language summary: {lang_summary}):\n"
        f"{actions_str}\n\n{suffix}"
    )})

    # Build messages: system prompt + ICL examples + actual query
    messages = [{"role": "system", "content": system_prompt}]

    # Add image-based ICL examples as few-shot user/assistant pairs
    icl_examples = _load_icl_examples()
    for user_content, assistant_content in icl_examples:
        messages.append({"role": "user", "content": user_content})
        messages.append({"role": "assistant", "content": assistant_content})

    # Add the actual query
    messages.append({"role": "user", "content": content})

    for attempt in range(max_retries):
        raw = ""
        try:
            response = client.chat.completions.create(
                model=gpt_model,
                messages=messages,
                temperature=0.0,
                max_tokens=2000,
            )
            raw = response.choices[0].message.content.strip()

            # Extract JSON from response — handle text + ```json ... ``` format
            import re
            json_match = re.search(r"```(?:json)?\s*\n?([\s\S]*?)```", raw)
            if json_match:
                json_str = json_match.group(1).strip()
            else:
                # Fallback: try to find a JSON array directly
                array_match = re.search(r"(\[[\s\S]*\])", raw)
                if array_match:
                    json_str = array_match.group(1).strip()
                else:
                    raise ValueError(f"No JSON found in response: {raw[:200]}")

            # Strip inline comments (// ...) that GPT sometimes adds
            json_str = re.sub(r"//[^\n]*", "", json_str)

            parsed = json.loads(json_str)
            assert isinstance(parsed, list) and len(parsed) == 8, f"Expected 8 actions, got {len(parsed)}"

            # Handle {"step": N, "action": [...]} format
            if isinstance(parsed[0], dict):
                parsed = [item["action"] for item in parsed]

            for i, row in enumerate(parsed):
                assert isinstance(row, list) and len(row) == 7, f"Action {i} has {len(row)} dims, expected 7"

            # Extract reasoning text (everything before the JSON block)
            reasoning = raw
            if json_match:
                reasoning = raw[:json_match.start()].strip()

            if mode == "normal":
                result_actions = [np.array(a) for a in parsed]
            else:
                result_actions = []
                for i, residual in enumerate(parsed):
                    orig = np.array(openvla_actions[i]) if not isinstance(openvla_actions[i], np.ndarray) else openvla_actions[i]
                    result_actions.append(orig + np.array(residual))

            return result_actions, reasoning, raw

        except Exception as e:
            if attempt < max_retries - 1:
                logger.warning(f"GPT attempt {attempt+1} failed: {e}. Raw: {raw[:300]}. Retrying...")
                time.sleep(2)
            else:
                logger.error(f"GPT failed after {max_retries} attempts: {e}. Raw: {raw[:300]}")
                return openvla_actions, f"PARSE_ERROR: {e}", raw


def query_gpt_direct(
    client,
    task_description: str,
    curr_obs_img: np.ndarray,
    prev_obs_img: Optional[np.ndarray],
    gpt_model: str = "gpt-4o",
    max_retries: int = 3,
    save_dir: Optional[str] = None,
    chunk_idx: int = 0,
    exact_values: bool = False,
) -> list:
    """Query GPT to generate an action chunk from scratch (no OpenVLA proposal)."""
    system_prompt = get_system_prompt("direct", exact_values=exact_values)

    # Save observation images to disk for inspection
    if save_dir is not None:
        chunk_dir = os.path.join(save_dir, f"chunk_{chunk_idx:03d}")
        os.makedirs(chunk_dir, exist_ok=True)
        save_obs_image(curr_obs_img, os.path.join(chunk_dir, "current_obs.png"))
        if prev_obs_img is not None:
            save_obs_image(prev_obs_img, os.path.join(chunk_dir, "previous_obs.png"))

    content = [{"type": "text", "text": f"Task: {task_description}"}]

    if prev_obs_img is not None:
        content.append({"type": "text", "text": "Previous observation (BEFORE last action chunk):"})
        content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encode_image_to_base64(prev_obs_img)}"}})
    else:
        content.append({"type": "text", "text": "(No previous observation — this is the very first chunk of the episode.)"})

    content.append({"type": "text", "text": "Current observation (AFTER last action chunk — current state):"})
    content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encode_image_to_base64(curr_obs_img)}"}})

    content.append({"type": "text", "text": "Generate the NEXT 8-step action chunk as JSON:"})

    # Build messages: system prompt + ICL examples + actual query
    messages = [{"role": "system", "content": system_prompt}]

    icl_examples = _load_icl_examples()
    for user_content, assistant_content in icl_examples:
        messages.append({"role": "user", "content": user_content})
        messages.append({"role": "assistant", "content": assistant_content})

    messages.append({"role": "user", "content": content})

    for attempt in range(max_retries):
        raw = ""
        try:
            response = client.chat.completions.create(
                model=gpt_model,
                messages=messages,
                temperature=0.0,
                max_tokens=2000,
            )
            raw = response.choices[0].message.content.strip()

            import re
            json_match = re.search(r"```(?:json)?\s*\n?([\s\S]*?)```", raw)
            if json_match:
                json_str = json_match.group(1).strip()
            else:
                array_match = re.search(r"(\[[\s\S]*\])", raw)
                if array_match:
                    json_str = array_match.group(1).strip()
                else:
                    raise ValueError(f"No JSON found in response: {raw[:200]}")

            json_str = re.sub(r"//[^\n]*", "", json_str)
            parsed = json.loads(json_str)
            assert isinstance(parsed, list) and len(parsed) == 8, f"Expected 8 actions, got {len(parsed)}"

            if isinstance(parsed[0], dict):
                parsed = [item["action"] for item in parsed]

            for i, row in enumerate(parsed):
                assert isinstance(row, list) and len(row) == 7, f"Action {i} has {len(row)} dims, expected 7"

            reasoning = raw
            if json_match:
                reasoning = raw[:json_match.start()].strip()

            result_actions = [np.array(a) for a in parsed]
            return result_actions, reasoning, raw

        except Exception as e:
            if attempt < max_retries - 1:
                logger.warning(f"GPT direct attempt {attempt+1} failed: {e}. Raw: {raw[:300]}. Retrying...")
                time.sleep(2)
            else:
                logger.error(f"GPT direct failed after {max_retries} attempts: {e}. Raw: {raw[:300]}")
                # Fallback: return zero actions (no motion) with open gripper
                zero_actions = [np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]) for _ in range(8)]
                return zero_actions, f"PARSE_ERROR: {e}", raw


# ---------------------------------------------------------------------------
# Video utilities
# ---------------------------------------------------------------------------

def add_text_overlay(img: np.ndarray, text: str, position="top") -> np.ndarray:
    """Add text label to an image frame. Returns a new image with a banner."""
    h, w = img.shape[:2]
    banner_h = 30
    canvas = np.zeros((h + banner_h, w, 3), dtype=np.uint8)

    # Draw banner background
    if "openvla_only" in text:
        banner_color = (50, 50, 180)   # blue
    elif "gpt_normal" in text:
        banner_color = (50, 160, 50)   # green
    elif "gpt_residual" in text:
        banner_color = (180, 100, 30)  # orange
    elif "gpt_direct" in text:
        banner_color = (160, 50, 160)  # purple
    else:
        banner_color = (80, 80, 80)

    canvas[:banner_h, :] = banner_color
    canvas[banner_h:, :] = img

    # Draw text using PIL for better rendering
    pil_img = Image.fromarray(canvas)
    draw = ImageDraw.Draw(pil_img)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
    except (IOError, OSError):
        font = ImageFont.load_default()
    draw.text((8, 5), text, fill=(255, 255, 255), font=font)
    return np.array(pil_img)


def save_video(frames: List[np.ndarray], path: str, fps: int = 30):
    """Save frames as MP4."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    writer = imageio.get_writer(path, fps=fps)
    for frame in frames:
        writer.append_data(frame)
    writer.close()


def save_side_by_side_video(
    all_frames: Dict[str, List[np.ndarray]],
    path: str,
    fps: int = 30,
):
    """
    Create a side-by-side video from multiple modes.
    all_frames: {mode_name: [frame0, frame1, ...]}
    Pads shorter videos with their last frame.
    """
    modes = list(all_frames.keys())
    if not modes:
        return

    # Find max length
    max_len = max(len(all_frames[m]) for m in modes)
    if max_len == 0:
        return

    # Pad shorter sequences with last frame
    padded = {}
    for m in modes:
        frames = all_frames[m]
        if len(frames) == 0:
            continue
        while len(frames) < max_len:
            frames.append(frames[-1])
        padded[m] = frames

    modes = list(padded.keys())
    if not modes:
        return

    os.makedirs(os.path.dirname(path), exist_ok=True)
    writer = imageio.get_writer(path, fps=fps)

    for t in range(max_len):
        panels = []
        for m in modes:
            frame = padded[m][t]
            labeled = add_text_overlay(frame, m)
            panels.append(labeled)

        # Make all panels the same height
        max_h = max(p.shape[0] for p in panels)
        max_w = max(p.shape[1] for p in panels)
        aligned = []
        for p in panels:
            if p.shape[0] < max_h or p.shape[1] < max_w:
                padded_frame = np.zeros((max_h, max_w, 3), dtype=np.uint8)
                padded_frame[:p.shape[0], :p.shape[1]] = p
                aligned.append(padded_frame)
            else:
                aligned.append(p)

        # Add 2px white separator between panels
        sep = np.ones((max_h, 2, 3), dtype=np.uint8) * 255
        combined_parts = []
        for i, panel in enumerate(aligned):
            if i > 0:
                combined_parts.append(sep)
            combined_parts.append(panel)
        combined = np.concatenate(combined_parts, axis=1)
        writer.append_data(combined)

    writer.close()


# ---------------------------------------------------------------------------
# Episode runner
# ---------------------------------------------------------------------------

def run_episode_with_mode(
    cfg,
    env,
    task_description,
    model,
    resize_size,
    gpt_client,
    eval_mode,
    processor=None,
    action_head=None,
    proprio_projector=None,
    noisy_action_projector=None,
    initial_state=None,
    log_file=None,
    save_dir=None,
):
    """Run a single episode with the specified evaluation mode."""
    env.reset()
    if initial_state is not None:
        obs = env.set_init_state(initial_state)
    else:
        obs = env.get_observation()

    action_queue = deque(maxlen=cfg.num_open_loop_steps)
    t = 0
    replay_images = []
    max_steps = TASK_MAX_STEPS[cfg.task_suite_name]
    success = False

    prev_obs_img = None
    gpt_calls = 0
    gpt_fallbacks = 0
    gpt_logs = []  # Collect GPT reasoning and actions per chunk

    try:
        while t < max_steps + cfg.num_steps_wait:
            if t < cfg.num_steps_wait:
                obs, reward, done, info = env.step(get_libero_dummy_action(cfg.model_family))
                t += 1
                continue

            observation, img = prepare_observation(obs, resize_size)
            replay_images.append(img)

            if len(action_queue) == 0:
                if eval_mode == "gpt_direct" and gpt_client is not None:
                    # GPT generates actions from scratch — no OpenVLA
                    try:
                        actions, reasoning, raw_response = query_gpt_direct(
                            client=gpt_client,
                            task_description=task_description,
                            curr_obs_img=img,
                            prev_obs_img=prev_obs_img,
                            gpt_model=cfg.gpt_model,
                            save_dir=os.path.join(save_dir, eval_mode) if save_dir else None,
                            chunk_idx=gpt_calls,
                            exact_values=cfg.gpt_exact_values,
                        )
                        lang = actions_to_language(actions)
                        logger.info(f"  [GPT-direct chunk {gpt_calls}, step {t}] generated: {lang}")
                        logger.info(f"  [GPT-direct chunk {gpt_calls}, step {t}] {reasoning}")

                        gpt_logs.append({
                            "chunk": gpt_calls,
                            "step": t,
                            "generated_language": lang,
                            "reasoning": reasoning,
                            "raw_response": raw_response,
                            "openvla_actions": None,
                            "corrected_actions": [a.tolist() if hasattr(a, "tolist") else a for a in actions],
                        })
                        gpt_calls += 1
                    except Exception as e:
                        logger.warning(f"GPT direct generation failed: {e}")
                        gpt_fallbacks += 1
                        # Fallback: zero actions with open gripper
                        actions = [np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]) for _ in range(8)]
                else:
                    # Get OpenVLA actions first
                    actions = get_action(
                        cfg, model, observation, task_description,
                        processor=processor, action_head=action_head,
                        proprio_projector=proprio_projector,
                        noisy_action_projector=noisy_action_projector,
                        use_film=cfg.use_film,
                    )

                    # Apply GPT correction if needed
                    if eval_mode != "openvla_only" and gpt_client is not None:
                        gpt_correction_mode = "normal" if eval_mode == "gpt_normal" else "residual"
                        try:
                            corrected, reasoning, raw_response = query_gpt_for_correction(
                                client=gpt_client,
                                task_description=task_description,
                                curr_obs_img=img,
                                openvla_actions=actions,
                                prev_obs_img=prev_obs_img,
                                mode=gpt_correction_mode,
                                gpt_model=cfg.gpt_model,
                                save_dir=os.path.join(save_dir, eval_mode) if save_dir else None,
                                chunk_idx=gpt_calls,
                                exact_values=cfg.gpt_exact_values,
                            )
                            lang = actions_to_language(actions)
                            logger.info(f"  [GPT chunk {gpt_calls}, step {t}] proposed: {lang}")
                            logger.info(f"  [GPT chunk {gpt_calls}, step {t}] {reasoning}")

                            gpt_logs.append({
                                "chunk": gpt_calls,
                                "step": t,
                                "proposed_language": lang,
                                "reasoning": reasoning,
                                "raw_response": raw_response,
                                "openvla_actions": [a.tolist() if hasattr(a, "tolist") else a for a in actions],
                                "corrected_actions": [a.tolist() if hasattr(a, "tolist") else a for a in corrected],
                            })
                            actions = corrected
                            gpt_calls += 1
                        except Exception as e:
                            logger.warning(f"GPT correction failed, using original: {e}")
                            gpt_fallbacks += 1

                prev_obs_img = img.copy()
                action_queue.extend(actions)

            action = action_queue.popleft()
            action = process_action(action, cfg.model_family)
            obs, reward, done, info = env.step(action.tolist())
            if done:
                success = True
                break
            t += 1

    except Exception as e:
        log_message(f"Episode error: {e}", log_file)

    return success, replay_images, {"gpt_calls": gpt_calls, "gpt_fallbacks": gpt_fallbacks, "gpt_logs": gpt_logs}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

@dataclass
class GPTEvalConfig(BaseGenerateConfig):
    gpt_mode: str = "all"                 # "openvla_only", "gpt_normal", "gpt_residual", "gpt_direct", "all"
    gpt_model: str = "gpt-4o"
    gpt_exact_values: bool = False         # If True, ask GPT for exact per-step values; if False, use small/medium/large
    video_dir: str = "./rollout_videos"   # Root directory for organized videos
    video_fps: int = 30


@draccus.wrap()
def eval_with_gpt(cfg: GPTEvalConfig) -> None:
    """Run LIBERO eval with GPT correction, organized video output."""
    validate_config(cfg)
    set_seed_everywhere(cfg.seed)

    # Initialize OpenVLA model
    model, action_head, proprio_projector, noisy_action_projector, processor = initialize_model(cfg)
    resize_size = get_image_resize_size(cfg)

    # Initialize GPT client
    gpt_client = None
    if cfg.gpt_mode != "openvla_only":
        from openai import OpenAI
        gpt_client = OpenAI()

    # Determine modes
    if cfg.gpt_mode == "all":
        eval_modes = ["openvla_only", "gpt_normal", "gpt_residual", "gpt_direct"]
    else:
        eval_modes = [cfg.gpt_mode]

    # Setup logging
    os.makedirs(cfg.local_log_dir, exist_ok=True)
    run_id = f"GPT-EVAL-{cfg.task_suite_name}-{DATE_TIME}"
    local_log_filepath = os.path.join(cfg.local_log_dir, run_id + ".txt")
    log_file = open(local_log_filepath, "w")

    # Initialize LIBERO
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[cfg.task_suite_name]()
    num_tasks = task_suite.n_tasks

    # Determine task range
    task_start = cfg.task_id_start
    task_end = num_tasks if cfg.task_id_end == -1 else min(cfg.task_id_end, num_tasks)

    log_message(f"Task suite: {cfg.task_suite_name}", log_file)
    log_message(f"Task range: [{task_start}, {task_end})", log_file)
    log_message(f"Eval modes: {eval_modes}", log_file)
    log_message(f"GPT model: {cfg.gpt_model}", log_file)
    log_message(f"Video dir: {cfg.video_dir}", log_file)

    # Results tracker
    results = {mode: {"total_episodes": 0, "total_successes": 0, "tasks": {}} for mode in eval_modes}

    for task_id in tqdm.tqdm(range(task_start, task_end), desc="Tasks"):
        task = task_suite.get_task(task_id)
        initial_states = task_suite.get_task_init_states(task_id)
        env, task_description = get_libero_env(task, cfg.model_family, resolution=cfg.env_img_res)

        # Override task description if custom prompt is provided
        if cfg.custom_task_description:
            log_message(f"Overriding task description: '{task_description}' -> '{cfg.custom_task_description}'", log_file)
            task_description = cfg.custom_task_description

        # Clean task name for directory — include timestamp for uniqueness
        task_dir_name = f"{DATE_TIME}__{task_description.lower().replace(' ', '_').replace('/', '_')[:60]}"

        for episode_idx in tqdm.tqdm(
            range(cfg.num_trials_per_task),
            desc=f"Task {task_id}",
            leave=False,
        ):
            initial_state = initial_states[episode_idx]

            # Video directory for this (task, episode)
            ep_video_dir = os.path.join(
                cfg.video_dir, cfg.task_suite_name, task_dir_name, f"ep{episode_idx:03d}"
            )

            # Run each mode with the SAME initial state
            mode_frames = {}  # mode -> list of frames
            mode_success = {}

            for eval_mode in eval_modes:
                log_message(
                    f"\nTask {task_id}: {task_description} | Ep {episode_idx} | Mode: {eval_mode}",
                    log_file,
                )

                success, replay_images, stats = run_episode_with_mode(
                    cfg, env, task_description, model, resize_size, gpt_client,
                    eval_mode, processor, action_head, proprio_projector,
                    noisy_action_projector, initial_state, log_file,
                    save_dir=ep_video_dir,
                )

                mode_frames[eval_mode] = replay_images
                mode_success[eval_mode] = success

                # Update results
                results[eval_mode]["total_episodes"] += 1
                if success:
                    results[eval_mode]["total_successes"] += 1
                task_results = results[eval_mode]["tasks"].setdefault(
                    task_description, {"episodes": 0, "successes": 0}
                )
                task_results["episodes"] += 1
                if success:
                    task_results["successes"] += 1

                log_message(
                    f"  [{eval_mode}] success={success} "
                    f"steps={len(replay_images)} "
                    f"gpt_calls={stats['gpt_calls']} fallbacks={stats['gpt_fallbacks']}",
                    log_file,
                )

                # Save individual video
                video_path = os.path.join(
                    ep_video_dir, f"{eval_mode}--success={success}.mp4"
                )
                save_video(replay_images, video_path, fps=cfg.video_fps)
                log_message(f"  Video: {video_path}", log_file)

                # Save GPT logs per episode — two versions
                if stats.get("gpt_logs"):
                    os.makedirs(ep_video_dir, exist_ok=True)
                    meta = {
                        "task_description": task_description,
                        "task_id": task_id,
                        "episode_idx": episode_idx,
                        "eval_mode": eval_mode,
                        "success": success,
                        "total_steps": len(replay_images),
                        "gpt_calls": stats["gpt_calls"],
                        "gpt_fallbacks": stats["gpt_fallbacks"],
                    }

                    # 1. Full log — complete raw responses + actions
                    full_log_path = os.path.join(ep_video_dir, f"{eval_mode}--gpt_logs_full.json")
                    with open(full_log_path, "w") as gf:
                        json.dump({**meta, "chunks": stats["gpt_logs"]}, gf, indent=2)

                    # 2. Reasoning-only log — just the explanations, easy to read
                    reasoning_path = os.path.join(ep_video_dir, f"{eval_mode}--gpt_reasoning.txt")
                    with open(reasoning_path, "w") as rf:
                        rf.write(f"Task: {task_description}\n")
                        rf.write(f"Mode: {eval_mode}\n")
                        rf.write(f"Success: {success}\n")
                        rf.write(f"GPT calls: {stats['gpt_calls']}\n")
                        rf.write("=" * 60 + "\n\n")
                        for chunk in stats["gpt_logs"]:
                            rf.write(f"--- Chunk {chunk['chunk']}, Step {chunk['step']} ---\n")
                            rf.write(f"{chunk['reasoning']}\n\n")

                    log_message(f"  GPT full logs: {full_log_path}", log_file)
                    log_message(f"  GPT reasoning: {reasoning_path}", log_file)

            # Save side-by-side comparison video
            if len(mode_frames) > 1:
                # Build labeled mode names with success info
                labeled_frames = {}
                for m in eval_modes:
                    if m in mode_frames:
                        label = f"{m} ({'OK' if mode_success[m] else 'FAIL'})"
                        labeled_frames[label] = mode_frames[m]

                sbs_path = os.path.join(ep_video_dir, "side_by_side.mp4")
                save_side_by_side_video(labeled_frames, sbs_path, fps=cfg.video_fps)
                log_message(f"  Side-by-side: {sbs_path}", log_file)

    # Final summary
    log_message("\n" + "=" * 70, log_file)
    log_message("FINAL RESULTS", log_file)
    log_message("=" * 70, log_file)

    for eval_mode in eval_modes:
        r = results[eval_mode]
        rate = r["total_successes"] / r["total_episodes"] if r["total_episodes"] > 0 else 0
        log_message(
            f"  {eval_mode:20s}: {rate:.4f} ({r['total_successes']}/{r['total_episodes']})",
            log_file,
        )
        for task_desc, task_r in r["tasks"].items():
            task_rate = task_r["successes"] / task_r["episodes"] if task_r["episodes"] > 0 else 0
            log_message(
                f"    {task_desc:50s}: {task_rate:.3f} ({task_r['successes']}/{task_r['episodes']})",
                log_file,
            )

    # Save results JSON
    results_path = os.path.join(cfg.local_log_dir, run_id + "_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    log_message(f"\nResults saved to: {results_path}", log_file)
    log_message(f"Videos saved to: {cfg.video_dir}", log_file)
    log_file.close()


if __name__ == "__main__":
    eval_with_gpt()
