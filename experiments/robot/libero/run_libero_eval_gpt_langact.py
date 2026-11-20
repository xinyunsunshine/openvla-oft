"""
run_libero_eval_gpt_langact.py

Closed-loop evaluation: GPT generates language-action prompts that steer a
langact-finetuned OpenVLA-OFT model.  At each action-chunk boundary the loop is:

    observe → GPT plans next motion → generate langact prompt
    → VLA executes 8-step chunk → observe result → repeat

Modes:
  gpt_langact       GPT generates langact prompts, VLA executes
  openvla_langact   VLA with auto-computed langact (no GPT, uses task desc)
  openvla_only      Baseline: VLA with plain task description (no langact)

Features:
  - Proprioceptive state (EEF pos, euler, gripper) sent to GPT as text
  - In-context trajectory examples in the system prompt
  - Task decomposition: GPT breaks the task into subtasks before execution
  - Verification: GPT assesses outcome of each chunk and self-corrects
  - Structured JSON output with reasoning, langact prompt, confidence
  - Bounding-box annotation support (if detector available)

Usage:
    cd /proj/inf-scaling/zwhong/idan/vla-rl/openvla-oft

    export OPENAI_API_KEY='sk-...'

    python experiments/robot/libero/run_libero_eval_gpt_langact.py \
        --pretrained_checkpoint runs/<langact_verbose_checkpoint> \
        --task_suite_name libero_goal \
        --task_id_start 0 --task_id_end 1 \
        --num_trials_per_task 3 \
        --eval_mode gpt_langact \
        --gpt_model gpt-4o \
        --video_dir ./rollout_videos_langact
"""

import base64
import io
import json
import logging
import os
import re
import sys
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import draccus
import imageio
import numpy as np
import torch
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

# langact utilities for computing observed motion
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "scripts", "libero_langact"))
from langact_utils import (  # noqa: E402
    axis_angle_to_euler,
    euler_diff,
    summarize_numeric_actions,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


# =============================================================================
# Langact vocabulary and scale reference for GPT
# =============================================================================

LANGACT_VOCABULARY = """\
The low-level policy understands language-action prompts that describe the \
desired motion over the next 8 simulation steps. Prompts are comma-separated \
phrases composed from the vocabulary below.

## Direction vocabulary

Translation (in centimeters):
  move forward / move back       (robot's x-axis)
  move left    / move right      (robot's y-axis)
  move up      / move down       (robot's z-axis)

Rotation (in degrees, rounded to nearest 10):
  tilt left    / tilt right      (roll)
  tilt back    / tilt forward    (pitch)
  rotate counterclockwise / rotate clockwise  (yaw)

Gripper:
  open gripper / close gripper

## Magnitude ranges (cumulative over 8 steps)

Translation:
  Very small:  0.1 – 0.5 cm   (barely visible nudge)
  Small:       0.5 – 2.0 cm   (fine adjustment)
  Medium:      2.0 – 5.0 cm   (typical reaching)
  Large:       5.0 – 8.0 cm   (fast repositioning)

Rotation:
  10 degrees   (small correction)
  20 degrees   (noticeable rotation)

## Prompt format

Verbose (with magnitudes): "<motion> <value> <unit>, <motion> <value> <unit>, ..., <gripper>"
Examples:
  "move forward 3.0 cm, move down 2.0 cm, close gripper"
  "move left 1.5 cm, tilt right 10 degrees, open gripper"
  "close gripper"
  "move forward 5.0 cm, move up 3.0 cm, tilt back 10 degrees, open gripper"

Translation values use 1 decimal place in cm (e.g. 3.0, 0.5, 1.2).
Rotation values are integers in degrees, rounded to nearest 10 (e.g. 10, 20).
Omit any axis with negligible motion. Always end with a gripper command.

## Important notes

- Translation is reliably steerable. Rotation is less reliable — use sparingly.
- The gripper should be OPEN while approaching, CLOSED while grasping/carrying, \
and OPEN to release/place.
- Each prompt covers only 8 steps (~0.5-8 cm of motion). Plan one chunk at a time.
- Magnitudes are cumulative over the 8-step chunk, NOT per-step.\
"""


CALIBRATION_INFO = """\
## Unit calibration (exact conversions)

The proprioceptive state you receive uses SI units (meters, radians). \
The langact prompts use centimeters and degrees. Here are the exact conversions:

Position:
  1 cm in prompt  =  0.01 meters of EEF delta
  "move forward 3.0 cm" means the EEF x-coordinate should increase by 0.030 m
  "move left 1.5 cm"    means the EEF y-coordinate should increase by 0.015 m
  "move up 2.0 cm"      means the EEF z-coordinate should increase by 0.020 m

Rotation:
  10 degrees in prompt  =  0.1745 radians of Euler angle change
  20 degrees in prompt  =  0.3491 radians of Euler angle change
  Rotation values are always rounded to the nearest 10 degrees.

Gripper:
  gripper qpos > 0.03  →  gripper is currently open
  gripper qpos ≤ 0.03  →  gripper is currently closed

## How to use proprioception for planning

You receive the EEF position in meters. To decide how far to move:
  1. Estimate the target position from the image (e.g., "the drawer handle
     appears to be roughly at x≈0.45, y≈0.10").
  2. Compare with current EEF position (e.g., EEF x=0.40).
  3. Compute the delta: 0.45 - 0.40 = 0.05 m = 5.0 cm.
  4. Generate: "move forward 5.0 cm"

The LIBERO tabletop workspace is approximately:
  x (forward/back):  0.15 – 0.75 m   (robot facing +x)
  y (left/right):   -0.40 – 0.40 m   (robot centered at y≈0)
  z (up/down):       0.75 – 1.15 m   (table surface ≈ 0.81 m)

## Verifying your commands

After each chunk executes, you will see the "observed motion" — the actual
EEF displacement converted to the same cm/degrees format as the prompt.
Compare this against what you commanded:
  - If commanded "move forward 3.0 cm" and observed "move forward 2.8 cm",
    the execution was accurate — proceed normally.
  - If commanded "move forward 3.0 cm" and observed "move forward 0.5 cm",
    there may be a collision or the policy undershoot — try a different approach.
  - If commanded "move left 2.0 cm" and observed "move right 1.0 cm",
    the policy may be ignoring or misinterpreting — try a smaller magnitude
    or a different direction.\
"""


# =============================================================================
# In-context trajectory examples (text-based)
# =============================================================================

ICL_TRAJECTORY_EXAMPLES = """\
## In-context trajectory examples

Below are real sequences of language-action prompts from successful task \
completions.  Each example shows the full sequence of prompts needed to \
complete the task from start to finish.  Use these to understand the typical \
prompt patterns, number of chunks, and gripper timing.

### Example A — "open the middle drawer of the cabinet"
Step  0: move forward 1.8 cm, move down 1.5 cm, open gripper
         (approach the drawer handle)
Step  1: move forward 2.3 cm, move down 0.8 cm, open gripper
         (continue approaching)
Step  2: move forward 1.0 cm, move down 0.3 cm, close gripper
         (reached handle, grasp it)
Step  3: move back 3.5 cm, close gripper
         (pull drawer open)
Step  4: move back 4.0 cm, close gripper
         (continue pulling)
Step  5: move back 2.0 cm, open gripper
         (release handle)
Outcome: drawer is open. Total chunks: 6.

### Example B — "put the bowl on the plate"
Step  0: move forward 2.0 cm, move down 1.0 cm, move left 1.5 cm, open gripper
         (approach the bowl)
Step  1: move forward 1.5 cm, move down 2.0 cm, open gripper
         (descend toward bowl)
Step  2: move forward 0.5 cm, move down 0.5 cm, close gripper
         (grasp bowl)
Step  3: move up 3.0 cm, close gripper
         (lift bowl)
Step  4: move back 2.0 cm, move right 3.0 cm, close gripper
         (transport toward plate)
Step  5: move forward 1.0 cm, move down 2.5 cm, close gripper
         (lower onto plate)
Step  6: move down 1.0 cm, open gripper
         (release bowl)
Step  7: move up 2.0 cm, move back 1.0 cm, open gripper
         (retract)
Outcome: bowl is on the plate. Total chunks: 8.

### Example C — "pick up the red mug and place it on the shelf"
Step  0: move forward 3.0 cm, move down 1.0 cm, open gripper
         (approach mug)
Step  1: move forward 1.5 cm, move down 1.5 cm, move right 0.5 cm, open gripper
         (align with mug handle)
Step  2: move down 0.5 cm, close gripper
         (grasp)
Step  3: move up 4.0 cm, close gripper
         (lift)
Step  4: move up 3.0 cm, move back 1.0 cm, close gripper
         (lift higher for shelf)
Step  5: move forward 2.0 cm, move left 2.0 cm, close gripper
         (move toward shelf)
Step  6: move forward 1.5 cm, move down 1.0 cm, close gripper
         (position over shelf)
Step  7: move down 1.5 cm, open gripper
         (place on shelf)
Step  8: move up 2.0 cm, move back 1.5 cm, open gripper
         (retract)
Outcome: mug is on the shelf. Total chunks: 9.

### Example D — "put the wine bottle on the rack" (from real training data)
Step  0: move back 0.3 cm, move down 0.9 cm, open gripper
         (begin approach, descending toward bottle)
Step  1: move forward 0.2 cm, move left 0.6 cm, open gripper
         (lateral adjustment toward bottle)
Step  2: move down 1.3 cm, open gripper
         (continue descending)
Step  3: move back 0.3 cm, move down 2.4 cm, open gripper
         (descending toward bottle top)
Step  4: move back 1.0 cm, move down 3.1 cm, open gripper
         (reaching bottle — moving down and back)
Step  5: move back 0.8 cm, move down 2.3 cm, move right 0.4 cm, open gripper
         (final descent toward bottle neck)
Step  6: move forward 2.3 cm, move down 1.1 cm, move right 1.8 cm, open gripper
         (aligning with bottle for grasp)
Step  7: move forward 0.7 cm, move down 0.5 cm, move right 1.8 cm, open gripper
         (fine alignment)
Step  8: move back 2.6 cm, move down 1.8 cm, move right 2.3 cm, open gripper
         (positioning hand around bottle)
Step  9: move back 0.8 cm, move down 3.2 cm, move right 2.0 cm, open gripper
         (final approach for grasp)
Step 10: move forward 0.2 cm, move down 0.8 cm, close gripper
         (grasp the wine bottle)
Step 11: move forward 1.9 cm, move up 4.7 cm, close gripper
         (lift bottle)
Step 12: move forward 0.8 cm, move up 8.9 cm, move right 1.8 cm, close gripper
         (lift high, begin transport toward rack)
Step 13: move back 0.7 cm, move up 7.3 cm, move right 4.4 cm, close gripper
         (transport — moving right toward rack)
Step 14: move back 0.5 cm, move up 4.5 cm, move right 6.5 cm, close gripper
         (continue moving right + up toward rack)
Step 15: move back 0.4 cm, move down 1.6 cm, move right 6.3 cm, close gripper
         (approaching rack position, beginning descent)
Step 16: move forward 0.3 cm, move down 3.7 cm, close gripper
         (lowering bottle onto rack)
Step 17: move forward 1.0 cm, open gripper
         (release bottle on rack)
Step 18: move back 0.2 cm, move up 0.2 cm, open gripper
         (retract)
Outcome: wine bottle is on the rack. Total chunks: 19.
Note: this task requires significant lateral (right) movement to reach the rack,
a long approach phase (10 chunks) because the bottle requires careful alignment,
and the bottle must be lifted high before transport.

### Example E — "put the bowl on the plate" (from real training data)
Step  0: move forward 3.9 cm, move down 0.9 cm, move left 1.5 cm, open gripper
         (approach the bowl)
Step  1: move forward 9.1 cm, move down 2.1 cm, move left 2.1 cm, open gripper
         (large reach toward bowl)
Step  2: move forward 5.6 cm, move down 7.4 cm, open gripper
         (descend rapidly toward bowl)
Step  3: move back 3.0 cm, move down 8.6 cm, open gripper
         (continue descending — overshooting forward, correcting back)
Step  4: move back 5.7 cm, move down 6.0 cm, move right 1.5 cm, close gripper
         (final descent and grasp the bowl)
Step  5: move back 0.5 cm, move up 0.9 cm, close gripper
         (slight lift after grasp)
Step  6: move forward 2.9 cm, move up 6.9 cm, close gripper
         (lift bowl up)
Step  7: move forward 7.5 cm, move up 4.6 cm, move right 3.0 cm, close gripper
         (transport bowl toward plate — forward + right)
Step  8: move forward 4.3 cm, move down 6.0 cm, close gripper
         (approaching plate, lowering)
Step  9: move back 0.5 cm, move down 5.7 cm, close gripper
         (lowering onto plate)
Step 10: move forward 0.1 cm, move up 2.4 cm, open gripper
         (release bowl, retract upward)
Outcome: bowl is on the plate. Total chunks: 11.
Note: this task uses large magnitudes (5-9 cm) during approach and transport,
with the bowl grasped around chunk 4-5 and placed around chunk 9-10.

### Key patterns to notice:
1. Approach phase: forward + down, gripper OPEN (2-3 chunks)
2. Grasp: small/no motion, gripper changes from OPEN → CLOSE
3. Lift: primarily up, gripper CLOSED (1-2 chunks)
4. Transport: lateral/forward motion, gripper CLOSED (1-3 chunks)
5. Place: down motion, then gripper OPEN
6. Retract: up + back, gripper OPEN (1 chunk)
7. Wine bottle tasks: require more chunks (longer approach + careful alignment)
8. Real trajectories use magnitudes from 0.1 cm to 9+ cm depending on distance\
"""


# =============================================================================
# System prompts
# =============================================================================

def build_system_prompt(include_icl: bool = True) -> str:
    """Build the system prompt for GPT langact generation."""
    prompt = (
        "You are a high-level robot motion planner for a tabletop manipulation task.\n\n"
        "You observe the scene through RGB images and robot proprioception, then "
        "generate a language-action prompt that tells a low-level visuomotor policy "
        "what motion to execute next.\n\n"
        "The low-level policy is an OpenVLA model fine-tuned on language-action "
        "prompts. It reliably follows translation commands (forward/back, left/right, "
        "up/down) and gripper commands. Rotation commands are less reliable — only "
        "include rotation when clearly necessary.\n\n"
        + LANGACT_VOCABULARY + "\n\n"
        + CALIBRATION_INFO + "\n\n"
    )

    if include_icl:
        prompt += ICL_TRAJECTORY_EXAMPLES + "\n\n"

    prompt += (
        "## Your task\n\n"
        "At each step you receive:\n"
        "1. The task description (what the robot should accomplish)\n"
        "2. The current RGB observation image\n"
        "3. The previous RGB observation image (if available)\n"
        "4. The robot's proprioceptive state (EEF position, orientation, gripper)\n"
        "5. History of previous prompts and their observed outcomes\n\n"
        "Follow these steps:\n\n"
        "1. **OBSERVE**: Describe what you see. Where is the gripper? Where are the "
        "relevant objects? What is the gripper holding (if anything)?\n\n"
        "2. **ASSESS PROGRESS**: If history is provided, what happened in the last "
        "chunk? Did the robot move as expected? Is it closer to completing the task?\n\n"
        "3. **IDENTIFY SUBTASK**: What is the current subtask? "
        "(e.g., approach, align, descend, grasp, lift, transport, lower, place, retract)\n\n"
        "4. **PLAN MOTION**: For each degree of freedom, decide direction and magnitude "
        "for this ONE chunk (8 steps). Think about:\n"
        "   - Translation: which direction and how far? (0.5–8.0 cm)\n"
        "   - Gripper: should it be open or closed?\n"
        "   - Rotation: only if clearly needed (e.g., to align with an angled surface)\n\n"
        "5. **GENERATE PROMPT**: Compose the language-action prompt string.\n\n"
        "6. **OUTPUT**: Respond with a JSON object inside a ```json code block:\n"
        "```json\n"
        "{\n"
        '  "observation": "Brief description of what you see",\n'
        '  "progress": "What happened since last step (or \'first step\' if no history)",\n'
        '  "subtask": "current subtask name (approach/align/descend/grasp/lift/transport/lower/place/retract/done)",\n'
        '  "reasoning": "Why this motion is needed",\n'
        '  "langact_prompt": "move forward 3.0 cm, move down 2.0 cm, close gripper",\n'
        '  "confidence": 0.8,\n'
        '  "task_complete": false\n'
        "}\n"
        "```\n\n"
        "IMPORTANT RULES:\n"
        "- The `langact_prompt` MUST follow the exact vocabulary and format described above.\n"
        "- Always include a gripper command (open gripper / close gripper) at the end.\n"
        "- Use translation magnitudes with 1 decimal (e.g., 3.0 cm, not 3 cm).\n"
        "- Rotation must be 10 or 20 degrees (rounded to nearest 10).\n"
        "- Plan for only ONE 8-step chunk. Do NOT plan the entire trajectory.\n"
        "- Set `task_complete` to true only if you can see the task is fully done.\n"
    )

    return prompt


def build_decomposition_prompt() -> str:
    """Build a prompt for initial task decomposition."""
    return (
        "You are a robot task planner. Given a task description and the initial "
        "observation of a tabletop scene, decompose the task into a sequence of "
        "high-level subtasks.\n\n"
        "Each subtask should be one of:\n"
        "  - approach: move gripper toward the target object (open gripper)\n"
        "  - align: fine-adjust position to be directly above/beside the object\n"
        "  - descend: move down to reach the object\n"
        "  - grasp: close gripper around the object\n"
        "  - lift: raise the grasped object\n"
        "  - transport: move the object horizontally toward the target location\n"
        "  - lower: move the object down toward the target surface\n"
        "  - place: release the object (open gripper)\n"
        "  - retract: move gripper away after placing\n"
        "  - push/pull: for tasks involving pushing or pulling (e.g., opening drawers)\n\n"
        "Output a JSON object:\n"
        "```json\n"
        "{\n"
        '  "scene_description": "What you see in the image",\n'
        '  "task_analysis": "How to accomplish this task",\n'
        '  "subtasks": ["approach", "descend", "grasp", "lift", "transport", "lower", "place", "retract"]\n'
        "}\n"
        "```\n"
    )


# =============================================================================
# Image-based ICL from extracted trajectories
# =============================================================================

ICL_TRAJECTORIES_DIR = os.path.join(
    os.path.dirname(__file__), "..", "..", "..", "icl_langact_trajectories"
)

_image_icl_cache: Optional[list] = None


def load_image_icl_examples(icl_dir: Optional[str] = None, max_chunks_per_traj: int = 6) -> list:
    """Load image-based ICL examples from extracted trajectory data.

    Each example becomes a user/assistant message pair showing an observation
    image + proprio state, with the assistant providing the correct langact prompt.

    To keep the context window manageable, we sample key chunks from each
    trajectory (first, mid, grasp transition, last) rather than all chunks.

    Args:
        icl_dir: Path to extracted ICL directory (with icl_langact_metadata.json).
        max_chunks_per_traj: Max number of chunks to include per trajectory.

    Returns:
        List of (user_content, assistant_content) message pairs.
    """
    global _image_icl_cache
    if _image_icl_cache is not None:
        return _image_icl_cache

    if icl_dir is None:
        icl_dir = ICL_TRAJECTORIES_DIR

    meta_path = os.path.join(icl_dir, "icl_langact_metadata.json")
    if not os.path.exists(meta_path):
        logger.warning(f"Image ICL metadata not found at {meta_path}. Skipping image ICL.")
        _image_icl_cache = []
        return _image_icl_cache

    with open(meta_path) as f:
        metadata = json.load(f)

    pairs = []

    for traj in metadata.get("trajectories", []):
        traj_dir = os.path.join(icl_dir, traj["traj_dir"])
        chunks = traj["chunks"]
        saved_images = {img["step"]: img for img in traj["saved_images"]}
        num_chunks = len(chunks)

        if num_chunks == 0:
            continue

        # Select key chunks: first, ~1/3, grasp point, ~2/3, near-end, last
        key_indices = _select_key_chunks(chunks, max_chunks_per_traj)

        # Build the full prompt sequence text as context
        prompt_sequence = []
        for c in chunks:
            grip = c["gripper_state"]
            prompt_sequence.append(
                f"  Chunk {c['chunk_idx']}: \"{c['prompt']}\" (grip={grip})"
            )
        prompt_seq_text = "\n".join(prompt_sequence)

        for idx in key_indices:
            chunk = chunks[idx]
            step = chunk["step_start"]

            img_info = saved_images.get(step)
            if img_info is None:
                continue

            img_path = os.path.join(traj_dir, img_info["filename"])
            if not os.path.exists(img_path):
                continue

            with open(img_path, "rb") as f:
                img_b64 = base64.b64encode(f.read()).decode("utf-8")

            pos_cm = chunk["eef_pos_cm"]
            euler_deg = chunk["eef_euler_deg"]
            grip = chunk["gripper_state"]

            proprio_text = (
                f"EEF position (cm): x={pos_cm[0]:.1f}, y={pos_cm[1]:.1f}, z={pos_cm[2]:.1f}\n"
                f"EEF orientation (degrees): roll={euler_deg[0]:.1f}°, "
                f"pitch={euler_deg[1]:.1f}°, yaw={euler_deg[2]:.1f}°\n"
                f"Gripper: {grip}"
            )

            # Determine subtask phase from chunk position and gripper transitions
            phase = _infer_phase(chunks, idx)

            task_desc = traj.get("task_description", f"Task from {traj['suite']}")

            # User message: observation + proprio + task context
            user_content = [
                {"type": "text", "text": (
                    f"[ICL EXAMPLE — {traj['suite']}]\n"
                    f"Task: {task_desc}\n"
                    f"This is chunk {chunk['chunk_idx']} of {num_chunks} "
                    f"from a successful trajectory.\n"
                    f"Full prompt sequence for this task:\n{prompt_seq_text}"
                )},
                {"type": "text", "text": "Current observation:"},
                {"type": "image_url", "image_url": {
                    "url": f"data:image/png;base64,{img_b64}",
                }},
                {"type": "text", "text": f"Proprioceptive state:\n{proprio_text}"},
                {"type": "text", "text": "Generate the next language-action prompt. Output JSON:"},
            ]

            # Assistant response: the ground-truth langact prompt
            assistant_content = (
                f"```json\n"
                f"{{\n"
                f'  "observation": "ICL demonstration — {traj["suite"]} task",\n'
                f'  "progress": "Demonstration chunk {chunk["chunk_idx"]} of {num_chunks}",\n'
                f'  "subtask": "{phase}",\n'
                f'  "reasoning": "Following the demonstrated trajectory",\n'
                f'  "langact_prompt": "{chunk["prompt"]}",\n'
                f'  "confidence": 1.0,\n'
                f'  "task_complete": {str(idx == len(chunks) - 1).lower()}\n'
                f"}}\n"
                f"```"
            )

            pairs.append((user_content, assistant_content))

    _image_icl_cache = pairs
    logger.info(f"Loaded {len(pairs)} image ICL examples from {icl_dir}")
    return _image_icl_cache


def _select_key_chunks(chunks: list, max_n: int) -> list:
    """Select key chunk indices that capture the trajectory structure.

    Prioritizes: first chunk, grasp transition, place transition, last chunk,
    then fills in evenly spaced chunks.
    """
    n = len(chunks)
    if n <= max_n:
        return list(range(n))

    key = {0, n - 1}

    # Find gripper transitions (open→closed = grasp, closed→open = place)
    for i in range(1, n):
        prev_grip = chunks[i - 1]["gripper_state"]
        curr_grip = chunks[i]["gripper_state"]
        if prev_grip != curr_grip:
            key.add(i)

    # Fill remaining slots with evenly spaced indices
    remaining = max_n - len(key)
    if remaining > 0:
        candidates = [i for i in range(n) if i not in key]
        step = max(1, len(candidates) // (remaining + 1))
        for j in range(1, remaining + 1):
            idx = min(j * step, len(candidates) - 1)
            key.add(candidates[idx])

    result = sorted(key)[:max_n]
    return result


def _infer_phase(chunks: list, idx: int) -> str:
    """Infer the subtask phase from chunk position and gripper state."""
    n = len(chunks)
    chunk = chunks[idx]
    grip = chunk["gripper_state"]

    # Find first grasp (open→closed transition)
    grasp_idx = None
    release_idx = None
    for i in range(1, n):
        if chunks[i - 1]["gripper_state"] == "open" and chunks[i]["gripper_state"] == "closed":
            if grasp_idx is None:
                grasp_idx = i
        if chunks[i - 1]["gripper_state"] == "closed" and chunks[i]["gripper_state"] == "open":
            release_idx = i

    if grasp_idx is not None:
        if idx < grasp_idx - 1:
            return "approach"
        if idx == grasp_idx - 1:
            return "descend"
        if idx == grasp_idx:
            return "grasp"
        if release_idx is not None:
            if idx < release_idx and grip == "closed":
                if idx <= grasp_idx + 1:
                    return "lift"
                return "transport"
            if idx == release_idx:
                return "place"
            if idx > release_idx:
                return "retract"
        if grip == "closed":
            return "transport"

    frac = idx / max(n - 1, 1)
    if frac < 0.3:
        return "approach"
    if frac < 0.7:
        return "transport"
    return "place"


# =============================================================================
# Image and state utilities
# =============================================================================

def encode_image_to_base64(img: np.ndarray) -> str:
    """Convert numpy RGB image to base64-encoded PNG."""
    pil_img = Image.fromarray(img)
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def format_proprio_state(obs: dict, prev_obs: Optional[dict] = None) -> str:
    """Format proprioceptive state as human-readable text for GPT.

    Includes both meters (raw) and cm (prompt-scale) so GPT can directly
    compute distances in the same units as langact prompts.
    """
    eef_pos = obs["robot0_eef_pos"]
    eef_quat = obs["robot0_eef_quat"]
    eef_axang = quat2axisangle(eef_quat)
    euler = axis_angle_to_euler(eef_axang.reshape(1, 3))[0]
    grip = obs["robot0_gripper_qpos"]
    grip_state = "open" if grip[0] > 0.03 else "closed"

    lines = [
        f"EEF position (meters): x={eef_pos[0]:.4f}, y={eef_pos[1]:.4f}, z={eef_pos[2]:.4f}",
        f"EEF position (cm):     x={eef_pos[0]*100:.1f},   y={eef_pos[1]*100:.1f},   z={eef_pos[2]*100:.1f}",
        f"EEF orientation (degrees): roll={np.degrees(euler[0]):.1f}°, "
        f"pitch={np.degrees(euler[1]):.1f}°, yaw={np.degrees(euler[2]):.1f}°",
        f"Gripper: {grip_state} (qpos: {grip[0]:.3f}, {grip[1]:.3f})",
    ]

    if prev_obs is not None:
        prev_pos = prev_obs["robot0_eef_pos"]
        delta = eef_pos - prev_pos
        lines.append(
            f"Delta since last chunk (cm): dx={delta[0]*100:+.1f}, "
            f"dy={delta[1]*100:+.1f}, dz={delta[2]*100:+.1f}"
        )

    return "\n".join(lines)


def compute_observed_motion(obs_before: dict, obs_after: dict) -> str:
    """Compute the actual motion that occurred between two observations.

    Returns a verbose langact-style description of what happened.
    """
    state_before = np.concatenate([
        obs_before["robot0_eef_pos"],
        quat2axisangle(obs_before["robot0_eef_quat"]),
    ])
    state_after = np.concatenate([
        obs_after["robot0_eef_pos"],
        quat2axisangle(obs_after["robot0_eef_quat"]),
    ])

    delta_pos = state_after[:3] - state_before[:3]
    euler_before = axis_angle_to_euler(state_before[3:6].reshape(1, 3))[0]
    euler_after = axis_angle_to_euler(state_after[3:6].reshape(1, 3))[0]
    delta_rot = euler_diff(euler_after.reshape(1, 3), euler_before.reshape(1, 3))[0]

    gripper = 1.0 - np.clip(obs_after["robot0_gripper_qpos"][0], 0, 1)
    lang_action = np.concatenate([delta_pos, delta_rot, [gripper]])
    text = summarize_numeric_actions(lang_action, sum_decimal="1f", include_rotation=True)
    return text or "close gripper"


def save_obs_image(img: np.ndarray, path: str) -> None:
    """Save observation image to disk."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    Image.fromarray(img).save(path)


# =============================================================================
# GPT query functions
# =============================================================================

def query_gpt_decompose(
    client,
    task_description: str,
    obs_img: np.ndarray,
    gpt_model: str = "gpt-4o",
) -> dict:
    """Ask GPT to decompose the task into subtasks."""
    system_prompt = build_decomposition_prompt()

    content = [
        {"type": "text", "text": f"Task: {task_description}"},
        {"type": "text", "text": "Initial observation:"},
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encode_image_to_base64(obs_img)}"}},
        {"type": "text", "text": "Decompose this task into subtasks. Output JSON:"},
    ]

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": content},
    ]

    try:
        response = client.chat.completions.create(
            model=gpt_model,
            messages=messages,
            temperature=0.0,
            max_completion_tokens=1000,
        )
        raw = response.choices[0].message.content.strip()
        json_match = re.search(r"```(?:json)?\s*\n?([\s\S]*?)```", raw)
        if json_match:
            return json.loads(json_match.group(1).strip())
        array_match = re.search(r"(\{[\s\S]*\})", raw)
        if array_match:
            return json.loads(array_match.group(1).strip())
    except Exception as e:
        logger.warning(f"Task decomposition failed: {e}")

    return {
        "scene_description": "unknown",
        "task_analysis": "unknown",
        "subtasks": ["approach", "grasp", "lift", "transport", "place"],
    }


def query_gpt_langact(
    client,
    task_description: str,
    curr_obs_img: np.ndarray,
    curr_obs_raw: dict,
    prev_obs_img: Optional[np.ndarray],
    prev_obs_raw: Optional[dict],
    history: List[dict],
    decomposition: Optional[dict],
    gpt_model: str = "gpt-4o",
    max_retries: int = 3,
    include_icl: bool = True,
    use_image_icl: bool = False,
    icl_trajectories_dir: Optional[str] = None,
) -> Tuple[str, dict, str]:
    """Query GPT to generate a langact prompt for the next action chunk.

    Args:
        use_image_icl: If True, include observation images from extracted
            trajectories as few-shot user/assistant pairs.
        icl_trajectories_dir: Path to extracted ICL directory. If None, uses
            the default path.

    Returns:
        (langact_prompt, parsed_json, raw_response)
    """
    system_prompt = build_system_prompt(include_icl=include_icl and not use_image_icl)
    proprio_text = format_proprio_state(curr_obs_raw, prev_obs=prev_obs_raw)

    content: list = [
        {"type": "text", "text": f"Task: {task_description}"},
    ]

    if decomposition:
        subtask_list = ", ".join(decomposition.get("subtasks", []))
        content.append({
            "type": "text",
            "text": f"Task plan (subtasks): {subtask_list}",
        })

    if prev_obs_img is not None:
        content.append({"type": "text", "text": "Previous observation (before last chunk):"})
        content.append({"type": "image_url", "image_url": {
            "url": f"data:image/png;base64,{encode_image_to_base64(prev_obs_img)}",
        }})
    else:
        content.append({"type": "text", "text": "(No previous observation — this is the first step.)"})

    content.append({"type": "text", "text": "Current observation:"})
    content.append({"type": "image_url", "image_url": {
        "url": f"data:image/png;base64,{encode_image_to_base64(curr_obs_img)}",
    }})

    content.append({"type": "text", "text": f"Proprioceptive state:\n{proprio_text}"})

    if history:
        history_lines = []
        for h in history[-5:]:
            history_lines.append(
                f"  Chunk {h['chunk']}: prompt=\"{h['prompt']}\" → observed=\"{h['observed_motion']}\""
            )
        content.append({
            "type": "text",
            "text": f"Recent history (last {len(history_lines)} chunks):\n" + "\n".join(history_lines),
        })
    else:
        content.append({"type": "text", "text": "History: none (first chunk of episode)"})

    content.append({"type": "text", "text": "Generate the next language-action prompt. Output JSON:"})

    messages = [
        {"role": "system", "content": system_prompt},
    ]

    # Insert image-based ICL as few-shot user/assistant pairs
    if use_image_icl:
        icl_pairs = load_image_icl_examples(icl_dir=icl_trajectories_dir)
        for user_content, assistant_content in icl_pairs:
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
                max_completion_tokens=1500,
            )
            raw = response.choices[0].message.content.strip()

            json_match = re.search(r"```(?:json)?\s*\n?([\s\S]*?)```", raw)
            if json_match:
                json_str = json_match.group(1).strip()
            else:
                obj_match = re.search(r"(\{[\s\S]*\})", raw)
                if obj_match:
                    json_str = obj_match.group(1).strip()
                else:
                    raise ValueError(f"No JSON found in response: {raw[:200]}")

            json_str = re.sub(r"//[^\n]*", "", json_str)
            parsed = json.loads(json_str)

            langact_prompt = parsed.get("langact_prompt", "")
            if not langact_prompt:
                raise ValueError("Empty langact_prompt in response")

            _validate_langact_prompt(langact_prompt)

            reasoning = raw[:raw.find("```")].strip() if "```" in raw else raw
            return langact_prompt, parsed, raw

        except Exception as e:
            if attempt < max_retries - 1:
                logger.warning(f"GPT langact attempt {attempt+1} failed: {e}. Retrying...")
                time.sleep(2)
            else:
                logger.error(f"GPT langact failed after {max_retries} attempts: {e}")
                fallback_prompt = "move forward 1.0 cm, open gripper"
                return fallback_prompt, {"langact_prompt": fallback_prompt, "error": str(e)}, raw


def _validate_langact_prompt(prompt: str) -> None:
    """Basic validation that a langact prompt uses the correct vocabulary."""
    prompt_lower = prompt.lower().strip()

    if not prompt_lower:
        raise ValueError("Empty prompt")

    has_gripper = "open gripper" in prompt_lower or "close gripper" in prompt_lower
    if not has_gripper:
        raise ValueError(f"Prompt missing gripper command: {prompt}")

    valid_directions = [
        "move forward", "move back", "move left", "move right",
        "move up", "move down",
        "tilt left", "tilt right", "tilt back", "tilt forward",
        "rotate counterclockwise", "rotate clockwise",
        "open gripper", "close gripper",
    ]
    parts = [p.strip() for p in prompt_lower.split(",")]
    for part in parts:
        part_clean = re.sub(r"\d+\.?\d*\s*(cm|degrees?)", "", part).strip()
        if part_clean not in valid_directions:
            raise ValueError(f"Unknown direction phrase: '{part_clean}' in '{prompt}'")


# =============================================================================
# Open-loop multi-step planning + verification/self-correction
# =============================================================================

def build_multistep_system_prompt(
    plan_horizon: int = 3,
    include_icl: bool = True,
    use_image_icl: bool = False,
) -> str:
    """System prompt for open-loop multi-step planning.

    GPT generates `plan_horizon` langact prompts at once. The VLA executes
    them sequentially, then GPT re-observes and either verifies + corrects
    or plans the next batch.
    """
    prompt = (
        "You are a high-level robot motion planner for a tabletop manipulation task.\n\n"
        "You observe the scene through RGB images and robot proprioception, then "
        "generate a SHORT PLAN of consecutive language-action prompts. Each prompt "
        "tells a low-level visuomotor policy what motion to execute for one 8-step "
        "action chunk.\n\n"
        "The low-level policy is an OpenVLA model fine-tuned on language-action "
        "prompts. It reliably follows translation commands (forward/back, left/right, "
        "up/down) and gripper commands. Rotation commands are less reliable — only "
        "include rotation when clearly necessary.\n\n"
        + LANGACT_VOCABULARY + "\n\n"
        + CALIBRATION_INFO + "\n\n"
    )

    if include_icl and not use_image_icl:
        prompt += ICL_TRAJECTORY_EXAMPLES + "\n\n"

    prompt += (
        f"## Your task — Multi-Step Planning (horizon = {plan_horizon})\n\n"
        "At each planning step you receive:\n"
        "1. The task description\n"
        "2. The current RGB observation image\n"
        "3. The previous RGB observation image (if available)\n"
        "4. Robot proprioceptive state (EEF position, orientation, gripper)\n"
        "5. History of previous prompts and their observed outcomes\n"
        "6. Task decomposition (if available)\n\n"
        "Follow these steps:\n\n"
        "1. **OBSERVE**: Describe the scene — gripper position, object locations, "
        "what the gripper holds.\n\n"
        "2. **ASSESS PROGRESS**: Review history. Did previous commands achieve "
        "their intended effect?\n\n"
        "3. **IDENTIFY PHASE**: Where are we in the task? "
        "(approach / align / descend / grasp / lift / transport / lower / place / retract)\n\n"
        f"4. **PLAN {plan_horizon} CHUNKS**: Generate {plan_horizon} consecutive langact "
        "prompts. Think about the cumulative motion — each chunk moves ~1–8 cm, so "
        f"{plan_horizon} chunks cover ~{plan_horizon}–{plan_horizon*8} cm total. "
        "Consider:\n"
        "   - What trajectory shape is needed? (e.g., descend then close gripper)\n"
        "   - Where should the gripper state change? (open→close for grasp, close→open for place)\n"
        "   - Do NOT plan past a grasp or place transition — those are critical moments "
        "where re-observation is needed.\n\n"
        "5. **OUTPUT**: Respond with a JSON object inside a ```json code block:\n"
        "```json\n"
        "{\n"
        '  "observation": "Brief description of what you see",\n'
        '  "progress": "Assessment of recent history",\n'
        '  "phase": "current phase (approach/descend/grasp/lift/transport/lower/place/retract)",\n'
        '  "reasoning": "Why this plan is appropriate",\n'
        '  "plan": [\n'
        '    {"prompt": "move forward 3.0 cm, move down 2.0 cm, open gripper", '
        '"intent": "approach the target object"},\n'
        '    {"prompt": "move forward 2.0 cm, move down 4.0 cm, open gripper", '
        '"intent": "descend to grasp height"},\n'
        '    {"prompt": "move forward 1.0 cm, move down 1.0 cm, close gripper", '
        '"intent": "close gripper to grasp"}\n'
        "  ],\n"
        '  "expect_grasp_or_place": true,\n'
        '  "confidence": 0.8,\n'
        '  "task_complete": false\n'
        "}\n"
        "```\n\n"
        "IMPORTANT RULES:\n"
        f"- The `plan` array MUST contain exactly {plan_horizon} entries.\n"
        "- Each `prompt` MUST follow the exact langact vocabulary and format.\n"
        "- Always include a gripper command at the end of each prompt.\n"
        "- Set `expect_grasp_or_place` to true if ANY prompt in the plan changes "
        "the gripper state (open→close or close→open). This triggers re-observation "
        "and verification after execution.\n"
        "- Set `task_complete` to true only if the task is fully done.\n"
        f"- If fewer than {plan_horizon} chunks remain, pad with small motions "
        "(e.g., 'move forward 0.1 cm, open gripper').\n"
    )

    return prompt


def query_gpt_multistep(
    client,
    task_description: str,
    curr_obs_img: np.ndarray,
    curr_obs_raw: dict,
    prev_obs_img: Optional[np.ndarray],
    prev_obs_raw: Optional[dict],
    history: List[dict],
    decomposition: Optional[dict],
    plan_horizon: int = 3,
    gpt_model: str = "gpt-4o",
    max_retries: int = 3,
    include_icl: bool = True,
    use_image_icl: bool = False,
    icl_trajectories_dir: Optional[str] = None,
) -> Tuple[List[dict], dict, str]:
    """Query GPT to generate a multi-step plan of langact prompts.

    Returns:
        (plan, parsed_json, raw_response)
        where plan is a list of {"prompt": str, "intent": str} dicts.
    """
    system_prompt = build_multistep_system_prompt(
        plan_horizon=plan_horizon,
        include_icl=include_icl and not use_image_icl,
        use_image_icl=use_image_icl,
    )
    proprio_text = format_proprio_state(curr_obs_raw, prev_obs=prev_obs_raw)

    content: list = [
        {"type": "text", "text": f"Task: {task_description}"},
    ]

    if decomposition:
        subtask_list = ", ".join(decomposition.get("subtasks", []))
        content.append({
            "type": "text",
            "text": f"Task plan (subtasks): {subtask_list}",
        })

    if prev_obs_img is not None:
        content.append({"type": "text", "text": "Previous observation (before last chunk):"})
        content.append({"type": "image_url", "image_url": {
            "url": f"data:image/png;base64,{encode_image_to_base64(prev_obs_img)}",
        }})
    else:
        content.append({"type": "text", "text": "(No previous observation — first step.)"})

    content.append({"type": "text", "text": "Current observation:"})
    content.append({"type": "image_url", "image_url": {
        "url": f"data:image/png;base64,{encode_image_to_base64(curr_obs_img)}",
    }})

    content.append({"type": "text", "text": f"Proprioceptive state:\n{proprio_text}"})

    if history:
        history_lines = []
        for h in history[-8:]:
            history_lines.append(
                f"  Chunk {h['chunk']}: prompt=\"{h['prompt']}\" → observed=\"{h['observed_motion']}\""
            )
        content.append({
            "type": "text",
            "text": f"Recent history (last {len(history_lines)} chunks):\n" + "\n".join(history_lines),
        })
    else:
        content.append({"type": "text", "text": "History: none (first planning step)"})

    content.append({
        "type": "text",
        "text": f"Generate a plan of {plan_horizon} consecutive langact prompts. Output JSON:",
    })

    messages = [
        {"role": "system", "content": system_prompt},
    ]

    if use_image_icl:
        icl_pairs = load_image_icl_examples(icl_dir=icl_trajectories_dir)
        for user_content, assistant_content in icl_pairs:
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
                max_completion_tokens=2000,
            )
            raw = response.choices[0].message.content.strip()

            json_match = re.search(r"```(?:json)?\s*\n?([\s\S]*?)```", raw)
            if json_match:
                json_str = json_match.group(1).strip()
            else:
                obj_match = re.search(r"(\{[\s\S]*\})", raw)
                if obj_match:
                    json_str = obj_match.group(1).strip()
                else:
                    raise ValueError(f"No JSON found in response: {raw[:200]}")

            json_str = re.sub(r"//[^\n]*", "", json_str)
            parsed = json.loads(json_str)

            plan = parsed.get("plan", [])
            if not plan or not isinstance(plan, list):
                raise ValueError(f"Missing or invalid 'plan' array in response")

            for i, entry in enumerate(plan):
                prompt = entry.get("prompt", "")
                if not prompt:
                    raise ValueError(f"Empty prompt in plan entry {i}")
                _validate_langact_prompt(prompt)

            return plan, parsed, raw

        except Exception as e:
            if attempt < max_retries - 1:
                logger.warning(f"GPT multistep attempt {attempt+1} failed: {e}. Retrying...")
                time.sleep(2)
            else:
                logger.error(f"GPT multistep failed after {max_retries} attempts: {e}")
                fallback = [
                    {"prompt": "move forward 1.0 cm, open gripper", "intent": "fallback"}
                    for _ in range(plan_horizon)
                ]
                return fallback, {"plan": fallback, "error": str(e)}, raw


def query_gpt_verify(
    client,
    task_description: str,
    obs_before_img: np.ndarray,
    obs_after_img: np.ndarray,
    obs_after_raw: dict,
    obs_before_raw: dict,
    plan_executed: List[dict],
    observed_motions: List[str],
    decomposition: Optional[dict],
    gpt_model: str = "gpt-4o",
    max_retries: int = 2,
) -> Tuple[Optional[str], dict, str]:
    """Verify plan execution and optionally generate a corrective prompt.

    Compares what was commanded vs what was observed, asks GPT whether
    correction is needed.

    Returns:
        (correction_prompt_or_none, parsed_json, raw_response)
        correction_prompt is None if no correction needed.
    """
    system_prompt = (
        "You are a robot motion verifier. After a sequence of action chunks was "
        "executed, you compare the commanded motions against the observed motions "
        "and the resulting scene.\n\n"
        "Your job:\n"
        "1. Check if each commanded motion was executed accurately.\n"
        "2. Assess whether the overall goal was advanced.\n"
        "3. If something went wrong (e.g., missed a grasp, overshot, wrong direction), "
        "generate ONE corrective langact prompt to fix it.\n"
        "4. If execution was satisfactory, indicate no correction is needed.\n\n"
        + LANGACT_VOCABULARY + "\n\n"
        + CALIBRATION_INFO + "\n\n"
        "Respond with JSON inside a ```json code block:\n"
        "```json\n"
        "{\n"
        '  "assessment": "Brief assessment of plan execution",\n'
        '  "errors_detected": [\n'
        '    {"chunk": 0, "issue": "description of problem", "severity": "minor|major"}\n'
        "  ],\n"
        '  "needs_correction": true,\n'
        '  "correction_prompt": "move back 2.0 cm, move down 1.0 cm, close gripper",\n'
        '  "correction_reasoning": "Why this correction is needed",\n'
        '  "confidence": 0.7\n'
        "}\n"
        "```\n\n"
        "Rules:\n"
        "- Set `needs_correction` to false if execution was acceptable (even if not perfect).\n"
        "- If `needs_correction` is false, set `correction_prompt` to null.\n"
        "- Only generate corrections for MAJOR errors (missed grasp, wrong direction, "
        "significant overshoot/undershoot).\n"
        "- Minor deviations (< 1 cm) do not need correction — the next planning cycle "
        "will handle them.\n"
        "- The correction prompt follows the same langact format.\n"
    )

    # Build execution log
    exec_log_lines = []
    for i, (plan_entry, obs_motion) in enumerate(zip(plan_executed, observed_motions)):
        exec_log_lines.append(
            f"  Chunk {i}: commanded=\"{plan_entry['prompt']}\" | "
            f"intent=\"{plan_entry.get('intent', '?')}\" | "
            f"observed=\"{obs_motion}\""
        )
    exec_log = "\n".join(exec_log_lines)

    proprio_text = format_proprio_state(obs_after_raw, prev_obs=obs_before_raw)

    content = [
        {"type": "text", "text": f"Task: {task_description}"},
        {"type": "text", "text": "Observation BEFORE plan execution:"},
        {"type": "image_url", "image_url": {
            "url": f"data:image/png;base64,{encode_image_to_base64(obs_before_img)}",
        }},
        {"type": "text", "text": "Observation AFTER plan execution:"},
        {"type": "image_url", "image_url": {
            "url": f"data:image/png;base64,{encode_image_to_base64(obs_after_img)}",
        }},
        {"type": "text", "text": f"Current proprioceptive state:\n{proprio_text}"},
        {"type": "text", "text": f"Execution log ({len(plan_executed)} chunks):\n{exec_log}"},
        {"type": "text", "text": "Verify execution and output JSON:"},
    ]

    if decomposition:
        subtask_list = ", ".join(decomposition.get("subtasks", []))
        content.insert(1, {
            "type": "text",
            "text": f"Task plan (subtasks): {subtask_list}",
        })

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": content},
    ]

    for attempt in range(max_retries):
        raw = ""
        try:
            response = client.chat.completions.create(
                model=gpt_model,
                messages=messages,
                temperature=0.0,
                max_completion_tokens=1500,
            )
            raw = response.choices[0].message.content.strip()

            json_match = re.search(r"```(?:json)?\s*\n?([\s\S]*?)```", raw)
            if json_match:
                json_str = json_match.group(1).strip()
            else:
                obj_match = re.search(r"(\{[\s\S]*\})", raw)
                if obj_match:
                    json_str = obj_match.group(1).strip()
                else:
                    raise ValueError(f"No JSON in verify response: {raw[:200]}")

            json_str = re.sub(r"//[^\n]*", "", json_str)
            parsed = json.loads(json_str)

            correction_prompt = None
            if parsed.get("needs_correction", False):
                correction_prompt = parsed.get("correction_prompt")
                if correction_prompt:
                    _validate_langact_prompt(correction_prompt)

            return correction_prompt, parsed, raw

        except Exception as e:
            if attempt < max_retries - 1:
                logger.warning(f"GPT verify attempt {attempt+1} failed: {e}. Retrying...")
                time.sleep(2)
            else:
                logger.warning(f"GPT verify failed after {max_retries} attempts: {e}. Skipping correction.")
                return None, {"assessment": "verification failed", "error": str(e)}, raw


def _execute_one_chunk(
    cfg, env, model, obs, resize_size, langact_prompt,
    processor, action_head, proprio_projector, noisy_action_projector,
    replay_images: list,
) -> Tuple[dict, bool, bool, int, np.ndarray]:
    """Execute one langact chunk and return (obs, done, success, steps_taken, observation_img)."""
    observation, img = prepare_observation(obs, resize_size)
    replay_images.append(img)

    with torch.no_grad():
        actions = get_action(
            cfg=cfg, model=model, obs=observation, task_label=langact_prompt,
            processor=processor, action_head=action_head,
            proprio_projector=proprio_projector,
            noisy_action_projector=noisy_action_projector,
            use_film=cfg.use_film,
        )

    done = False
    success = False
    steps = 0
    for action in actions:
        action = process_action(action, cfg.model_family)
        obs, reward, done, info = env.step(action.tolist())
        _, step_img = prepare_observation(obs, resize_size)
        replay_images.append(step_img)
        steps += 1
        if done:
            success = True
            break

    return obs, done, success, steps, img


def run_episode_gpt_multistep(
    cfg,
    env,
    task_description: str,
    model,
    resize_size: int,
    gpt_client,
    processor=None,
    action_head=None,
    proprio_projector=None,
    noisy_action_projector=None,
    initial_state=None,
    save_dir: Optional[str] = None,
) -> Tuple[bool, List[np.ndarray], dict]:
    """Run an episode with open-loop multi-step planning and verification.

    Loop:
      1. GPT generates N langact prompts (a "plan")
      2. VLA executes all N chunks open-loop
      3. GPT verifies the outcome: compares commanded vs observed
      4. If major error detected → GPT generates ONE corrective prompt → VLA executes it
      5. Repeat from step 1
    """
    env.reset()
    if initial_state is not None:
        obs = env.set_init_state(initial_state)
    else:
        obs = env.get_observation()

    plan_horizon = cfg.plan_horizon
    max_steps = TASK_MAX_STEPS[cfg.task_suite_name]
    t = 0
    replay_images = []
    success = False

    for _ in range(cfg.num_steps_wait):
        obs, _, _, _ = env.step(get_libero_dummy_action(cfg.model_family))
        t += 1

    _, init_img = prepare_observation(obs, resize_size)
    decomposition = None
    if cfg.use_decomposition:
        decomposition = query_gpt_decompose(gpt_client, task_description, init_img, cfg.gpt_model)
        logger.info(f"  Task decomposition: {decomposition.get('subtasks', [])}")

    history: List[dict] = []
    prev_obs_img: Optional[np.ndarray] = None
    prev_obs_raw: Optional[dict] = None
    chunk_idx = 0
    plan_idx = 0
    gpt_logs: List[dict] = []

    ep_save_dir = os.path.join(save_dir, "gpt_multistep") if save_dir else None

    try:
        while t < max_steps + cfg.num_steps_wait:
            # ── Phase 1: GPT generates a multi-step plan ──
            observation, img = prepare_observation(obs, resize_size)

            plan, plan_json, plan_raw = query_gpt_multistep(
                client=gpt_client,
                task_description=task_description,
                curr_obs_img=img,
                curr_obs_raw=obs,
                prev_obs_img=prev_obs_img,
                prev_obs_raw=prev_obs_raw,
                history=history,
                decomposition=decomposition,
                plan_horizon=plan_horizon,
                gpt_model=cfg.gpt_model,
                include_icl=cfg.use_icl,
                use_image_icl=cfg.use_image_icl,
                icl_trajectories_dir=cfg.icl_trajectories_dir or None,
            )

            logger.info(f"  [plan {plan_idx}, chunk {chunk_idx}, step {t}] "
                         f"GPT plan ({len(plan)} chunks):")
            for i, entry in enumerate(plan):
                logger.info(f"    {i}: \"{entry['prompt']}\"  ({entry.get('intent', '')})")

            # Save plan debug info
            if ep_save_dir:
                plan_dir = os.path.join(ep_save_dir, f"plan_{plan_idx:03d}")
                os.makedirs(plan_dir, exist_ok=True)
                save_obs_image(img, os.path.join(plan_dir, "obs_before_plan.png"))
                if prev_obs_img is not None:
                    save_obs_image(prev_obs_img, os.path.join(plan_dir, "obs_previous.png"))
                proprio_text = format_proprio_state(obs, prev_obs=prev_obs_raw)
                with open(os.path.join(plan_dir, "plan_input.json"), "w") as f:
                    json.dump({
                        "task_description": task_description,
                        "decomposition": decomposition,
                        "proprio_state": proprio_text,
                        "history": [
                            {"chunk": h["chunk"], "prompt": h["prompt"],
                             "observed_motion": h["observed_motion"]}
                            for h in history[-8:]
                        ],
                        "plan_idx": plan_idx,
                        "chunk_idx": chunk_idx,
                        "step": t,
                    }, f, indent=2)
                with open(os.path.join(plan_dir, "plan_response.json"), "w") as f:
                    json.dump({"parsed": plan_json, "raw": plan_raw}, f, indent=2)

            # Remember the observation before open-loop execution
            obs_before_plan = obs
            img_before_plan = img.copy()

            # ── Phase 2: Execute plan open-loop ──
            plan_executed = []
            observed_motions = []
            done = False

            for plan_entry in plan:
                if done or t >= max_steps + cfg.num_steps_wait:
                    break

                prompt = plan_entry["prompt"]
                obs_before_chunk = obs

                obs, done, success, steps, chunk_img = _execute_one_chunk(
                    cfg, env, model, obs, resize_size, prompt,
                    processor, action_head, proprio_projector,
                    noisy_action_projector, replay_images,
                )
                t += steps

                observed_motion = compute_observed_motion(obs_before_chunk, obs)
                plan_executed.append(plan_entry)
                observed_motions.append(observed_motion)

                history_entry = {
                    "chunk": chunk_idx,
                    "step": t,
                    "prompt": prompt,
                    "observed_motion": observed_motion,
                    "gpt_json": {"intent": plan_entry.get("intent", ""), "plan_idx": plan_idx},
                    "source": "plan",
                }
                history.append(history_entry)
                gpt_logs.append({**history_entry, "raw_response": plan_raw})

                logger.info(f"    chunk {chunk_idx}: \"{prompt}\" → observed: \"{observed_motion}\"")

                # Save per-chunk debug
                if ep_save_dir:
                    chunk_dir = os.path.join(ep_save_dir, f"plan_{plan_idx:03d}",
                                              f"chunk_{chunk_idx:03d}")
                    os.makedirs(chunk_dir, exist_ok=True)
                    save_obs_image(chunk_img, os.path.join(chunk_dir, "obs_before.png"))
                    _, post_img = prepare_observation(obs, resize_size)
                    save_obs_image(post_img, os.path.join(chunk_dir, "obs_after.png"))
                    with open(os.path.join(chunk_dir, "chunk_outcome.json"), "w") as f:
                        json.dump({
                            "commanded_prompt": prompt,
                            "intent": plan_entry.get("intent", ""),
                            "observed_motion": observed_motion,
                            "post_proprio": format_proprio_state(obs, prev_obs=obs_before_chunk),
                            "done": done,
                            "success": success,
                        }, f, indent=2)

                chunk_idx += 1

                if done:
                    break

            if done:
                break

            # Check for task_complete signal from planning
            if plan_json.get("task_complete", False):
                logger.info("  GPT signaled task_complete=true in plan.")
                break

            # ── Phase 3: Verification / Self-Correction ──
            if cfg.use_verification and plan_executed:
                _, post_plan_img = prepare_observation(obs, resize_size)

                correction_prompt, verify_json, verify_raw = query_gpt_verify(
                    client=gpt_client,
                    task_description=task_description,
                    obs_before_img=img_before_plan,
                    obs_after_img=post_plan_img,
                    obs_after_raw=obs,
                    obs_before_raw=obs_before_plan,
                    plan_executed=plan_executed,
                    observed_motions=observed_motions,
                    decomposition=decomposition,
                    gpt_model=cfg.gpt_model,
                )

                assessment = verify_json.get("assessment", "")
                errors = verify_json.get("errors_detected", [])
                logger.info(f"  Verification: {assessment}")
                if errors:
                    for err in errors:
                        logger.info(f"    error: chunk {err.get('chunk', '?')}, "
                                     f"{err.get('severity', '?')}: {err.get('issue', '?')}")

                # Save verification debug
                if ep_save_dir:
                    plan_dir = os.path.join(ep_save_dir, f"plan_{plan_idx:03d}")
                    save_obs_image(post_plan_img, os.path.join(plan_dir, "obs_after_plan.png"))
                    with open(os.path.join(plan_dir, "verify_response.json"), "w") as f:
                        json.dump({"parsed": verify_json, "raw": verify_raw}, f, indent=2)

                # Execute correction if needed
                if correction_prompt and t < max_steps + cfg.num_steps_wait:
                    logger.info(f"  Correction: \"{correction_prompt}\"")
                    obs_before_corr = obs

                    obs, done, success, steps, corr_img = _execute_one_chunk(
                        cfg, env, model, obs, resize_size, correction_prompt,
                        processor, action_head, proprio_projector,
                        noisy_action_projector, replay_images,
                    )
                    t += steps

                    corr_observed = compute_observed_motion(obs_before_corr, obs)
                    logger.info(f"    correction observed: \"{corr_observed}\"")

                    history.append({
                        "chunk": chunk_idx,
                        "step": t,
                        "prompt": correction_prompt,
                        "observed_motion": corr_observed,
                        "gpt_json": verify_json,
                        "source": "correction",
                    })
                    gpt_logs.append({
                        "chunk": chunk_idx,
                        "step": t,
                        "prompt": correction_prompt,
                        "observed_motion": corr_observed,
                        "gpt_json": verify_json,
                        "raw_response": verify_raw,
                        "source": "correction",
                    })

                    if ep_save_dir:
                        corr_dir = os.path.join(ep_save_dir, f"plan_{plan_idx:03d}",
                                                 f"correction_{chunk_idx:03d}")
                        os.makedirs(corr_dir, exist_ok=True)
                        save_obs_image(corr_img, os.path.join(corr_dir, "obs_before.png"))
                        _, corr_post = prepare_observation(obs, resize_size)
                        save_obs_image(corr_post, os.path.join(corr_dir, "obs_after.png"))
                        with open(os.path.join(corr_dir, "correction.json"), "w") as f:
                            json.dump({
                                "correction_prompt": correction_prompt,
                                "correction_reasoning": verify_json.get("correction_reasoning", ""),
                                "observed_motion": corr_observed,
                            }, f, indent=2)

                    chunk_idx += 1

                    if done:
                        break

            # Update prev obs for next planning cycle
            prev_obs_img = img_before_plan
            prev_obs_raw = obs_before_plan
            plan_idx += 1

    except Exception as e:
        logger.error(f"Episode error: {e}", exc_info=True)

    # Save episode summary
    if ep_save_dir:
        episode_summary = {
            "task_description": task_description,
            "decomposition": decomposition,
            "success": success,
            "total_plans": plan_idx,
            "total_chunks": chunk_idx,
            "total_steps": len(replay_images),
            "plan_horizon": plan_horizon,
            "use_verification": cfg.use_verification,
            "gpt_model": cfg.gpt_model,
            "chunk_log": [
                {
                    "chunk": g["chunk"],
                    "step": g["step"],
                    "prompt": g["prompt"],
                    "observed_motion": g["observed_motion"],
                    "source": g.get("source", "plan"),
                    "intent": g["gpt_json"].get("intent", ""),
                }
                for g in gpt_logs
            ],
        }
        with open(os.path.join(ep_save_dir, "episode_summary.json"), "w") as f:
            json.dump(episode_summary, f, indent=2)

    stats = {
        "gpt_calls": plan_idx + (1 if cfg.use_verification else 0) * plan_idx,
        "gpt_logs": gpt_logs,
        "decomposition": decomposition,
        "total_steps": len(replay_images),
        "total_plans": plan_idx,
    }
    return success, replay_images, stats


# =============================================================================
# Video utilities
# =============================================================================

def add_text_overlay(img: np.ndarray, text: str) -> np.ndarray:
    """Add colored text banner to top of image frame."""
    h, w = img.shape[:2]
    banner_h = 30
    canvas = np.zeros((h + banner_h, w, 3), dtype=np.uint8)

    if "gpt_langact" in text:
        banner_color = (50, 160, 50)
    elif "openvla_langact" in text:
        banner_color = (180, 100, 30)
    elif "openvla_only" in text:
        banner_color = (50, 50, 180)
    else:
        banner_color = (80, 80, 80)

    canvas[:banner_h, :] = banner_color
    canvas[banner_h:, :] = img

    pil_img = Image.fromarray(canvas)
    draw = ImageDraw.Draw(pil_img)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
    except (IOError, OSError):
        font = ImageFont.load_default()
    draw.text((8, 5), text, fill=(255, 255, 255), font=font)
    return np.array(pil_img)


def save_video(frames: List[np.ndarray], path: str, fps: int = 30) -> None:
    """Save frames as MP4."""
    if not frames:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    writer = imageio.get_writer(path, fps=fps)
    for frame in frames:
        writer.append_data(frame)
    writer.close()


def save_side_by_side_video(
    all_frames: Dict[str, List[np.ndarray]],
    path: str,
    fps: int = 30,
) -> None:
    """Create a side-by-side comparison video from multiple modes."""
    modes = list(all_frames.keys())
    if not modes:
        return
    max_len = max(len(all_frames[m]) for m in modes)
    if max_len == 0:
        return

    padded = {}
    for m in modes:
        frames = list(all_frames[m])
        if not frames:
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
        panels = [add_text_overlay(padded[m][t], m) for m in modes]
        max_h = max(p.shape[0] for p in panels)
        max_w = max(p.shape[1] for p in panels)
        aligned = []
        for p in panels:
            if p.shape[0] < max_h or p.shape[1] < max_w:
                pad = np.zeros((max_h, max_w, 3), dtype=np.uint8)
                pad[: p.shape[0], : p.shape[1]] = p
                aligned.append(pad)
            else:
                aligned.append(p)
        sep = np.ones((max_h, 2, 3), dtype=np.uint8) * 255
        parts = []
        for i, panel in enumerate(aligned):
            if i > 0:
                parts.append(sep)
            parts.append(panel)
        writer.append_data(np.concatenate(parts, axis=1))
    writer.close()


# =============================================================================
# Episode runners
# =============================================================================

def run_episode_gpt_langact(
    cfg,
    env,
    task_description: str,
    model,
    resize_size: int,
    gpt_client,
    processor=None,
    action_head=None,
    proprio_projector=None,
    noisy_action_projector=None,
    initial_state=None,
    save_dir: Optional[str] = None,
) -> Tuple[bool, List[np.ndarray], dict]:
    """Run a single episode with GPT generating langact prompts for the VLA.

    Loop: observe → GPT plans → langact prompt → VLA executes 8 steps → repeat.
    """
    env.reset()
    if initial_state is not None:
        obs = env.set_init_state(initial_state)
    else:
        obs = env.get_observation()

    max_steps = TASK_MAX_STEPS[cfg.task_suite_name]
    t = 0
    replay_images = []
    success = False

    # Stabilization
    for _ in range(cfg.num_steps_wait):
        obs, _, _, _ = env.step(get_libero_dummy_action(cfg.model_family))
        t += 1

    # Task decomposition
    _, init_img = prepare_observation(obs, resize_size)
    decomposition = None
    if cfg.use_decomposition:
        decomposition = query_gpt_decompose(gpt_client, task_description, init_img, cfg.gpt_model)
        logger.info(f"  Task decomposition: {decomposition.get('subtasks', [])}")

    history: List[dict] = []
    prev_obs_img: Optional[np.ndarray] = None
    prev_obs_raw: Optional[dict] = None
    chunk_idx = 0
    gpt_logs: List[dict] = []

    ep_save_dir = os.path.join(save_dir, "gpt_langact") if save_dir else None

    try:
        while t < max_steps + cfg.num_steps_wait:
            observation, img = prepare_observation(obs, resize_size)
            replay_images.append(img)

            # Query GPT for langact prompt
            langact_prompt, parsed_json, raw_response = query_gpt_langact(
                client=gpt_client,
                task_description=task_description,
                curr_obs_img=img,
                curr_obs_raw=obs,
                prev_obs_img=prev_obs_img,
                prev_obs_raw=prev_obs_raw,
                history=history,
                decomposition=decomposition,
                gpt_model=cfg.gpt_model,
                include_icl=cfg.use_icl,
                use_image_icl=cfg.use_image_icl,
                icl_trajectories_dir=cfg.icl_trajectories_dir or None,
            )

            logger.info(f"  [chunk {chunk_idx}, step {t}] GPT → \"{langact_prompt}\"")
            if parsed_json.get("subtask"):
                logger.info(f"    subtask: {parsed_json['subtask']}, confidence: {parsed_json.get('confidence', '?')}")

            # Save full debug info for this chunk
            if ep_save_dir:
                chunk_dir = os.path.join(ep_save_dir, f"chunk_{chunk_idx:03d}")
                os.makedirs(chunk_dir, exist_ok=True)
                save_obs_image(img, os.path.join(chunk_dir, "current_obs.png"))
                if prev_obs_img is not None:
                    save_obs_image(prev_obs_img, os.path.join(chunk_dir, "previous_obs.png"))
                proprio_text = format_proprio_state(obs, prev_obs=prev_obs_raw)
                gpt_input = {
                    "task_description": task_description,
                    "decomposition": decomposition,
                    "proprio_state": proprio_text,
                    "history": [
                        {
                            "chunk": h["chunk"],
                            "prompt": h["prompt"],
                            "observed_motion": h["observed_motion"],
                        }
                        for h in history[-5:]
                    ],
                    "chunk_idx": chunk_idx,
                    "step": t,
                    "use_icl": cfg.use_icl,
                    "use_image_icl": cfg.use_image_icl,
                }
                with open(os.path.join(chunk_dir, "gpt_input.json"), "w") as f:
                    json.dump(gpt_input, f, indent=2)
                with open(os.path.join(chunk_dir, "gpt_response.json"), "w") as f:
                    json.dump({"parsed": parsed_json, "raw": raw_response}, f, indent=2)

            # VLA: predict action chunk using the langact prompt as task_label
            obs_before = obs
            with torch.no_grad():
                actions = get_action(
                    cfg=cfg,
                    model=model,
                    obs=observation,
                    task_label=langact_prompt,
                    processor=processor,
                    action_head=action_head,
                    proprio_projector=proprio_projector,
                    noisy_action_projector=noisy_action_projector,
                    use_film=cfg.use_film,
                )

            # Execute the 8-step action chunk
            done = False
            for action in actions:
                action = process_action(action, cfg.model_family)
                obs, reward, done, info = env.step(action.tolist())
                _, step_img = prepare_observation(obs, resize_size)
                replay_images.append(step_img)
                t += 1
                if done:
                    success = True
                    break

            # Compute observed motion for history / verification
            observed_motion = compute_observed_motion(obs_before, obs)

            # Update history
            history_entry = {
                "chunk": chunk_idx,
                "step": t,
                "prompt": langact_prompt,
                "observed_motion": observed_motion,
                "gpt_json": parsed_json,
            }
            history.append(history_entry)
            gpt_logs.append({
                **history_entry,
                "raw_response": raw_response,
            })

            logger.info(f"    observed: \"{observed_motion}\"")

            # Save post-action state into the same chunk dir
            if ep_save_dir:
                chunk_dir = os.path.join(ep_save_dir, f"chunk_{chunk_idx:03d}")
                _, post_img = prepare_observation(obs, resize_size)
                save_obs_image(post_img, os.path.join(chunk_dir, "post_action_obs.png"))
                post_proprio = format_proprio_state(obs, prev_obs=obs_before)
                outcome = {
                    "commanded_prompt": langact_prompt,
                    "observed_motion": observed_motion,
                    "post_action_proprio": post_proprio,
                    "done": done,
                    "success": success,
                }
                with open(os.path.join(chunk_dir, "chunk_outcome.json"), "w") as f:
                    json.dump(outcome, f, indent=2)

            prev_obs_img = img.copy()
            prev_obs_raw = obs_before
            chunk_idx += 1

            if done:
                break

            # Early termination if GPT says task is complete
            if parsed_json.get("task_complete", False):
                logger.info("  GPT signaled task_complete=true, ending episode.")
                break

    except Exception as e:
        logger.error(f"Episode error: {e}", exc_info=True)

    # Save full episode summary
    if ep_save_dir:
        episode_summary = {
            "task_description": task_description,
            "decomposition": decomposition,
            "success": success,
            "total_chunks": chunk_idx,
            "total_steps": len(replay_images),
            "gpt_model": cfg.gpt_model,
            "use_icl": cfg.use_icl,
            "use_image_icl": cfg.use_image_icl,
            "use_decomposition": cfg.use_decomposition,
            "chunk_log": [
                {
                    "chunk": g["chunk"],
                    "step": g["step"],
                    "prompt": g["prompt"],
                    "observed_motion": g["observed_motion"],
                    "gpt_reasoning": g["gpt_json"].get("reasoning", ""),
                    "gpt_subtask": g["gpt_json"].get("subtask", ""),
                    "gpt_confidence": g["gpt_json"].get("confidence", ""),
                    "gpt_observation": g["gpt_json"].get("observation", ""),
                }
                for g in gpt_logs
            ],
        }
        with open(os.path.join(ep_save_dir, "episode_summary.json"), "w") as f:
            json.dump(episode_summary, f, indent=2)

    stats = {
        "gpt_calls": chunk_idx,
        "gpt_logs": gpt_logs,
        "decomposition": decomposition,
        "total_steps": len(replay_images),
    }
    return success, replay_images, stats


def run_episode_openvla_langact(
    cfg,
    env,
    task_description: str,
    model,
    resize_size: int,
    processor=None,
    action_head=None,
    proprio_projector=None,
    noisy_action_projector=None,
    initial_state=None,
    initial_prompt: str = "move forward 3.0 cm, move down 2.0 cm, close gripper",
) -> Tuple[bool, List[np.ndarray], dict]:
    """Baseline: VLA with auto-computed langact prompts (no GPT).

    Uses the initial_prompt for the first chunk, then computes the actual
    observed motion after each chunk and uses that as the next prompt.
    """
    env.reset()
    if initial_state is not None:
        obs = env.set_init_state(initial_state)
    else:
        obs = env.get_observation()

    max_steps = TASK_MAX_STEPS[cfg.task_suite_name]
    t = 0
    replay_images = []
    success = False

    for _ in range(cfg.num_steps_wait):
        obs, _, _, _ = env.step(get_libero_dummy_action(cfg.model_family))
        t += 1

    current_prompt = initial_prompt
    chunk_idx = 0

    try:
        while t < max_steps + cfg.num_steps_wait:
            observation, img = prepare_observation(obs, resize_size)
            replay_images.append(img)

            obs_before = obs

            with torch.no_grad():
                actions = get_action(
                    cfg=cfg, model=model, obs=observation, task_label=current_prompt,
                    processor=processor, action_head=action_head,
                    proprio_projector=proprio_projector,
                    noisy_action_projector=noisy_action_projector,
                    use_film=cfg.use_film,
                )

            done = False
            for action in actions:
                action = process_action(action, cfg.model_family)
                obs, reward, done, info = env.step(action.tolist())
                _, step_img = prepare_observation(obs, resize_size)
                replay_images.append(step_img)
                t += 1
                if done:
                    success = True
                    break

            current_prompt = compute_observed_motion(obs_before, obs)
            logger.info(f"  [auto-langact chunk {chunk_idx}] → \"{current_prompt}\"")
            chunk_idx += 1

            if done:
                break

    except Exception as e:
        logger.error(f"Episode error: {e}", exc_info=True)

    return success, replay_images, {"gpt_calls": 0, "gpt_logs": [], "total_steps": len(replay_images)}


def run_episode_openvla_only(
    cfg,
    env,
    task_description: str,
    model,
    resize_size: int,
    processor=None,
    action_head=None,
    proprio_projector=None,
    noisy_action_projector=None,
    initial_state=None,
) -> Tuple[bool, List[np.ndarray], dict]:
    """Baseline: standard VLA with task description (no langact, no GPT)."""
    env.reset()
    if initial_state is not None:
        obs = env.set_init_state(initial_state)
    else:
        obs = env.get_observation()

    action_queue = deque(maxlen=cfg.num_open_loop_steps)
    max_steps = TASK_MAX_STEPS[cfg.task_suite_name]
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
            obs, reward, done, info = env.step(action.tolist())
            if done:
                success = True
                break
            t += 1

    except Exception as e:
        logger.error(f"Episode error: {e}", exc_info=True)

    return success, replay_images, {"gpt_calls": 0, "gpt_logs": [], "total_steps": len(replay_images)}


# =============================================================================
# Main evaluation
# =============================================================================

@dataclass
class GPTLangactEvalConfig(BaseGenerateConfig):
    # GPT settings
    eval_mode: str = "gpt_langact"       # "gpt_langact", "gpt_multistep", "openvla_langact", "openvla_only", "all"
    gpt_model: str = "gpt-4o"
    use_icl: bool = True                 # Include text-based in-context trajectory examples
    use_image_icl: bool = False          # Include image-based ICL from extracted trajectories
    icl_trajectories_dir: str = ""       # Path to extracted ICL dir (empty = default)
    use_decomposition: bool = True       # Task decomposition before execution
    plan_horizon: int = 3                # Number of chunks in each multi-step plan (gpt_multistep mode)
    use_verification: bool = True        # Verification/self-correction after plan execution (gpt_multistep mode)
    video_dir: str = "./rollout_videos_langact"
    video_fps: int = 30
    initial_langact_prompt: str = "move forward 3.0 cm, move down 2.0 cm, close gripper"
    custom_task_description: str = ""    # Override task name for GPT planning


@draccus.wrap()
def eval_gpt_langact(cfg: GPTLangactEvalConfig) -> None:
    """Run LIBERO evaluation with GPT-generated langact prompts."""
    validate_config(cfg)
    set_seed_everywhere(cfg.seed)

    model, action_head, proprio_projector, noisy_action_projector, processor = initialize_model(cfg)
    resize_size = get_image_resize_size(cfg)

    gpt_client = None
    if cfg.eval_mode in ("gpt_langact", "gpt_multistep", "all"):
        from openai import OpenAI
        gpt_client = OpenAI()

    if cfg.eval_mode == "all":
        eval_modes = ["openvla_only", "openvla_langact", "gpt_langact", "gpt_multistep"]
    else:
        eval_modes = [cfg.eval_mode]

    os.makedirs(cfg.local_log_dir, exist_ok=True)
    run_id = f"GPT-LANGACT-{cfg.task_suite_name}-{DATE_TIME}"
    log_filepath = os.path.join(cfg.local_log_dir, run_id + ".txt")
    log_file = open(log_filepath, "w")

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[cfg.task_suite_name]()
    num_tasks = task_suite.n_tasks
    task_start = cfg.task_id_start
    task_end = num_tasks if cfg.task_id_end == -1 else min(cfg.task_id_end, num_tasks)

    log_message(f"Task suite: {cfg.task_suite_name}", log_file)
    log_message(f"Task range: [{task_start}, {task_end})", log_file)
    log_message(f"Eval modes: {eval_modes}", log_file)
    log_message(f"GPT model: {cfg.gpt_model}", log_file)
    log_message(f"ICL: {cfg.use_icl}, Decomposition: {cfg.use_decomposition}", log_file)

    results = {mode: {"total_episodes": 0, "total_successes": 0, "tasks": {}} for mode in eval_modes}

    for task_id in tqdm.tqdm(range(task_start, task_end), desc="Tasks"):
        task = task_suite.get_task(task_id)
        initial_states = task_suite.get_task_init_states(task_id)
        env, task_description = get_libero_env(task, cfg.model_family, resolution=cfg.env_img_res)

        if cfg.custom_task_description:
            log_message(f"Overriding task: '{task_description}' → '{cfg.custom_task_description}'", log_file)
            task_description = cfg.custom_task_description

        task_dir = f"{DATE_TIME}__{task_description.lower().replace(' ', '_').replace('/', '_')[:60]}"

        for ep_idx in tqdm.tqdm(range(cfg.num_trials_per_task), desc=f"Task {task_id}", leave=False):
            initial_state = initial_states[ep_idx]
            ep_video_dir = os.path.join(cfg.video_dir, cfg.task_suite_name, task_dir, f"ep{ep_idx:03d}")

            mode_frames = {}
            mode_success = {}

            for eval_mode in eval_modes:
                log_message(
                    f"\nTask {task_id}: {task_description} | Ep {ep_idx} | Mode: {eval_mode}",
                    log_file,
                )

                if eval_mode == "gpt_langact":
                    success, frames, stats = run_episode_gpt_langact(
                        cfg, env, task_description, model, resize_size, gpt_client,
                        processor, action_head, proprio_projector, noisy_action_projector,
                        initial_state, save_dir=ep_video_dir,
                    )
                elif eval_mode == "gpt_multistep":
                    success, frames, stats = run_episode_gpt_multistep(
                        cfg, env, task_description, model, resize_size, gpt_client,
                        processor, action_head, proprio_projector, noisy_action_projector,
                        initial_state, save_dir=ep_video_dir,
                    )
                elif eval_mode == "openvla_langact":
                    success, frames, stats = run_episode_openvla_langact(
                        cfg, env, task_description, model, resize_size,
                        processor, action_head, proprio_projector, noisy_action_projector,
                        initial_state, initial_prompt=cfg.initial_langact_prompt,
                    )
                elif eval_mode == "openvla_only":
                    success, frames, stats = run_episode_openvla_only(
                        cfg, env, task_description, model, resize_size,
                        processor, action_head, proprio_projector, noisy_action_projector,
                        initial_state,
                    )
                else:
                    raise ValueError(f"Unknown eval_mode: {eval_mode}")

                mode_frames[eval_mode] = frames
                mode_success[eval_mode] = success

                results[eval_mode]["total_episodes"] += 1
                if success:
                    results[eval_mode]["total_successes"] += 1
                task_results = results[eval_mode]["tasks"].setdefault(
                    task_description, {"episodes": 0, "successes": 0},
                )
                task_results["episodes"] += 1
                if success:
                    task_results["successes"] += 1

                log_message(
                    f"  [{eval_mode}] success={success} steps={stats['total_steps']} "
                    f"gpt_calls={stats['gpt_calls']}",
                    log_file,
                )

                video_path = os.path.join(ep_video_dir, f"{eval_mode}--success={success}.mp4")
                save_video(frames, video_path, fps=cfg.video_fps)

                if stats.get("gpt_logs"):
                    os.makedirs(ep_video_dir, exist_ok=True)
                    meta = {
                        "task_description": task_description,
                        "task_id": task_id,
                        "episode_idx": ep_idx,
                        "eval_mode": eval_mode,
                        "success": success,
                        "total_steps": stats["total_steps"],
                        "gpt_calls": stats["gpt_calls"],
                        "decomposition": stats.get("decomposition"),
                    }
                    log_path = os.path.join(ep_video_dir, f"{eval_mode}--gpt_logs.json")
                    with open(log_path, "w") as f:
                        json.dump({**meta, "chunks": stats["gpt_logs"]}, f, indent=2, default=str)

                    reasoning_path = os.path.join(ep_video_dir, f"{eval_mode}--reasoning.txt")
                    with open(reasoning_path, "w") as rf:
                        rf.write(f"Task: {task_description}\n")
                        rf.write(f"Mode: {eval_mode}\n")
                        rf.write(f"Success: {success}\n")
                        rf.write(f"Decomposition: {stats.get('decomposition', {}).get('subtasks', 'N/A')}\n")
                        rf.write("=" * 60 + "\n\n")
                        for chunk in stats["gpt_logs"]:
                            rf.write(f"--- Chunk {chunk['chunk']}, Step {chunk['step']} ---\n")
                            gpt_json = chunk.get("gpt_json", {})
                            rf.write(f"Prompt: {chunk['prompt']}\n")
                            rf.write(f"Observed: {chunk['observed_motion']}\n")
                            rf.write(f"Subtask: {gpt_json.get('subtask', '?')}\n")
                            rf.write(f"Observation: {gpt_json.get('observation', '?')}\n")
                            rf.write(f"Progress: {gpt_json.get('progress', '?')}\n")
                            rf.write(f"Reasoning: {gpt_json.get('reasoning', '?')}\n")
                            rf.write(f"Confidence: {gpt_json.get('confidence', '?')}\n")
                            rf.write("\n")

            if len(mode_frames) > 1:
                labeled = {
                    f"{m} ({'OK' if mode_success[m] else 'FAIL'})": mode_frames[m]
                    for m in eval_modes if m in mode_frames
                }
                sbs_path = os.path.join(ep_video_dir, "side_by_side.mp4")
                save_side_by_side_video(labeled, sbs_path, fps=cfg.video_fps)

    # Final summary
    log_message("\n" + "=" * 70, log_file)
    log_message("FINAL RESULTS", log_file)
    log_message("=" * 70, log_file)

    for mode in eval_modes:
        r = results[mode]
        rate = r["total_successes"] / r["total_episodes"] if r["total_episodes"] > 0 else 0
        log_message(f"  {mode:20s}: {rate:.4f} ({r['total_successes']}/{r['total_episodes']})", log_file)
        for desc, tr in r["tasks"].items():
            task_rate = tr["successes"] / tr["episodes"] if tr["episodes"] > 0 else 0
            log_message(f"    {desc:50s}: {task_rate:.3f} ({tr['successes']}/{tr['episodes']})", log_file)

    results_path = os.path.join(cfg.local_log_dir, run_id + "_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    log_message(f"\nResults: {results_path}", log_file)
    log_message(f"Videos: {cfg.video_dir}", log_file)
    log_file.close()


if __name__ == "__main__":
    eval_gpt_langact()
