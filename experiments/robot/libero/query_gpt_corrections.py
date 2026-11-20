"""
query_gpt_corrections.py

Loads collected LIBERO episode data and queries GPT-4o to correct OpenVLA-OFT
action chunks. Two modes:

  1. "normal"   - GPT outputs the full corrected 8-action chunk (absolute values)
  2. "residual" - GPT outputs exact residual deltas to ADD to OpenVLA's actions

Usage:
    # Query GPT on collected data (normal mode)
    python experiments/robot/libero/query_gpt_corrections.py \
        --collect_dir ./collected_data/libero_spatial \
        --output_dir ./gpt_corrections/libero_spatial \
        --mode normal \
        --max_episodes 5 \
        --max_chunks_per_episode 5

    # Query GPT on collected data (residual mode)
    python experiments/robot/libero/query_gpt_corrections.py \
        --collect_dir ./collected_data/libero_spatial \
        --output_dir ./gpt_corrections/libero_spatial \
        --mode residual \
        --max_episodes 5 \
        --max_chunks_per_episode 5

    # Both modes at once
    python experiments/robot/libero/query_gpt_corrections.py \
        --collect_dir ./collected_data/libero_spatial \
        --output_dir ./gpt_corrections/libero_spatial \
        --mode both \
        --max_episodes 5

Set OPENAI_API_KEY environment variable before running.
"""

import argparse
import base64
import io
import json
import os
import pickle
import time
from glob import glob
from typing import List, Optional

import numpy as np
from PIL import Image


# ---------------------------------------------------------------------------
# Image encoding
# ---------------------------------------------------------------------------

def encode_image_to_base64(img: np.ndarray) -> str:
    """Convert numpy RGB image (H,W,3 uint8) to base64 PNG string."""
    pil_img = Image.fromarray(img)
    buffer = io.BytesIO()
    pil_img.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


# ---------------------------------------------------------------------------
# GPT prompts
# ---------------------------------------------------------------------------

SYSTEM_PROMPT_NORMAL = """\
You are a robot action correction system for a tabletop manipulation task.

You receive:
- Two RGB images from the robot's third-person camera:
  "Previous observation" (from ~8 control steps ago) and "Current observation".
- A task description (what the robot should accomplish).
- A proposed 8-step action chunk from a learned policy.

Each action is a 7-dimensional vector:
  [dx, dy, dz, ax, ay, az, gripper]
where:
  dx, dy, dz  = end-effector position deltas (typically in [-0.05, 0.05])
  ax, ay, az  = end-effector orientation deltas (axis-angle, small values)
  gripper      = gripper command: values near 0 mean open, values near 1 mean close
                 (after post-processing: -1 = open, +1 = close)

Follow these steps:
1. EVALUATE: Look at the current observation and the task description. Assess \
whether the proposed action chunk is reasonable — is it moving toward the right \
object? Is the gripper opening/closing at the right time? Is the direction correct?
2. DECIDE: If the proposed actions already look good, output them unchanged. \
Do not modify actions that are already correct.
3. CORRECT: If the actions have problems, reason about the failure mode \
(e.g., wrong direction, gripper should be open instead of closed, overshooting) \
and output corrected actions.

Output format: First write a brief assessment (1-2 sentences), then output \
a JSON array of 8 arrays, each with 7 floats. The JSON must be inside a \
```json code block.\
"""

SYSTEM_PROMPT_RESIDUAL = """\
You are a robot action residual correction system for a tabletop manipulation task.

You receive:
- Two RGB images from the robot's third-person camera:
  "Previous observation" (from ~8 control steps ago) and "Current observation".
- A task description (what the robot should accomplish).
- A proposed 8-step action chunk from a learned policy.

Each action is a 7-dimensional vector:
  [dx, dy, dz, ax, ay, az, gripper]
where:
  dx, dy, dz  = end-effector position deltas (typically in [-0.05, 0.05])
  ax, ay, az  = end-effector orientation deltas (axis-angle, small values)
  gripper      = gripper command: values near 0 mean open, values near 1 mean close
                 (after post-processing: -1 = open, +1 = close)

The corrected action will be computed as:
  corrected_action = proposed_action + your_residual

Follow these steps:
1. EVALUATE: Look at the current observation and the task description. Assess \
whether the proposed action chunk is reasonable — is it moving toward the right \
object? Is the gripper opening/closing at the right time? Is the direction correct?
2. DECIDE: If the proposed actions already look good, output all zeros \
(no correction needed). Do not add residuals to actions that are already correct.
3. CORRECT: If the actions have problems, reason about the failure mode \
(e.g., wrong direction, gripper should be open instead of closed, overshooting) \
and output residual corrections. Keep residuals small (position: [-0.03, 0.03]).

Output format: First write a brief assessment (1-2 sentences), then output \
a JSON array of 8 arrays, each with 7 floats (the residuals). The JSON must \
be inside a ```json code block.\
"""


def build_user_message(
    task_description: str,
    curr_obs_img: np.ndarray,
    openvla_actions: list,
    prev_obs_img: Optional[np.ndarray] = None,
    mode: str = "normal",
) -> list:
    """Build the user message content for the GPT API call."""
    actions_formatted = []
    for i, a in enumerate(openvla_actions):
        vals = a if isinstance(a, list) else a.tolist()
        actions_formatted.append({"step": i, "action": [round(v, 5) for v in vals]})
    actions_str = json.dumps(actions_formatted, indent=2)

    content = []
    content.append({"type": "text", "text": f"Task: {task_description}"})

    # Previous observation
    if prev_obs_img is not None:
        content.append({"type": "text", "text": "Previous observation (~8 steps ago):"})
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{encode_image_to_base64(prev_obs_img)}"},
        })
    else:
        content.append({"type": "text", "text": "(No previous observation available — this is the first chunk.)"})

    # Current observation
    content.append({"type": "text", "text": "Current observation:"})
    content.append({
        "type": "image_url",
        "image_url": {"url": f"data:image/png;base64,{encode_image_to_base64(curr_obs_img)}"},
    })

    # Proposed actions
    if mode == "normal":
        content.append({
            "type": "text",
            "text": (
                f"Proposed actions from policy:\n{actions_str}\n\n"
                "Output the corrected 8-step action chunk as a JSON array of 8 arrays (each 7 floats):"
            ),
        })
    else:  # residual
        content.append({
            "type": "text",
            "text": (
                f"Proposed actions from policy:\n{actions_str}\n\n"
                "Output residual corrections as a JSON array of 8 arrays (each 7 floats). "
                "The final action = proposed + residual. Output zeros where no correction is needed:"
            ),
        })

    return content


# ---------------------------------------------------------------------------
# GPT query
# ---------------------------------------------------------------------------

def query_gpt(
    client,
    task_description: str,
    curr_obs_img: np.ndarray,
    openvla_actions: list,
    prev_obs_img: Optional[np.ndarray] = None,
    mode: str = "normal",
    gpt_model: str = "gpt-4o",
    temperature: float = 0.0,
    max_retries: int = 3,
) -> dict:
    """
    Query GPT for action correction.

    Returns dict with:
        "raw_response": str (GPT's raw text output)
        "parsed_actions": list of 8 lists of 7 floats, or None if parsing failed
        "mode": "normal" or "residual"
        "corrected_actions": list of 8 np arrays (final actions after applying correction)
        "error": str or None
    """
    system_prompt = SYSTEM_PROMPT_NORMAL if mode == "normal" else SYSTEM_PROMPT_RESIDUAL
    user_content = build_user_message(
        task_description, curr_obs_img, openvla_actions, prev_obs_img, mode
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=gpt_model,
                messages=messages,
                temperature=temperature,
                max_tokens=2000,
            )
            raw = response.choices[0].message.content.strip()

            # Parse JSON from response (handle markdown fences)
            text = raw
            if "```" in text:
                # Extract content between code fences
                parts = text.split("```")
                for part in parts[1:]:
                    # Skip the language identifier line if present
                    lines = part.strip().split("\n")
                    if lines[0].strip().lower() in ("json", "python", ""):
                        text = "\n".join(lines[1:])
                    else:
                        text = part
                    break
                # Remove trailing fence
                if text.endswith("```"):
                    text = text[:-3]

            parsed = json.loads(text.strip())

            # Validate shape
            assert isinstance(parsed, list) and len(parsed) == 8
            for row in parsed:
                assert isinstance(row, list) and len(row) == 7

            # Compute corrected actions
            if mode == "normal":
                corrected = [np.array(a) for a in parsed]
            else:  # residual
                corrected = []
                for i, residual in enumerate(parsed):
                    orig = np.array(openvla_actions[i]) if isinstance(openvla_actions[i], list) else openvla_actions[i]
                    corrected.append(orig + np.array(residual))

            return {
                "raw_response": raw,
                "parsed_actions": parsed,
                "mode": mode,
                "corrected_actions": [a.tolist() for a in corrected],
                "error": None,
                "usage": {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens,
                },
            }

        except Exception as e:
            if attempt < max_retries - 1:
                print(f"  Attempt {attempt + 1} failed: {e}. Retrying...")
                time.sleep(2)
            else:
                return {
                    "raw_response": raw if "raw" in dir() else "",
                    "parsed_actions": None,
                    "mode": mode,
                    "corrected_actions": None,
                    "error": str(e),
                    "usage": None,
                }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Query GPT for action corrections on collected LIBERO data")
    parser.add_argument("--collect_dir", type=str, required=True,
                        help="Directory with collected .pkl episode files")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save GPT correction results")
    parser.add_argument("--mode", type=str, default="both", choices=["normal", "residual", "both"],
                        help="Correction mode: normal, residual, or both")
    parser.add_argument("--gpt_model", type=str, default="gpt-4o",
                        help="GPT model to use")
    parser.add_argument("--max_episodes", type=int, default=None,
                        help="Max number of episodes to process (None = all)")
    parser.add_argument("--max_chunks_per_episode", type=int, default=None,
                        help="Max chunks to query per episode (None = all)")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="GPT sampling temperature")
    parser.add_argument("--skip_first_chunk", action="store_true",
                        help="Skip first chunk of each episode (no prev_obs available)")
    args = parser.parse_args()

    # Import OpenAI here so the collection script doesn't need it
    from openai import OpenAI
    client = OpenAI()

    # Determine modes to run
    modes = ["normal", "residual"] if args.mode == "both" else [args.mode]

    # Find episode files
    episode_files = sorted(glob(os.path.join(args.collect_dir, "*.pkl")))
    if args.max_episodes is not None:
        episode_files = episode_files[:args.max_episodes]

    print(f"Found {len(episode_files)} episode files in {args.collect_dir}")
    print(f"Modes: {modes}")
    print(f"GPT model: {args.gpt_model}")

    # Process each mode
    for mode in modes:
        mode_dir = os.path.join(args.output_dir, mode)
        os.makedirs(mode_dir, exist_ok=True)
        print(f"\n{'='*60}")
        print(f"Mode: {mode}")
        print(f"Output: {mode_dir}")
        print(f"{'='*60}")

        total_chunks = 0
        total_cost_tokens = 0

        for ep_file in episode_files:
            ep_name = os.path.splitext(os.path.basename(ep_file))[0]
            result_path = os.path.join(mode_dir, f"{ep_name}.json")

            # Skip if already processed
            if os.path.exists(result_path):
                print(f"  Skipping {ep_name} (already processed)")
                continue

            # Load episode data
            with open(ep_file, "rb") as f:
                ep_data = pickle.load(f)

            task_desc = ep_data["task_description"]
            chunks = ep_data["chunks"]
            openvla_success = ep_data["success"]

            if args.max_chunks_per_episode is not None:
                chunks = chunks[:args.max_chunks_per_episode]

            print(f"\n  Episode: {ep_name}")
            print(f"  Task: {task_desc}")
            print(f"  OpenVLA success: {openvla_success}")
            print(f"  Chunks to process: {len(chunks)}")

            ep_results = {
                "episode_file": ep_file,
                "task_description": task_desc,
                "openvla_success": openvla_success,
                "mode": mode,
                "gpt_model": args.gpt_model,
                "chunk_results": [],
            }

            for chunk in chunks:
                chunk_idx = chunk["chunk_idx"]

                # Optionally skip first chunk (no prev observation)
                if args.skip_first_chunk and chunk["prev_obs_img"] is None:
                    print(f"    Chunk {chunk_idx}: skipped (no prev obs)")
                    continue

                print(f"    Chunk {chunk_idx} (step {chunk['step']})...", end=" ", flush=True)

                result = query_gpt(
                    client=client,
                    task_description=task_desc,
                    curr_obs_img=chunk["curr_obs_img"],
                    openvla_actions=chunk["openvla_actions"],
                    prev_obs_img=chunk["prev_obs_img"],
                    mode=mode,
                    gpt_model=args.gpt_model,
                    temperature=args.temperature,
                )

                # Add original actions for reference
                result["openvla_actions"] = [
                    a.tolist() if hasattr(a, "tolist") else a
                    for a in chunk["openvla_actions"]
                ]
                result["chunk_idx"] = chunk_idx
                result["step"] = chunk["step"]

                ep_results["chunk_results"].append(result)
                total_chunks += 1

                if result["error"]:
                    print(f"ERROR: {result['error']}")
                else:
                    tokens = result["usage"]["total_tokens"] if result["usage"] else 0
                    total_cost_tokens += tokens
                    print(f"OK ({tokens} tokens)")

                # Rate limiting
                time.sleep(0.5)

            # Save episode results
            with open(result_path, "w") as f:
                json.dump(ep_results, f, indent=2)
            print(f"  Saved: {result_path}")

        print(f"\n{mode} mode complete: {total_chunks} chunks, ~{total_cost_tokens} tokens total")

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Collected data dir: {args.collect_dir}")
    print(f"Output dir:         {args.output_dir}")
    print(f"Episodes processed: {len(episode_files)}")
    print(f"Modes:              {modes}")
    print("\nTo analyze results, load the JSON files:")
    print("  import json")
    print(f'  with open("{args.output_dir}/<mode>/<episode>.json") as f:')
    print("      results = json.load(f)")
    print("  for chunk in results['chunk_results']:")
    print("      print(chunk['openvla_actions'])")
    print("      print(chunk['corrected_actions'])")


if __name__ == "__main__":
    main()
