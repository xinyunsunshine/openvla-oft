"""
Extract complete trajectory ICL examples from langact RLDS datasets.

Picks one trajectory from each of 3 LIBERO suites, saves observation images
at regular intervals and the full sequence of langact prompts. The output
is a structured directory + metadata JSON that can be loaded by
run_libero_eval_gpt_langact.py as image-based in-context learning examples.

Usage:
    cd /proj/inf-scaling/zwhong/idan/vla-rl/openvla-oft

    python scripts/libero_langact/extract_icl_trajectories.py \
        --data_dir /proj/inf-scaling/SWE_sim/modified_libero_rlds_new \
        --output_dir ./icl_langact_trajectories \
        --image_every 8 \
        --seed 42

Output:
    icl_langact_trajectories/
      icl_langact_metadata.json       # Full metadata for all trajectories
      traj_0_libero_goal/
        obs_step_000.png              # Observation at step 0
        obs_step_008.png              # Observation at step 8
        ...
      traj_1_libero_object/
        obs_step_000.png
        ...
      traj_2_libero_10/
        obs_step_000.png
        ...
"""

import argparse
import json
import os
import sys

import numpy as np
from PIL import Image

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import tensorflow as tf
tf.config.set_visible_devices([], "GPU")
import tensorflow_datasets as tfds

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from langact_utils import (
    axis_angle_to_euler,
    compute_language_actions,
    aggregate_language_actions,
    summarize_numeric_actions,
)


# ─── Configuration ────────────────────────────────────────────────────────────

# Default: pick one task from each of 3 suites (goal, object, 10)
DEFAULT_TRAJECTORIES = [
    {
        "suite": "libero_goal",
        "dataset": "libero_goal_no_noops_langact_verbose",
        "original_dataset": "libero_goal_no_noops",
        "episode_idx": 5,
        "description": "A pick-and-place task from libero_goal",
    },
    {
        "suite": "libero_object",
        "dataset": "libero_object_no_noops_langact_verbose",
        "original_dataset": "libero_object_no_noops",
        "episode_idx": 3,
        "description": "An object manipulation task from libero_object",
    },
    {
        "suite": "libero_10",
        "dataset": "libero_10_no_noops_langact_verbose",
        "original_dataset": "libero_10_no_noops",
        "episode_idx": 2,
        "description": "A long-horizon task from libero_10",
    },
]

WINDOW_SIZE = 8  # matches NUM_ACTIONS_CHUNK


def extract_trajectory(data_dir, dataset_name, episode_idx, output_dir,
                        image_every=8, traj_label="traj",
                        original_dataset=None):
    """Extract a single trajectory from an RLDS dataset.

    Args:
        original_dataset: Name of the original (non-langact) dataset to
            retrieve the true task description from.

    Returns a dict with the full trajectory metadata.
    """
    print(f"\n{'='*60}")
    print(f"Loading {dataset_name}, episode {episode_idx}")
    print(f"{'='*60}")

    builder = tfds.builder(dataset_name, data_dir=data_dir)
    ds = builder.as_dataset(split="train")

    for i, episode in enumerate(ds):
        if i == episode_idx:
            break
    else:
        raise ValueError(f"Episode {episode_idx} not found (only {i+1} episodes)")

    steps = list(episode["steps"])
    T = len(steps)
    print(f"  Episode has {T} steps")

    images = []
    states = np.zeros((T, 8), dtype=np.float64)
    actions = np.zeros((T, 7), dtype=np.float64)
    langact_prompts = []

    for t, step in enumerate(steps):
        img = step["observation"]["image"].numpy()  # (256, 256, 3)
        state = step["observation"]["state"].numpy()  # (8,)
        action = step["action"].numpy()  # (7,)
        lang = step["language_instruction"].numpy().decode("utf-8")

        images.append(img)
        states[t] = state
        actions[t] = action
        langact_prompts.append(lang)

    # Get the true task description from the original dataset
    task_description = ""
    if original_dataset:
        try:
            orig_builder = tfds.builder(original_dataset, data_dir=data_dir)
            orig_ds = orig_builder.as_dataset(split="train")
            for j, orig_ep in enumerate(orig_ds):
                if j == episode_idx:
                    orig_steps = list(orig_ep["steps"])
                    task_description = orig_steps[0]["language_instruction"].numpy().decode("utf-8")
                    print(f"  Task: \"{task_description}\"")
                    break
        except Exception as e:
            print(f"  WARNING: Could not load original dataset: {e}")

    if not task_description:
        task_description = langact_prompts[0]

    # Recompute the ground-truth langact from states/actions to get clean prompts
    lang_actions = compute_language_actions(states, actions)  # (T, 7)
    aggregated = aggregate_language_actions(lang_actions, window_size=WINDOW_SIZE)  # (T, 7)

    clean_prompts = []
    for t in range(T):
        text = summarize_numeric_actions(
            aggregated[t], sum_decimal="1f", include_rotation=True
        )
        clean_prompts.append(text or "close gripper")

    # Build chunk-level data: group by WINDOW_SIZE steps
    num_chunks = (T + WINDOW_SIZE - 1) // WINDOW_SIZE
    chunks = []

    for chunk_idx in range(num_chunks):
        t_start = chunk_idx * WINDOW_SIZE
        t_end = min(t_start + WINDOW_SIZE, T)

        # EEF state at chunk start
        eef_pos = states[t_start, :3]
        eef_euler = axis_angle_to_euler(states[t_start, 3:6].reshape(1, 3))[0]
        grip_qpos = states[t_start, 6:8]

        chunk_info = {
            "chunk_idx": chunk_idx,
            "step_start": t_start,
            "step_end": t_end,
            "prompt": clean_prompts[t_start],
            "dataset_prompt": langact_prompts[t_start],
            "eef_pos_meters": eef_pos.tolist(),
            "eef_pos_cm": (eef_pos * 100).tolist(),
            "eef_euler_deg": np.degrees(eef_euler).tolist(),
            "gripper_qpos": grip_qpos.tolist(),
            "gripper_state": "open" if grip_qpos[0] > 0.03 else "closed",
        }
        chunks.append(chunk_info)

    # Save observation images at requested interval
    os.makedirs(output_dir, exist_ok=True)
    saved_images = []

    for t in range(T):
        if t % image_every == 0 or t == T - 1:
            img_filename = f"obs_step_{t:03d}.png"
            img_path = os.path.join(output_dir, img_filename)
            Image.fromarray(images[t]).save(img_path)
            saved_images.append({
                "step": t,
                "chunk_idx": t // WINDOW_SIZE,
                "filename": img_filename,
                "prompt_at_step": clean_prompts[t],
            })

    print(f"  Saved {len(saved_images)} observation images to {output_dir}")
    print(f"  {num_chunks} action chunks")

    # Print the full prompt sequence
    print(f"\n  Full prompt sequence ({num_chunks} chunks):")
    for c in chunks:
        grip = c["gripper_state"]
        pos = c["eef_pos_cm"]
        print(f"    Chunk {c['chunk_idx']:2d} (step {c['step_start']:3d}): "
              f"\"{c['prompt']}\"  [EEF: x={pos[0]:.1f} y={pos[1]:.1f} z={pos[2]:.1f} cm, grip={grip}]")

    return {
        "dataset": dataset_name,
        "episode_idx": episode_idx,
        "task_description": task_description,
        "total_steps": T,
        "num_chunks": num_chunks,
        "window_size": WINDOW_SIZE,
        "chunks": chunks,
        "saved_images": saved_images,
        "output_dir": output_dir,
    }


def main():
    parser = argparse.ArgumentParser(description="Extract ICL trajectory examples from langact RLDS datasets")
    parser.add_argument("--data_dir", type=str,
                        default="/proj/inf-scaling/SWE_sim/modified_libero_rlds_new",
                        help="Root directory containing RLDS datasets")
    parser.add_argument("--output_dir", type=str,
                        default="./icl_langact_trajectories",
                        help="Output directory for ICL examples")
    parser.add_argument("--image_every", type=int, default=8,
                        help="Save an observation image every N steps (default: 8, i.e. every chunk)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for episode selection")
    parser.add_argument("--trajectories", type=str, default=None,
                        help="JSON file specifying trajectories to extract (overrides defaults)")
    args = parser.parse_args()

    np.random.seed(args.seed)

    # Load trajectory specifications
    if args.trajectories:
        with open(args.trajectories) as f:
            traj_specs = json.load(f)
    else:
        traj_specs = DEFAULT_TRAJECTORIES

    os.makedirs(args.output_dir, exist_ok=True)

    all_metadata = {
        "description": "ICL trajectory examples extracted from langact RLDS datasets",
        "data_dir": args.data_dir,
        "image_every": args.image_every,
        "window_size": WINDOW_SIZE,
        "trajectories": [],
    }

    for i, spec in enumerate(traj_specs):
        suite = spec["suite"]
        dataset = spec["dataset"]
        original_dataset = spec.get("original_dataset", None)
        ep_idx = spec["episode_idx"]
        desc = spec.get("description", "")

        traj_dir_name = f"traj_{i}_{suite}"
        traj_output_dir = os.path.join(args.output_dir, traj_dir_name)

        traj_meta = extract_trajectory(
            data_dir=args.data_dir,
            dataset_name=dataset,
            episode_idx=ep_idx,
            output_dir=traj_output_dir,
            image_every=args.image_every,
            traj_label=f"traj_{i}",
            original_dataset=original_dataset,
        )

        traj_meta["suite"] = suite
        traj_meta["traj_dir"] = traj_dir_name
        traj_meta["user_description"] = desc
        all_metadata["trajectories"].append(traj_meta)

    # Save metadata
    meta_path = os.path.join(args.output_dir, "icl_langact_metadata.json")
    with open(meta_path, "w") as f:
        json.dump(all_metadata, f, indent=2)
    print(f"\n{'='*60}")
    print(f"Metadata saved to: {meta_path}")
    print(f"{'='*60}")

    # Print summary
    print(f"\nSummary:")
    for traj in all_metadata["trajectories"]:
        print(f"  {traj['traj_dir']}: {traj['dataset']}, ep {traj['episode_idx']}, "
              f"{traj['total_steps']} steps, {traj['num_chunks']} chunks, "
              f"{len(traj['saved_images'])} images")

    print(f"\nTo use as ICL in GPT langact eval, pass:")
    print(f"  --icl_trajectories_dir {args.output_dir}")


if __name__ == "__main__":
    main()
