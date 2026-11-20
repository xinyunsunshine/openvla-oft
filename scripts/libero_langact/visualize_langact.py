"""
Visualize language-action labels alongside before/after observation images.

For each sampled timestep t, shows:
  - Left:  observation image at step t (before action chunk)
  - Right: observation image at step t+W (after action chunk, W=window_size)
  - Title: the language-action text describing the motion between the two frames

Usage:
    # Visualize the generated language-action dataset
    python scripts/libero_langact/visualize_langact.py \
        --data_dir /proj/inf-scaling/SWE_sim/modified_libero_rlds_new \
        --dataset libero_spatial_no_noops_langact_verbose \
        --num_episodes 3 \
        --steps_per_episode 5 \
        --output viz_langact.png

    # Compare original vs language-action side by side
    python scripts/libero_langact/visualize_langact.py \
        --data_dir /proj/inf-scaling/SWE_sim/modified_libero_rlds_new \
        --dataset libero_spatial_no_noops_langact_verbose \
        --original_dataset libero_spatial_no_noops \
        --num_episodes 2 \
        --steps_per_episode 4 \
        --output viz_compare.png
"""

import argparse
import os
import sys
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

tf.config.set_visible_devices([], "GPU")
import tensorflow_datasets as tfds

# Add script dir to path for langact_utils
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from langact_utils import (
    axis_angle_to_euler,
    euler_diff,
    summarize_numeric_actions,
)


def compute_actual_delta(steps, t_start, t_end):
    """Compute the actual EEF delta between two timesteps and return as verbose text.

    Returns (delta_text, delta_7d) where delta_text is the verbose language-action
    description of the actual motion and delta_7d is the raw 7D vector.
    """
    state_before = steps[t_start]["observation"]["state"].numpy()  # (8,)
    state_after = steps[t_end]["observation"]["state"].numpy()

    # Position delta
    delta_pos = state_after[:3] - state_before[:3]

    # Rotation delta via euler
    euler_before = axis_angle_to_euler(state_before[3:6].reshape(1, 3))[0]
    euler_after = axis_angle_to_euler(state_after[3:6].reshape(1, 3))[0]
    delta_rot = euler_diff(euler_after.reshape(1, 3), euler_before.reshape(1, 3))[0]

    # Gripper: use after state, invert (1=open, 0=close)
    gripper = 1.0 - np.clip(state_after[6], 0, 1)

    delta_7d = np.concatenate([delta_pos, delta_rot, [gripper]])
    delta_text = summarize_numeric_actions(delta_7d, sum_decimal="1f", include_rotation=True)
    return delta_text or "close gripper", delta_7d


def load_episodes(data_dir, dataset_name, num_episodes):
    """Load episodes from an RLDS dataset."""
    builder = tfds.builder(dataset_name, data_dir=data_dir)
    ds = builder.as_dataset(split="train")
    episodes = []
    for ep_idx, episode in enumerate(ds):
        if ep_idx >= num_episodes:
            break
        steps = list(episode["steps"])
        episodes.append(steps)
    return episodes


def pick_timesteps(num_steps, num_samples, window_size):
    """Pick evenly spaced timesteps that have a valid t+window frame."""
    max_t = num_steps - window_size
    if max_t <= 0:
        return [0]
    indices = np.linspace(0, max_t, num_samples, dtype=int)
    return list(indices)


def visualize_dataset(
    data_dir,
    dataset_name,
    original_dataset=None,
    num_episodes=3,
    steps_per_episode=5,
    window_size=8,
    output_path="viz_langact.png",
    image_key="image",
):
    """Create visualization grid."""
    print(f"Loading {dataset_name}...")
    episodes = load_episodes(data_dir, dataset_name, num_episodes)

    orig_episodes = None
    if original_dataset:
        print(f"Loading {original_dataset} for comparison...")
        orig_episodes = load_episodes(data_dir, original_dataset, num_episodes)

    # Determine grid size
    total_rows = 0
    row_data = []  # list of (before_img, after_img, lang_text, orig_text)
    for ep_idx, steps in enumerate(episodes):
        timesteps = pick_timesteps(len(steps), steps_per_episode, window_size)
        for t in timesteps:
            t_end = min(t + window_size, len(steps) - 1)
            before_img = steps[t]["observation"][image_key].numpy()
            after_img = steps[t_end]["observation"][image_key].numpy()
            lang = steps[t]["language_instruction"].numpy().decode("utf-8")

            orig_text = None
            if orig_episodes and ep_idx < len(orig_episodes):
                orig_steps = orig_episodes[ep_idx]
                if t < len(orig_steps):
                    orig_text = orig_steps[t]["language_instruction"].numpy().decode("utf-8")

            # Compute actual EEF delta from state observations
            actual_delta_text, actual_delta_7d = compute_actual_delta(steps, t, t_end)

            row_data.append({
                "ep": ep_idx,
                "t": t,
                "t_end": t_end,
                "before": before_img,
                "after": after_img,
                "lang": lang,
                "orig": orig_text,
                "actual_delta": actual_delta_text,
                "actual_delta_7d": actual_delta_7d,
            })

    nrows = len(row_data)
    ncols = 3 if original_dataset else 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5 * nrows))
    if nrows == 1:
        axes = axes[None, :]

    for i, row in enumerate(row_data):
        # Before image
        axes[i, 0].imshow(row["before"])
        axes[i, 0].set_title(f"Ep {row['ep']}, Step {row['t']} (before)", fontsize=10)
        axes[i, 0].axis("off")

        # After image
        axes[i, 1].imshow(row["after"])
        axes[i, 1].set_title(f"Step {row['t_end']} (after {row['t_end']-row['t']} steps)", fontsize=10)
        axes[i, 1].axis("off")

        # Language-action text, actual delta, and optionally original instruction
        if ncols == 3:
            # Comparison mode: show original, language-action prompt, and actual delta
            axes[i, 2].axis("off")
            wrapped_orig = "\n".join(textwrap.wrap(row["orig"] or "N/A", width=45))
            wrapped_lang = "\n".join(textwrap.wrap(row["lang"], width=45))
            wrapped_actual = "\n".join(textwrap.wrap(row["actual_delta"], width=45))

            axes[i, 2].text(
                0.05, 0.95,
                f"Original task:\n{wrapped_orig}",
                transform=axes[i, 2].transAxes,
                fontsize=8, verticalalignment="top", fontfamily="monospace",
                bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8),
            )
            axes[i, 2].text(
                0.05, 0.62,
                f"Prompt (noisy):\n{wrapped_lang}",
                transform=axes[i, 2].transAxes,
                fontsize=8, verticalalignment="top", fontfamily="monospace",
                bbox=dict(boxstyle="round", facecolor="lightcyan", alpha=0.8),
            )
            axes[i, 2].text(
                0.05, 0.28,
                f"Actual EEF delta:\n{wrapped_actual}",
                transform=axes[i, 2].transAxes,
                fontsize=8, verticalalignment="top", fontfamily="monospace",
                bbox=dict(boxstyle="round", facecolor="honeydew", alpha=0.8),
            )
        else:
            # No original dataset: show language-action prompt and actual delta
            wrapped_lang = "\n".join(textwrap.wrap(row["lang"], width=40))
            wrapped_actual = "\n".join(textwrap.wrap(row["actual_delta"], width=40))
            axes[i, 1].text(
                0.5, -0.02, f"Prompt: {wrapped_lang}",
                transform=axes[i, 1].transAxes,
                fontsize=8, ha="center", va="top", fontfamily="monospace",
                bbox=dict(boxstyle="round", facecolor="lightcyan", alpha=0.8),
            )
            axes[i, 1].text(
                0.5, -0.15, f"Actual: {wrapped_actual}",
                transform=axes[i, 1].transAxes,
                fontsize=8, ha="center", va="top", fontfamily="monospace",
                bbox=dict(boxstyle="round", facecolor="honeydew", alpha=0.8),
            )

    plt.suptitle(
        f"Language-Action Verification: {dataset_name}\n"
        f"(window={window_size} steps, showing before → after frames)",
        fontsize=13, fontweight="bold", y=1.01,
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved visualization to: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Visualize language-action labels with before/after images."
    )
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True,
                        help="Language-action dataset name (e.g., libero_spatial_no_noops_langact_verbose)")
    parser.add_argument("--original_dataset", type=str, default=None,
                        help="Original dataset name for comparison (e.g., libero_spatial_no_noops)")
    parser.add_argument("--num_episodes", type=int, default=3)
    parser.add_argument("--steps_per_episode", type=int, default=5)
    parser.add_argument("--window_size", type=int, default=8)
    parser.add_argument("--output", type=str, default="viz_langact.png")
    parser.add_argument("--image_key", type=str, default="image",
                        choices=["image", "wrist_image"])
    args = parser.parse_args()

    visualize_dataset(
        data_dir=args.data_dir,
        dataset_name=args.dataset,
        original_dataset=args.original_dataset,
        num_episodes=args.num_episodes,
        steps_per_episode=args.steps_per_episode,
        window_size=args.window_size,
        output_path=args.output,
        image_key=args.image_key,
    )


if __name__ == "__main__":
    main()
