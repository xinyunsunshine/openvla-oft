"""
visualize_gpt_corrections.py

Rich visualizations comparing OpenVLA actions vs GPT-corrected actions.

Generates per-episode multi-panel figures:
  Panel A: Observation images (prev + curr) with annotated task description
  Panel B: 3D end-effector trajectory (accumulated position deltas) — original vs corrected
  Panel C: Per-dimension action comparison over the 8-step chunk
  Panel D: Gripper command timeline
  Panel E: Correction magnitude heatmap across all chunks in the episode

Also generates an episode-level overview plotting the full trajectory of EE
position deltas across all chunks, showing where GPT made the largest corrections.

Usage:
    python experiments/robot/libero/visualize_gpt_corrections.py \
        --collect_dir ./collected_data/libero_spatial \
        --results_dir ./gpt_corrections/libero_spatial \
        --output_dir ./visualizations/libero_spatial \
        --mode both \
        --max_episodes 5

    # Visualize a single episode in detail
    python experiments/robot/libero/visualize_gpt_corrections.py \
        --collect_dir ./collected_data/libero_spatial \
        --results_dir ./gpt_corrections/libero_spatial \
        --output_dir ./visualizations/libero_spatial \
        --episode_file task00_ep000 \
        --mode both
"""

import argparse
import json
import os
import pickle
from glob import glob

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.patches as mpatches
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (needed for 3d projection)


DIM_NAMES = ["dx", "dy", "dz", "ax", "ay", "az", "gripper"]
POS_DIMS = [0, 1, 2]
ORI_DIMS = [3, 4, 5]
GRIP_DIM = 6


# ---------------------------------------------------------------------------
# Per-chunk visualization
# ---------------------------------------------------------------------------

def plot_chunk(
    chunk_collect,
    chunk_result,
    mode,
    task_description,
    episode_name,
    output_dir,
):
    """Generate a multi-panel figure for a single action chunk."""
    chunk_idx = chunk_collect["chunk_idx"]
    step = chunk_collect["step"]

    orig_actions = np.array(chunk_result["openvla_actions"])       # (8, 7)
    corr_actions = np.array(chunk_result["corrected_actions"])     # (8, 7)
    diff = corr_actions - orig_actions                             # (8, 7)

    curr_img = chunk_collect["curr_obs_img"]
    prev_img = chunk_collect.get("prev_obs_img")

    fig = plt.figure(figsize=(22, 12))
    fig.suptitle(
        f"Task: {task_description}\n"
        f"Episode: {episode_name} | Chunk {chunk_idx} (step {step}) | Mode: {mode}",
        fontsize=13, fontweight="bold", y=0.98,
    )

    gs = GridSpec(3, 4, figure=fig, hspace=0.35, wspace=0.35)

    # --- Panel A: Observation images ---
    ax_prev = fig.add_subplot(gs[0, 0])
    if prev_img is not None:
        ax_prev.imshow(prev_img)
        ax_prev.set_title("Previous obs (~8 steps ago)", fontsize=10)
    else:
        ax_prev.text(0.5, 0.5, "N/A\n(first chunk)", ha="center", va="center", fontsize=12)
        ax_prev.set_title("Previous obs", fontsize=10)
    ax_prev.axis("off")

    ax_curr = fig.add_subplot(gs[0, 1])
    ax_curr.imshow(curr_img)
    ax_curr.set_title("Current obs", fontsize=10)
    ax_curr.axis("off")

    # --- Wrist image if available ---
    wrist_img = chunk_collect.get("curr_wrist_img")
    ax_wrist = fig.add_subplot(gs[0, 2])
    if wrist_img is not None:
        ax_wrist.imshow(wrist_img)
        ax_wrist.set_title("Wrist camera", fontsize=10)
    else:
        ax_wrist.text(0.5, 0.5, "N/A", ha="center", va="center", fontsize=12)
        ax_wrist.set_title("Wrist camera", fontsize=10)
    ax_wrist.axis("off")

    # --- Panel B: 3D position trajectory ---
    ax3d = fig.add_subplot(gs[0, 3], projection="3d")
    orig_pos = np.cumsum(orig_actions[:, POS_DIMS], axis=0)
    corr_pos = np.cumsum(corr_actions[:, POS_DIMS], axis=0)
    # Prepend origin
    orig_pos = np.vstack([[0, 0, 0], orig_pos])
    corr_pos = np.vstack([[0, 0, 0], corr_pos])

    ax3d.plot(orig_pos[:, 0], orig_pos[:, 1], orig_pos[:, 2], "b.-", label="OpenVLA", linewidth=2, markersize=6)
    ax3d.plot(corr_pos[:, 0], corr_pos[:, 1], corr_pos[:, 2], "r.-", label=f"GPT ({mode})", linewidth=2, markersize=6)
    ax3d.scatter([0], [0], [0], c="green", s=80, zorder=5, label="Start")
    ax3d.set_xlabel("x (accum)")
    ax3d.set_ylabel("y (accum)")
    ax3d.set_zlabel("z (accum)")
    ax3d.set_title("EE position trajectory", fontsize=10)
    ax3d.legend(fontsize=8)

    # --- Panel C: Per-dimension action comparison (position dims) ---
    steps_x = np.arange(8)
    for plot_i, (dims, dim_labels, title) in enumerate([
        (POS_DIMS, ["dx", "dy", "dz"], "Position deltas"),
        (ORI_DIMS, ["ax", "ay", "az"], "Orientation deltas"),
    ]):
        ax = fig.add_subplot(gs[1, plot_i * 2: plot_i * 2 + 2])
        for j, (dim, label) in enumerate(zip(dims, dim_labels)):
            color = plt.cm.Set1(j)
            ax.plot(steps_x, orig_actions[:, dim], "o--", color=color, alpha=0.5, label=f"{label} (orig)")
            ax.plot(steps_x, corr_actions[:, dim], "s-", color=color, label=f"{label} (GPT)")
        ax.set_xlabel("Step in chunk")
        ax.set_ylabel("Action value")
        ax.set_title(title, fontsize=10)
        ax.legend(fontsize=7, ncol=2)
        ax.grid(True, alpha=0.3)
        ax.set_xticks(steps_x)

    # --- Panel D: Gripper timeline ---
    ax_grip = fig.add_subplot(gs[2, 0:2])
    ax_grip.step(steps_x, orig_actions[:, GRIP_DIM], "b-", where="mid", linewidth=2, label="OpenVLA")
    ax_grip.step(steps_x, corr_actions[:, GRIP_DIM], "r-", where="mid", linewidth=2, label=f"GPT ({mode})")
    ax_grip.axhline(0.5, color="gray", linestyle=":", alpha=0.5, label="Open/close threshold")
    ax_grip.set_xlabel("Step in chunk")
    ax_grip.set_ylabel("Gripper value")
    ax_grip.set_title("Gripper command", fontsize=10)
    ax_grip.legend(fontsize=8)
    ax_grip.grid(True, alpha=0.3)
    ax_grip.set_xticks(steps_x)
    ax_grip.set_ylim(-0.2, 1.2)

    # --- Panel E: Correction magnitude heatmap ---
    ax_heat = fig.add_subplot(gs[2, 2:4])
    abs_diff = np.abs(diff)
    im = ax_heat.imshow(abs_diff.T, aspect="auto", cmap="YlOrRd", interpolation="nearest")
    ax_heat.set_xlabel("Step in chunk")
    ax_heat.set_ylabel("Action dimension")
    ax_heat.set_yticks(range(7))
    ax_heat.set_yticklabels(DIM_NAMES)
    ax_heat.set_xticks(range(8))
    ax_heat.set_title("|Correction| heatmap", fontsize=10)
    plt.colorbar(im, ax=ax_heat, fraction=0.046, pad=0.04)

    # Save
    out_path = os.path.join(output_dir, f"{episode_name}_chunk{chunk_idx:03d}_{mode}.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# Episode-level overview
# ---------------------------------------------------------------------------

def plot_episode_overview(
    episode_collect,
    episode_result,
    mode,
    output_dir,
):
    """
    Full-episode trajectory visualization:
    - Top: EE position (dx, dy, dz) over ALL steps, original vs corrected
    - Middle: Gripper over all steps
    - Bottom: Per-chunk correction magnitude (L2 norm)
    """
    task_desc = episode_collect["task_description"]
    ep_name = os.path.splitext(os.path.basename(episode_result.get("episode_file", "unknown")))[0]
    success = episode_collect["success"]
    chunks_collect = episode_collect["chunks"]
    chunks_result = episode_result["chunk_results"]

    # Build a mapping from chunk_idx -> result
    result_map = {}
    for cr in chunks_result:
        if cr["error"] is None:
            result_map[cr["chunk_idx"]] = cr

    # Reconstruct full trajectories
    all_orig = []
    all_corr = []
    chunk_starts = []  # step indices where new chunks start
    chunk_l2s = []     # L2 correction magnitude per chunk

    for cc in chunks_collect:
        cidx = cc["chunk_idx"]
        orig_actions = np.array(cc["openvla_actions"])  # (8, 7)
        all_orig.append(orig_actions)

        if cidx in result_map:
            corr_actions = np.array(result_map[cidx]["corrected_actions"])
            l2 = np.sqrt(((corr_actions - orig_actions) ** 2).sum(axis=-1)).mean()
        else:
            corr_actions = orig_actions.copy()
            l2 = 0.0

        all_corr.append(corr_actions)
        chunk_starts.append(cc["step"])
        chunk_l2s.append(l2)

    all_orig = np.concatenate(all_orig, axis=0)  # (T, 7)
    all_corr = np.concatenate(all_corr, axis=0)

    T_total = len(all_orig)
    steps = np.arange(T_total)

    fig, axes = plt.subplots(4, 1, figsize=(18, 14), sharex=True)
    fig.suptitle(
        f"Episode Overview: {task_desc}\n"
        f"{ep_name} | Mode: {mode} | OpenVLA success: {success}",
        fontsize=13, fontweight="bold",
    )

    # Position dims
    for dim, label in zip(POS_DIMS, ["dx", "dy", "dz"]):
        axes[0].plot(steps, all_orig[:, dim], alpha=0.4, label=f"{label} (orig)")
        axes[0].plot(steps, all_corr[:, dim], linewidth=1.5, label=f"{label} (GPT)")
    axes[0].set_ylabel("Position delta")
    axes[0].set_title("Position action dimensions over episode")
    axes[0].legend(fontsize=7, ncol=6)
    axes[0].grid(True, alpha=0.3)

    # Orientation dims
    for dim, label in zip(ORI_DIMS, ["ax", "ay", "az"]):
        axes[1].plot(steps, all_orig[:, dim], alpha=0.4, label=f"{label} (orig)")
        axes[1].plot(steps, all_corr[:, dim], linewidth=1.5, label=f"{label} (GPT)")
    axes[1].set_ylabel("Orientation delta")
    axes[1].set_title("Orientation action dimensions over episode")
    axes[1].legend(fontsize=7, ncol=6)
    axes[1].grid(True, alpha=0.3)

    # Gripper
    axes[2].step(steps, all_orig[:, GRIP_DIM], alpha=0.5, where="mid", label="Gripper (orig)")
    axes[2].step(steps, all_corr[:, GRIP_DIM], where="mid", linewidth=1.5, label="Gripper (GPT)")
    axes[2].set_ylabel("Gripper")
    axes[2].set_title("Gripper command over episode")
    axes[2].legend(fontsize=8)
    axes[2].grid(True, alpha=0.3)

    # Chunk boundaries
    for ax in axes[:3]:
        for cs in chunk_starts:
            ax.axvline(cs - chunk_starts[0], color="gray", linestyle=":", alpha=0.3)

    # Per-chunk correction magnitude
    chunk_x = np.arange(len(chunk_l2s))
    colors = plt.cm.YlOrRd(np.array(chunk_l2s) / (max(chunk_l2s) + 1e-8))
    axes[3].bar(chunk_x, chunk_l2s, color=colors, edgecolor="black", linewidth=0.5)
    axes[3].set_xlabel("Chunk index")
    axes[3].set_ylabel("Mean L2 correction")
    axes[3].set_title("Correction magnitude per chunk (larger = GPT changed more)")
    axes[3].grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join(output_dir, f"{ep_name}_overview_{mode}.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# Accumulated EE position trajectory (2D projections)
# ---------------------------------------------------------------------------

def plot_accumulated_trajectory(
    episode_collect,
    episode_result,
    mode,
    output_dir,
):
    """Plot accumulated EE position from action deltas — 3 projections (XY, XZ, YZ) + 3D."""
    task_desc = episode_collect["task_description"]
    ep_name = os.path.splitext(os.path.basename(episode_result.get("episode_file", "unknown")))[0]

    chunks_collect = episode_collect["chunks"]
    chunks_result = episode_result["chunk_results"]
    result_map = {cr["chunk_idx"]: cr for cr in chunks_result if cr["error"] is None}

    all_orig_pos = []
    all_corr_pos = []
    for cc in chunks_collect:
        cidx = cc["chunk_idx"]
        orig = np.array(cc["openvla_actions"])[:, POS_DIMS]
        all_orig_pos.append(orig)
        if cidx in result_map:
            corr = np.array(result_map[cidx]["corrected_actions"])[:, POS_DIMS]
        else:
            corr = orig.copy()
        all_corr_pos.append(corr)

    orig_traj = np.cumsum(np.concatenate(all_orig_pos), axis=0)
    corr_traj = np.cumsum(np.concatenate(all_corr_pos), axis=0)
    orig_traj = np.vstack([[0, 0, 0], orig_traj])
    corr_traj = np.vstack([[0, 0, 0], corr_traj])

    fig = plt.figure(figsize=(18, 5))
    fig.suptitle(f"Accumulated EE trajectory: {task_desc}\n{ep_name} | {mode}", fontsize=12)

    projections = [("X", "Y", 0, 1), ("X", "Z", 0, 2), ("Y", "Z", 1, 2)]
    for i, (xlabel, ylabel, xi, yi) in enumerate(projections):
        ax = fig.add_subplot(1, 4, i + 1)
        ax.plot(orig_traj[:, xi], orig_traj[:, yi], "b.-", alpha=0.6, label="OpenVLA", markersize=2)
        ax.plot(corr_traj[:, xi], corr_traj[:, yi], "r.-", alpha=0.8, label=f"GPT ({mode})", markersize=2)
        ax.scatter([0], [0], c="green", s=60, zorder=5)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(f"{xlabel}-{ylabel} projection")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_aspect("equal", adjustable="datalim")

    ax3d = fig.add_subplot(1, 4, 4, projection="3d")
    ax3d.plot(orig_traj[:, 0], orig_traj[:, 1], orig_traj[:, 2], "b.-", alpha=0.6, label="OpenVLA", markersize=2)
    ax3d.plot(corr_traj[:, 0], corr_traj[:, 1], corr_traj[:, 2], "r.-", alpha=0.8, label=f"GPT ({mode})", markersize=2)
    ax3d.scatter([0], [0], [0], c="green", s=60, zorder=5)
    ax3d.set_xlabel("X")
    ax3d.set_ylabel("Y")
    ax3d.set_zlabel("Z")
    ax3d.set_title("3D")
    ax3d.legend(fontsize=8)

    plt.tight_layout()
    out_path = os.path.join(output_dir, f"{ep_name}_trajectory_{mode}.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--collect_dir", type=str, required=True,
                        help="Directory with collected .pkl episode files")
    parser.add_argument("--results_dir", type=str, required=True,
                        help="Directory with GPT correction results (from query_gpt_corrections.py)")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save visualizations")
    parser.add_argument("--mode", type=str, default="both", choices=["normal", "residual", "both"])
    parser.add_argument("--max_episodes", type=int, default=None)
    parser.add_argument("--episode_file", type=str, default=None,
                        help="Visualize a single episode (e.g. 'task00_ep000')")
    parser.add_argument("--max_chunks_per_episode", type=int, default=10,
                        help="Max per-chunk detail plots per episode (overview always uses all)")
    args = parser.parse_args()

    modes = ["normal", "residual"] if args.mode == "both" else [args.mode]
    os.makedirs(args.output_dir, exist_ok=True)

    # Find episodes
    if args.episode_file:
        pkl_files = [os.path.join(args.collect_dir, f"{args.episode_file}.pkl")]
    else:
        pkl_files = sorted(glob(os.path.join(args.collect_dir, "*.pkl")))
    if args.max_episodes:
        pkl_files = pkl_files[:args.max_episodes]

    print(f"Episodes: {len(pkl_files)}")
    print(f"Modes: {modes}")
    print(f"Output: {args.output_dir}")

    for pkl_path in pkl_files:
        ep_name = os.path.splitext(os.path.basename(pkl_path))[0]
        print(f"\n--- {ep_name} ---")

        with open(pkl_path, "rb") as f:
            ep_collect = pickle.load(f)

        for mode in modes:
            result_path = os.path.join(args.results_dir, mode, f"{ep_name}.json")
            if not os.path.exists(result_path):
                print(f"  No {mode} results for {ep_name}, skipping")
                continue

            with open(result_path) as f:
                ep_result = json.load(f)

            mode_dir = os.path.join(args.output_dir, mode)
            os.makedirs(mode_dir, exist_ok=True)

            # Episode overview
            path = plot_episode_overview(ep_collect, ep_result, mode, mode_dir)
            print(f"  Overview: {path}")

            # Accumulated trajectory
            path = plot_accumulated_trajectory(ep_collect, ep_result, mode, mode_dir)
            print(f"  Trajectory: {path}")

            # Per-chunk detail plots
            result_map = {}
            for cr in ep_result["chunk_results"]:
                if cr["error"] is None:
                    result_map[cr["chunk_idx"]] = cr

            n_plotted = 0
            for cc in ep_collect["chunks"]:
                if n_plotted >= args.max_chunks_per_episode:
                    break
                cidx = cc["chunk_idx"]
                if cidx not in result_map:
                    continue
                path = plot_chunk(cc, result_map[cidx], mode, ep_collect["task_description"], ep_name, mode_dir)
                print(f"  Chunk {cidx}: {path}")
                n_plotted += 1

    print(f"\nDone! Visualizations saved to {args.output_dir}")


if __name__ == "__main__":
    main()
