"""
analyze_gpt_corrections.py

Loads GPT correction results and compares normal vs residual modes.
Prints per-chunk diffs, summary statistics, and optionally saves
side-by-side visualizations.

Usage:
    python experiments/robot/libero/analyze_gpt_corrections.py \
        --results_dir ./gpt_corrections/libero_spatial \
        --save_viz
"""

import argparse
import json
import os
from glob import glob

import numpy as np


def load_results(results_dir, mode):
    """Load all episode result JSONs for a given mode."""
    mode_dir = os.path.join(results_dir, mode)
    if not os.path.isdir(mode_dir):
        print(f"  No results found for mode '{mode}' in {results_dir}")
        return []
    files = sorted(glob(os.path.join(mode_dir, "*.json")))
    results = []
    for f in files:
        with open(f) as fh:
            results.append(json.load(fh))
    return results


def compute_action_diff(original, corrected):
    """Compute per-dimension absolute difference between original and corrected actions."""
    orig = np.array(original)
    corr = np.array(corrected)
    return np.abs(corr - orig)


def analyze_mode(results, mode):
    """Print analysis for a single mode."""
    print(f"\n{'='*60}")
    print(f"Mode: {mode}")
    print(f"{'='*60}")

    total_chunks = 0
    total_errors = 0
    all_diffs = []  # (8, 7) per chunk
    dim_names = ["dx", "dy", "dz", "ax", "ay", "az", "grip"]

    for ep in results:
        for chunk in ep["chunk_results"]:
            total_chunks += 1
            if chunk["error"] is not None:
                total_errors += 1
                continue

            orig = chunk["openvla_actions"]
            corr = chunk["corrected_actions"]
            diff = compute_action_diff(orig, corr)
            all_diffs.append(diff)

    print(f"Total chunks: {total_chunks}")
    print(f"Errors: {total_errors}")
    print(f"Valid: {total_chunks - total_errors}")

    if not all_diffs:
        print("No valid results to analyze.")
        return

    all_diffs = np.array(all_diffs)  # (N, 8, 7)

    # Per-dimension statistics (averaged over chunks and steps)
    mean_diff = all_diffs.mean(axis=(0, 1))  # (7,)
    std_diff = all_diffs.std(axis=(0, 1))
    max_diff = all_diffs.max(axis=(0, 1))

    print(f"\nPer-dimension |correction| (averaged over all chunks and steps):")
    print(f"  {'Dim':<6} {'Mean':>10} {'Std':>10} {'Max':>10}")
    print(f"  {'-'*6} {'-'*10} {'-'*10} {'-'*10}")
    for i, name in enumerate(dim_names):
        print(f"  {name:<6} {mean_diff[i]:>10.5f} {std_diff[i]:>10.5f} {max_diff[i]:>10.5f}")

    # Per-step statistics (averaged over chunks)
    step_mean = all_diffs.mean(axis=0)  # (8, 7)
    print(f"\nPer-step mean |correction| (averaged over chunks):")
    header = f"  {'Step':<6}" + "".join(f" {name:>8}" for name in dim_names)
    print(header)
    print(f"  {'-'*6}" + " " + "-" * (8 * len(dim_names) + len(dim_names) - 1))
    for step_i in range(8):
        row = f"  {step_i:<6}" + "".join(f" {step_mean[step_i, d]:>8.5f}" for d in range(7))
        print(row)

    # Overall magnitude
    chunk_l2 = np.sqrt((all_diffs ** 2).sum(axis=-1))  # (N, 8)
    print(f"\nOverall L2 correction magnitude per step:")
    print(f"  Mean: {chunk_l2.mean():.5f}")
    print(f"  Std:  {chunk_l2.std():.5f}")
    print(f"  Max:  {chunk_l2.max():.5f}")

    return all_diffs


def print_example_chunks(results, mode, n=3):
    """Print a few example chunks showing original vs corrected."""
    print(f"\n--- Example chunks ({mode}) ---")
    count = 0
    for ep in results:
        for chunk in ep["chunk_results"]:
            if chunk["error"] is not None:
                continue
            if count >= n:
                return

            print(f"\nTask: {ep['task_description']}")
            print(f"Chunk {chunk['chunk_idx']} (step {chunk['step']}):")
            print(f"  {'Step':<5} {'Original':>50}  {'Corrected':>50}")
            for i in range(8):
                orig_str = "[" + ", ".join(f"{v:>8.4f}" for v in chunk["openvla_actions"][i]) + "]"
                corr_str = "[" + ", ".join(f"{v:>8.4f}" for v in chunk["corrected_actions"][i]) + "]"
                print(f"  {i:<5} {orig_str}  {corr_str}")
            count += 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, required=True)
    parser.add_argument("--save_viz", action="store_true",
                        help="Save a bar chart comparing correction magnitudes")
    args = parser.parse_args()

    for mode in ["normal", "residual"]:
        results = load_results(args.results_dir, mode)
        if not results:
            continue
        analyze_mode(results, mode)
        print_example_chunks(results, mode, n=2)

    # Compare modes if both exist
    normal_results = load_results(args.results_dir, "normal")
    residual_results = load_results(args.results_dir, "residual")

    if normal_results and residual_results:
        print(f"\n{'='*60}")
        print("COMPARISON: Normal vs Residual")
        print(f"{'='*60}")

        # Collect corrected actions per (episode, chunk) for both modes
        normal_map = {}
        for ep in normal_results:
            for chunk in ep["chunk_results"]:
                if chunk["error"] is None:
                    key = (ep["episode_file"], chunk["chunk_idx"])
                    normal_map[key] = np.array(chunk["corrected_actions"])

        residual_map = {}
        for ep in residual_results:
            for chunk in ep["chunk_results"]:
                if chunk["error"] is None:
                    key = (ep["episode_file"], chunk["chunk_idx"])
                    residual_map[key] = np.array(chunk["corrected_actions"])

        # Find common chunks
        common_keys = set(normal_map.keys()) & set(residual_map.keys())
        print(f"Common chunks: {len(common_keys)}")

        if common_keys:
            diffs = []
            for key in common_keys:
                diff = np.abs(normal_map[key] - residual_map[key])
                diffs.append(diff)
            diffs = np.array(diffs)

            dim_names = ["dx", "dy", "dz", "ax", "ay", "az", "grip"]
            mean_diff = diffs.mean(axis=(0, 1))
            print(f"\nMean |normal - residual| per dimension:")
            for i, name in enumerate(dim_names):
                print(f"  {name}: {mean_diff[i]:.5f}")

            overall_l2 = np.sqrt((diffs ** 2).sum(axis=-1)).mean()
            print(f"\nMean L2 distance between normal and residual corrections: {overall_l2:.5f}")

    if args.save_viz:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig_path = os.path.join(args.results_dir, "correction_comparison.png")

            # Gather per-dim mean corrections for each mode
            dim_names = ["dx", "dy", "dz", "ax", "ay", "az", "grip"]
            data = {}
            for mode in ["normal", "residual"]:
                results = load_results(args.results_dir, mode)
                if not results:
                    continue
                all_diffs = []
                for ep in results:
                    for chunk in ep["chunk_results"]:
                        if chunk["error"] is None:
                            orig = np.array(chunk["openvla_actions"])
                            corr = np.array(chunk["corrected_actions"])
                            all_diffs.append(np.abs(corr - orig))
                if all_diffs:
                    data[mode] = np.array(all_diffs).mean(axis=(0, 1))

            if data:
                x = np.arange(len(dim_names))
                width = 0.35
                fig, ax = plt.subplots(figsize=(10, 5))
                for i, (mode, vals) in enumerate(data.items()):
                    ax.bar(x + i * width, vals, width, label=mode)
                ax.set_xticks(x + width / 2)
                ax.set_xticklabels(dim_names)
                ax.set_ylabel("Mean |correction|")
                ax.set_title("GPT Correction Magnitude by Dimension")
                ax.legend()
                plt.tight_layout()
                plt.savefig(fig_path, dpi=150)
                print(f"\nSaved visualization: {fig_path}")
        except ImportError:
            print("\nmatplotlib not available, skipping visualization.")


if __name__ == "__main__":
    main()
