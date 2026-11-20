"""
aggregate_gpt_results.py

Aggregates GPT correction results from rollout_videos directory.
Shows: per-task success rates, GPT reasoning summaries, action diff stats.

Usage:
    python experiments/robot/libero/aggregate_gpt_results.py --video_dir ./rollout_videos/libero_goal
    python experiments/robot/libero/aggregate_gpt_results.py --video_dir ./rollout_videos/libero_goal --save_report
"""

import argparse
import json
import os
from glob import glob
from collections import defaultdict

import numpy as np


def load_all_gpt_logs(video_dir):
    """Find and load all gpt_logs.json files."""
    logs = []
    for log_path in sorted(glob(os.path.join(video_dir, "**", "*--gpt_logs.json"), recursive=True)):
        with open(log_path) as f:
            data = json.load(f)
        data["_path"] = log_path
        logs.append(data)
    return logs


def load_video_results(video_dir):
    """Parse success/failure from video filenames."""
    results = defaultdict(lambda: defaultdict(dict))  # task -> ep -> mode -> success
    for mp4 in sorted(glob(os.path.join(video_dir, "**", "*.mp4"), recursive=True)):
        basename = os.path.basename(mp4)
        if "side_by_side" in basename:
            continue
        parts = basename.replace(".mp4", "").split("--")
        if len(parts) >= 2:
            mode = parts[0]
            success_str = parts[1].replace("success=", "")
            success = success_str == "True"

            # Extract task and episode from path
            rel = os.path.relpath(mp4, video_dir)
            path_parts = rel.split(os.sep)
            if len(path_parts) >= 3:
                task = path_parts[0]
                ep = path_parts[1]
                results[task][ep][mode] = success
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_dir", type=str, required=True)
    parser.add_argument("--save_report", action="store_true", help="Save report to file")
    args = parser.parse_args()

    lines = []
    def out(s=""):
        print(s)
        lines.append(s)

    # Load video results
    video_results = load_video_results(args.video_dir)

    out("=" * 80)
    out("GPT CORRECTION EVALUATION REPORT")
    out("=" * 80)
    out(f"Video dir: {args.video_dir}")
    out()

    # Aggregate success rates by mode
    mode_stats = defaultdict(lambda: {"total": 0, "success": 0})
    task_mode_stats = defaultdict(lambda: defaultdict(lambda: {"total": 0, "success": 0}))

    for task, episodes in sorted(video_results.items()):
        for ep, modes in sorted(episodes.items()):
            for mode, success in modes.items():
                mode_stats[mode]["total"] += 1
                if success:
                    mode_stats[mode]["success"] += 1
                task_mode_stats[task][mode]["total"] += 1
                if success:
                    task_mode_stats[task][mode]["success"] += 1

    # Overall success rates
    out("OVERALL SUCCESS RATES")
    out("-" * 60)
    for mode in sorted(mode_stats.keys()):
        s = mode_stats[mode]
        rate = s["success"] / s["total"] if s["total"] > 0 else 0
        out(f"  {mode:25s}: {s['success']}/{s['total']} ({rate*100:.1f}%)")
    out()

    # Per-task success rates
    out("PER-TASK SUCCESS RATES")
    out("-" * 60)
    all_modes = sorted(mode_stats.keys())
    header = f"  {'Task':<45s}" + "".join(f" {m:>18s}" for m in all_modes)
    out(header)
    out("  " + "-" * (45 + 19 * len(all_modes)))
    for task in sorted(task_mode_stats.keys()):
        row = f"  {task:<45s}"
        for mode in all_modes:
            s = task_mode_stats[task][mode]
            if s["total"] > 0:
                rate = s["success"] / s["total"]
                row += f" {s['success']}/{s['total']} ({rate*100:4.0f}%)    "
            else:
                row += f" {'N/A':>18s}"
        out(row)
    out()

    # Load and summarize GPT reasoning
    gpt_logs = load_all_gpt_logs(args.video_dir)

    if gpt_logs:
        out("=" * 80)
        out("GPT REASONING SUMMARIES")
        out("=" * 80)
        out()

        for log in gpt_logs:
            task = log["task_description"]
            mode = log["eval_mode"]
            ep = log["episode_idx"]
            success = log["success"]
            n_chunks = len(log["chunks"])

            out(f"--- Task: {task} | Ep {ep} | Mode: {mode} | Success: {success} | Chunks: {n_chunks} ---")

            # Show reasoning for each chunk
            for chunk in log["chunks"]:
                reasoning = chunk.get("reasoning", "")
                if reasoning and reasoning != "PARSE_ERROR":
                    out(f"  [Chunk {chunk['chunk']}, step {chunk['step']}]")
                    # Indent reasoning
                    for line in reasoning.split("\n"):
                        out(f"    {line}")

                    # Show action diff magnitude
                    orig = np.array(chunk["openvla_actions"])
                    corr = np.array(chunk["corrected_actions"])
                    diff = np.abs(corr - orig)
                    l2 = np.sqrt((diff ** 2).sum(axis=-1)).mean()
                    out(f"    -> Mean L2 correction: {l2:.5f}")
                    out()

            out()

        # Aggregate correction statistics
        out("=" * 80)
        out("CORRECTION STATISTICS")
        out("=" * 80)
        dim_names = ["dx", "dy", "dz", "ax", "ay", "az", "grip"]

        for mode in ["gpt_normal", "gpt_residual"]:
            mode_logs = [l for l in gpt_logs if l["eval_mode"] == mode]
            if not mode_logs:
                continue

            all_diffs = []
            n_unchanged = 0
            n_total = 0
            for log in mode_logs:
                for chunk in log["chunks"]:
                    orig = np.array(chunk["openvla_actions"])
                    corr = np.array(chunk["corrected_actions"])
                    diff = np.abs(corr - orig)
                    all_diffs.append(diff)
                    n_total += 1
                    if diff.max() < 1e-4:
                        n_unchanged += 1

            if not all_diffs:
                continue

            all_diffs = np.array(all_diffs)  # (N, 8, 7)
            mean_diff = all_diffs.mean(axis=(0, 1))
            max_diff = all_diffs.max(axis=(0, 1))

            out(f"\n  Mode: {mode}")
            out(f"  Total chunks: {n_total}")
            out(f"  Unchanged (GPT kept original): {n_unchanged}/{n_total} ({n_unchanged/n_total*100:.0f}%)")
            out(f"\n  Per-dimension mean |correction|:")
            out(f"    {'Dim':<8s} {'Mean':>10s} {'Max':>10s}")
            for i, name in enumerate(dim_names):
                out(f"    {name:<8s} {mean_diff[i]:>10.5f} {max_diff[i]:>10.5f}")

            overall_l2 = np.sqrt((all_diffs ** 2).sum(axis=-1)).mean()
            out(f"\n  Overall mean L2 per step: {overall_l2:.5f}")

    # Save report
    if args.save_report:
        report_path = os.path.join(args.video_dir, "gpt_eval_report.txt")
        with open(report_path, "w") as f:
            f.write("\n".join(lines))
        print(f"\nReport saved to: {report_path}")


if __name__ == "__main__":
    main()
