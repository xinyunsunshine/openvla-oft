"""
Quick check: does the model output actions in the correct direction for each prompt?

Tests at initial state AND mid-rollout states (chunk 3, chunk 6) if precomputed
sim states are available.

Action vector: [dx, dy, dz, droll, dpitch, dyaw, gripper]
  dx:     + = forward,  - = back
  dy:     + = left,     - = right
  dz:     + = up,       - = down
  droll:  + = tilt left, - = tilt right
  dpitch: + = tilt back, - = tilt forward
  dyaw:   + = rotate CCW, - = rotate CW
  gripper: >= 0.5 = open, < 0.5 = close

Usage:
    cd /proj/inf-scaling/zwhong/idan/vla-rl/openvla-oft

    # Initial state only
    python scripts/libero_langact/check_action_directions.py \
        --pretrained_checkpoint <path_to_checkpoint>

    # With precomputed mid-traj states (from generate_mid_traj_states.py)
    python scripts/libero_langact/check_action_directions.py \
        --pretrained_checkpoint <path_to_checkpoint> \
        --precomputed_states_dir ./mid_traj_states/libero_goal_nonum150k
"""

import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "experiments", "robot"))

from experiments.robot.libero.libero_utils import (
    get_libero_dummy_action,
    get_libero_env,
    get_libero_image,
    get_libero_wrist_image,
    quat2axisangle,
)
from experiments.robot.openvla_utils import (
    get_action_head,
    get_noisy_action_projector,
    get_processor,
    get_proprio_projector,
    resize_image_for_policy,
)
from experiments.robot.robot_utils import (
    get_action,
    get_image_resize_size,
    get_model,
    set_seed_everywhere,
)
from libero.libero import benchmark

from test_steerability import SteerConfig, check_unnorm_key, prepare_observation, reset_env_to_sim_state


# Prompts to check — each has expected sign conditions (comma-separated for combos)
# Format: "dim op val" conditions joined by ", "
#   e.g. "dx > 0" or "dz > 0, droll < 0" for combo checks

CHECK_PROMPTS_VERBOSE = {
    # ── Single-axis translation ──
    "move_forward_5cm":  ("move forward 5.0 cm, open gripper",   "dx > 0"),
    "move_back_5cm":     ("move back 5.0 cm, open gripper",      "dx < 0"),
    "move_left_5cm":     ("move left 5.0 cm, open gripper",      "dy > 0"),
    "move_right_5cm":    ("move right 5.0 cm, open gripper",     "dy < 0"),
    "move_up_5cm":       ("move up 5.0 cm, open gripper",        "dz > 0"),
    "move_down_5cm":     ("move down 5.0 cm, open gripper",      "dz < 0"),
    # ── Single-axis rotation ──
    "tilt_left_20deg":   ("tilt left 20 degrees, open gripper",  "droll > 0"),
    "tilt_right_20deg":  ("tilt right 20 degrees, open gripper", "droll < 0"),
    "tilt_back_20deg":   ("tilt back 20 degrees, open gripper",  "dpitch > 0"),
    "tilt_fwd_20deg":    ("tilt forward 20 degrees, open gripper", "dpitch < 0"),
    "rotate_ccw_20deg":  ("rotate counterclockwise 20 degrees, open gripper", "dyaw > 0"),
    "rotate_cw_20deg":   ("rotate clockwise 20 degrees, open gripper", "dyaw < 0"),
    # ── Gripper ──
    "close_gripper":     ("close gripper",                       "grip < 0.5"),
    "open_gripper":      ("open gripper",                        "grip >= 0.5"),
    # ── Translation combos ──
    "fwd_up_close":      ("move forward 3.0 cm, move up 2.0 cm, close gripper",
                          "dx > 0, dz > 0"),
    "left_down_open":    ("move left 4.0 cm, move down 2.0 cm, open gripper",
                          "dy > 0, dz < 0"),
    "back_right":        ("move back 4.0 cm, move right 3.0 cm, close gripper",
                          "dx < 0, dy < 0"),
    "fwd_left_up":       ("move forward 3.0 cm, move left 2.0 cm, move up 2.0 cm, open gripper",
                          "dx > 0, dy > 0, dz > 0"),
    "back_left_down":    ("move back 3.0 cm, move left 2.0 cm, move down 2.0 cm, open gripper",
                          "dx < 0, dy > 0, dz < 0"),
    # ── Translation + rotation combos ──
    "up_tilt_right":     ("move up 3.0 cm, tilt right 10 degrees, open gripper",
                          "dz > 0, droll < 0"),
    "fwd_tilt_left":     ("move forward 3.0 cm, tilt left 10 degrees, open gripper",
                          "dx > 0, droll > 0"),
    "down_tilt_back":    ("move down 3.0 cm, tilt back 10 degrees, close gripper",
                          "dz < 0, dpitch > 0"),
    "fwd_rotate_cw":     ("move forward 3.0 cm, rotate clockwise 10 degrees, close gripper",
                          "dx > 0, dyaw < 0"),
    "left_tilt_right_cw": ("move left 3.0 cm, tilt right 10 degrees, rotate clockwise 10 degrees, open gripper",
                           "dy > 0, droll < 0, dyaw < 0"),
    # ── Rotation combos ──
    "tilt_left_tilt_back": ("tilt left 10 degrees, tilt back 10 degrees, open gripper",
                            "droll > 0, dpitch > 0"),
    "tilt_right_tilt_fwd": ("tilt right 10 degrees, tilt forward 10 degrees, close gripper",
                            "droll < 0, dpitch < 0"),
    "tilt_right_rotate_ccw": ("tilt right 10 degrees, rotate counterclockwise 10 degrees, open gripper",
                              "droll < 0, dyaw > 0"),
    # ── Full 6-axis combos (from training data distribution) ──
    "data_6axis_a":      ("move forward 2.5 cm, move right 1.0 cm, move up 3.0 cm, tilt left 10 degrees, tilt back 10 degrees, rotate clockwise 10 degrees, close gripper",
                          "dx > 0, dy < 0, dz > 0, droll > 0, dpitch > 0, dyaw < 0"),
    "data_6axis_b":      ("move forward 1.5 cm, move left 2.0 cm, move down 3.0 cm, tilt right 10 degrees, tilt forward 10 degrees, rotate counterclockwise 10 degrees, open gripper",
                          "dx > 0, dy > 0, dz < 0, droll < 0, dpitch < 0, dyaw > 0"),
}

CHECK_PROMPTS_NONUM = {
    # ── Single-axis translation ──
    "move_forward":      ("move forward, open gripper",          "dx > 0"),
    "move_back":         ("move back, open gripper",             "dx < 0"),
    "move_left":         ("move left, open gripper",             "dy > 0"),
    "move_right":        ("move right, open gripper",            "dy < 0"),
    "move_up":           ("move up, open gripper",               "dz > 0"),
    "move_down":         ("move down, open gripper",             "dz < 0"),
    # ── Single-axis rotation ──
    "tilt_left":         ("tilt left, open gripper",             "droll > 0"),
    "tilt_right":        ("tilt right, open gripper",            "droll < 0"),
    "tilt_back":         ("tilt back, open gripper",             "dpitch > 0"),
    "tilt_fwd":          ("tilt forward, open gripper",          "dpitch < 0"),
    "rotate_ccw":        ("rotate counterclockwise, open gripper", "dyaw > 0"),
    "rotate_cw":         ("rotate clockwise, open gripper",      "dyaw < 0"),
    # ── Gripper ──
    "close_gripper":     ("close gripper",                       "grip < 0.5"),
    "open_gripper":      ("open gripper",                        "grip >= 0.5"),
    # ── Translation combos ──
    "fwd_up_close":      ("move forward, move up, close gripper",
                          "dx > 0, dz > 0"),
    "left_down_open":    ("move left, move down, open gripper",
                          "dy > 0, dz < 0"),
    "back_right":        ("move back, move right, close gripper",
                          "dx < 0, dy < 0"),
    "fwd_left_up":       ("move forward, move left, move up, open gripper",
                          "dx > 0, dy > 0, dz > 0"),
    "back_left_down":    ("move back, move left, move down, open gripper",
                          "dx < 0, dy > 0, dz < 0"),
    # ── Translation + rotation combos ──
    "up_tilt_right":     ("move up, tilt right, open gripper",
                          "dz > 0, droll < 0"),
    "fwd_tilt_left":     ("move forward, tilt left, open gripper",
                          "dx > 0, droll > 0"),
    "down_tilt_back":    ("move down, tilt back, close gripper",
                          "dz < 0, dpitch > 0"),
    "fwd_rotate_cw":     ("move forward, rotate clockwise, close gripper",
                          "dx > 0, dyaw < 0"),
    "left_tilt_right_cw": ("move left, tilt right, rotate clockwise, open gripper",
                           "dy > 0, droll < 0, dyaw < 0"),
    # ── Rotation combos ──
    "tilt_left_tilt_back": ("tilt left, tilt back, open gripper",
                            "droll > 0, dpitch > 0"),
    "tilt_right_tilt_fwd": ("tilt right, tilt forward, close gripper",
                            "droll < 0, dpitch < 0"),
    "tilt_right_rotate_ccw": ("tilt right, rotate counterclockwise, open gripper",
                              "droll < 0, dyaw > 0"),
    # ── Full 6-axis combos (from training data, top frequency) ──
    "data_6axis_a":      ("move forward, move right, move up, tilt left, tilt back, rotate clockwise, close gripper",
                          "dx > 0, dy < 0, dz > 0, droll > 0, dpitch > 0, dyaw < 0"),
    "data_6axis_b":      ("move forward, move left, move down, tilt right, tilt forward, rotate counterclockwise, open gripper",
                          "dx > 0, dy > 0, dz < 0, droll < 0, dpitch < 0, dyaw > 0"),
}

PROMPT_MODE_MAP = {
    "verbose": CHECK_PROMPTS_VERBOSE,
    "nonum": CHECK_PROMPTS_NONUM,
}

# Default for backward compatibility
CHECK_PROMPTS = CHECK_PROMPTS_VERBOSE

DIM_NAMES = ["dx", "dy", "dz", "droll", "dpitch", "dyaw", "grip"]
DIM_INDEX = {name: i for i, name in enumerate(DIM_NAMES)}


def parse_expected(expected_str):
    """Parse expected conditions like 'dx > 0, dz > 0, droll < 0' into list of (dim_idx, op, val)."""
    conditions = []
    for cond in expected_str.split(", "):
        parts = cond.strip().split()
        dim_name = parts[0]
        op = parts[1]
        val = float(parts[2])
        conditions.append((DIM_INDEX[dim_name], dim_name, op, val))
    return conditions


def check_conditions(summed, conditions):
    """Check all conditions against summed action. Returns (all_pass, per_condition_results)."""
    per_cond = []
    for dim_idx, dim_name, op, val in conditions:
        v = summed[dim_idx]
        if op == ">":
            ok = v > val
        elif op == "<":
            ok = v < val
        elif op == ">=":
            ok = v >= val
        elif op == "<=":
            ok = v <= val
        else:
            ok = False
        per_cond.append((dim_name, op, val, v, ok))
    all_pass = all(ok for _, _, _, _, ok in per_cond)
    return all_pass, per_cond


def run_direction_check(cfg, model, processor, action_head, proprio_projector, noisy_action_projector, resize_size, observation, state_label):
    """Run all directional prompts and print results for a single state."""
    print(f"\n{'='*120}")
    print(f"  State: {state_label}")
    print(f"{'='*120}")

    results = []
    n_conditions_ok = 0
    n_conditions_total = 0

    for name, (prompt, expected_str) in CHECK_PROMPTS.items():
        with torch.no_grad():
            actions = get_action(
                cfg=cfg, model=model, obs=observation, task_label=prompt,
                processor=processor, action_head=action_head,
                proprio_projector=proprio_projector,
                noisy_action_projector=noisy_action_projector,
                use_film=cfg.use_film,
            )
        actions = np.array(actions)  # (8, 7)
        summed = actions.sum(axis=0)  # (7,)

        conditions = parse_expected(expected_str)
        all_pass, per_cond = check_conditions(summed, conditions)

        # Build per-condition result string
        cond_strs = []
        for dim_name, op, val, actual, ok in per_cond:
            mark = "ok" if ok else "FAIL"
            cond_strs.append(f"{dim_name}{op}{val:g}:{actual:+.3f}[{mark}]")
            n_conditions_total += 1
            if ok:
                n_conditions_ok += 1

        overall = "OK" if all_pass else "WRONG"
        all_dims = " ".join(f"{DIM_NAMES[i]}={v:+.3f}" for i, v in enumerate(summed))

        print(f"\n  {name:<35s} [{overall}]")
        print(f"    prompt:  {prompt}")
        print(f"    checks:  {', '.join(cond_strs)}")
        print(f"    all_dims: {all_dims}")

        results.append((name, all_pass, summed, per_cond))

    # Summary
    n_prompts_ok = sum(1 for _, ok, _, _ in results if ok)
    n_prompts_total = len(results)
    print(f"\n{'='*120}")
    print(f"  Prompt-level accuracy:    {n_prompts_ok}/{n_prompts_total} ({100*n_prompts_ok/n_prompts_total:.0f}%) prompts with ALL conditions correct")
    print(f"  Condition-level accuracy: {n_conditions_ok}/{n_conditions_total} ({100*n_conditions_ok/n_conditions_total:.0f}%) individual conditions correct")

    # Per-dimension breakdown
    dim_ok = {d: 0 for d in DIM_NAMES}
    dim_total = {d: 0 for d in DIM_NAMES}
    for _, _, _, per_cond in results:
        for dim_name, _, _, _, ok in per_cond:
            dim_total[dim_name] += 1
            if ok:
                dim_ok[dim_name] += 1
    print(f"\n  Per-dimension accuracy:")
    for d in DIM_NAMES:
        if dim_total[d] > 0:
            print(f"    {d:>8s}: {dim_ok[d]}/{dim_total[d]} ({100*dim_ok[d]/dim_total[d]:.0f}%)")

    return n_prompts_ok, n_prompts_total, results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_checkpoint", type=str, required=True)
    parser.add_argument("--task_suite_name", type=str, default="libero_goal")
    parser.add_argument("--task_id", type=int, default=0)
    parser.add_argument("--precomputed_states_dir", type=str, default=None,
                        help="Path to precomputed sim states (from generate_mid_traj_states.py). "
                             "If provided, also tests at chunk 3 and chunk 6 states.")
    parser.add_argument("--mid_rollout_chunks", type=int, nargs="+", default=[3, 6],
                        help="Which chunk states to test (only used with --precomputed_states_dir).")
    parser.add_argument("--prompt_mode", type=str, default="verbose",
                        choices=["verbose", "nonum"],
                        help="Prompt style: 'verbose' (with magnitudes) or 'nonum' (directions only).")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Directory to save results (JSON + text log). Default: auto-generated next to checkpoint.")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--lora_rank", type=int, default=32)
    parser.add_argument("--unnorm_key", type=str, default="")
    args = parser.parse_args()

    # Select prompt set based on mode
    global CHECK_PROMPTS
    CHECK_PROMPTS = PROMPT_MODE_MAP[args.prompt_mode]
    print(f"Prompt mode: {args.prompt_mode} ({len(CHECK_PROMPTS)} prompts)")

    cfg = SteerConfig(
        pretrained_checkpoint=args.pretrained_checkpoint,
        task_suite_name=args.task_suite_name,
        unnorm_key=args.unnorm_key,
        seed=args.seed,
        lora_rank=args.lora_rank,
    )
    set_seed_everywhere(cfg.seed)

    # Determine output dir
    if args.output_dir:
        output_dir = args.output_dir
    else:
        ckpt_name = os.path.basename(args.pretrained_checkpoint.rstrip("/"))
        output_dir = os.path.join("direction_check_results", f"{ckpt_name}_task{args.task_id}")
    os.makedirs(output_dir, exist_ok=True)

    # Tee stdout to a log file
    log_path = os.path.join(output_dir, "log.txt")
    log_file = open(log_path, "w")

    class Tee:
        def __init__(self, *files):
            self.files = files
        def write(self, data):
            for f in self.files:
                f.write(data)
                f.flush()
        def flush(self):
            for f in self.files:
                f.flush()

    original_stdout = sys.stdout
    sys.stdout = Tee(original_stdout, log_file)

    print("Loading model...")
    model = get_model(cfg)
    proprio_projector = get_proprio_projector(cfg, model.llm_dim, proprio_dim=8) if cfg.use_proprio else None
    action_head = get_action_head(cfg, model.llm_dim) if cfg.use_l1_regression else None
    noisy_action_projector = None
    processor = get_processor(cfg)
    check_unnorm_key(cfg, model)
    resize_size = get_image_resize_size(cfg)

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_name]()
    task = task_suite.get_task(args.task_id)
    env, task_description = get_libero_env(task, cfg.model_family, resolution=cfg.env_img_res)
    task_label = task_description.replace(" ", "_").replace("/", "_")

    print(f"Task {args.task_id}: {task_description}")
    print(f"Checkpoint: {args.pretrained_checkpoint}")
    print(f"Output: {output_dir}")

    initial_states = task_suite.get_task_init_states(args.task_id)
    initial_state = initial_states[0]

    all_summary = {}
    all_json_data = {
        "checkpoint": args.pretrained_checkpoint,
        "task_suite": args.task_suite_name,
        "task_id": args.task_id,
        "task_description": task_description,
        "seed": args.seed,
        "states": {},
    }

    # ── Initial state ──
    env.reset()
    obs = env.set_init_state(initial_state)
    for _ in range(cfg.num_steps_wait):
        obs, _, _, _ = env.step(get_libero_dummy_action(cfg.model_family))

    observation, _ = prepare_observation(obs, resize_size)
    n_ok, n_total, results = run_direction_check(
        cfg, model, processor, action_head, proprio_projector,
        noisy_action_projector, resize_size, observation, "INITIAL"
    )
    all_summary["initial"] = (n_ok, n_total)
    all_json_data["states"]["initial"] = _results_to_dict(results, n_ok, n_total)

    # ── Mid-rollout states ──
    if args.precomputed_states_dir:
        for num_chunks in args.mid_rollout_chunks:
            state_path = os.path.join(
                args.precomputed_states_dir,
                f"task_{args.task_id}_{task_label}",
                f"sim_state_chunk_{num_chunks}.npy",
            )
            if not os.path.exists(state_path):
                print(f"\nSkipping chunk {num_chunks}: {state_path} not found")
                continue

            print(f"\nLoading sim state from: {state_path}")
            sim_state = np.load(state_path)
            obs_mid = reset_env_to_sim_state(env, sim_state, cfg)
            observation_mid, _ = prepare_observation(obs_mid, resize_size)

            state_key = f"chunk_{num_chunks}"
            n_ok, n_total, results = run_direction_check(
                cfg, model, processor, action_head, proprio_projector,
                noisy_action_projector, resize_size, observation_mid,
                f"AFTER CHUNK {num_chunks} ({num_chunks * 8} steps)"
            )
            all_summary[state_key] = (n_ok, n_total)
            all_json_data["states"][state_key] = _results_to_dict(results, n_ok, n_total)

    # ── Overall summary ──
    print(f"\n{'='*60}")
    print("OVERALL SUMMARY")
    print(f"{'='*60}")
    for state_name, (n_ok, n_total) in all_summary.items():
        print(f"  {state_name:<20s}: {n_ok}/{n_total} ({100*n_ok/n_total:.0f}%)")

    env.close()

    # ── Save JSON ──
    json_path = os.path.join(output_dir, "results.json")
    with open(json_path, "w") as f:
        json.dump(all_json_data, f, indent=2)

    # Restore stdout
    sys.stdout = original_stdout
    log_file.close()

    print(f"\nResults saved to: {output_dir}/")
    print(f"  results.json  — structured data (actions, conditions, per-dim accuracy)")
    print(f"  log.txt       — full text log")


def _results_to_dict(results, n_prompts_ok, n_prompts_total):
    """Convert run_direction_check results to a JSON-serializable dict."""
    prompts_data = {}
    dim_ok = {d: 0 for d in DIM_NAMES}
    dim_total = {d: 0 for d in DIM_NAMES}
    n_cond_ok = 0
    n_cond_total = 0

    for name, all_pass, summed, per_cond in results:
        prompt_text = CHECK_PROMPTS[name][0]
        expected_str = CHECK_PROMPTS[name][1]
        cond_results = []
        for dim_name, op, val, actual, ok in per_cond:
            cond_results.append({
                "dim": dim_name,
                "op": op,
                "threshold": val,
                "actual": round(float(actual), 4),
                "correct": ok,
            })
            dim_total[dim_name] += 1
            n_cond_total += 1
            if ok:
                dim_ok[dim_name] += 1
                n_cond_ok += 1

        prompts_data[name] = {
            "prompt": prompt_text,
            "expected": expected_str,
            "all_correct": all_pass,
            "summed_actions": {DIM_NAMES[i]: round(float(v), 4) for i, v in enumerate(summed)},
            "conditions": cond_results,
        }

    per_dim = {}
    for d in DIM_NAMES:
        if dim_total[d] > 0:
            per_dim[d] = {"correct": dim_ok[d], "total": dim_total[d], "accuracy": round(dim_ok[d] / dim_total[d], 3)}

    return {
        "prompt_accuracy": {"correct": n_prompts_ok, "total": n_prompts_total},
        "condition_accuracy": {"correct": n_cond_ok, "total": n_cond_total},
        "per_dim_accuracy": per_dim,
        "prompts": prompts_data,
    }


if __name__ == "__main__":
    main()
