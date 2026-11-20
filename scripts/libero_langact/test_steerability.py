"""
Test language steerability of a language-action fine-tuned OpenVLA-OFT model.

Loads a trained checkpoint, runs LIBERO environments, and tests steering at
both the initial state AND mid-rollout states (after executing the original
task description for N chunks). For each test state, sends a battery of
language-action prompts covering all directions, rotations, gripper, and
combinations. Generates before/after images, rollout videos, and comparison grids.

Usage:
    cd /proj/inf-scaling/zwhong/idan/vla-rl/openvla-oft

    # Full test
    python scripts/libero_langact/test_steerability.py \
        --pretrained_checkpoint /path/to/langact/checkpoint \
        --task_suite_name libero_goal \
        --num_tasks 3 \
        --output_dir ./steerability_results

    # Quick test with subset of prompts
    python scripts/libero_langact/test_steerability.py \
        --pretrained_checkpoint /path/to/langact/checkpoint \
        --num_tasks 1 \
        --prompts move_left_v move_right_v tilt_left_v tilt_right_v \
        --output_dir ./steerability_quick
"""

import argparse
import os
import sys
import textwrap
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import imageio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

# Add project paths
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
    invert_gripper_action,
    normalize_gripper_action,
    set_seed_everywhere,
)

from libero.libero import benchmark


# ─── Test prompts: full coverage of all language-action primitives ────────────

STEER_PROMPTS_VERBOSE = OrderedDict(
    [
        # ── Translation (single axis, 5cm = ~p75) ──
        ("move_forward", "move forward 5.0 cm, open gripper"),
        ("move_back", "move back 5.0 cm, open gripper"),
        ("move_left", "move left 5.0 cm, open gripper"),
        ("move_right", "move right 5.0 cm, open gripper"),
        ("move_up", "move up 5.0 cm, open gripper"),
        ("move_down", "move down 5.0 cm, open gripper"),
        # ── Rotation (single axis) ──
        ("tilt_left", "tilt left 20 degrees, open gripper"),
        ("tilt_right", "tilt right 20 degrees, open gripper"),
        ("tilt_back", "tilt back 20 degrees, open gripper"),
        ("tilt_forward", "tilt forward 20 degrees, open gripper"),
        ("rotate_ccw", "rotate counterclockwise 20 degrees, open gripper"),
        ("rotate_cw", "rotate clockwise 20 degrees, open gripper"),
        # ── Gripper ──
        ("close_gripper", "close gripper"),
        ("open_gripper", "open gripper"),
        # ── Magnitude sweep: move_forward at different magnitudes ──
        # Data range: 0.1-10.3 cm, median=1.8, p25=0.5, p75=4.3, p95=8.3
        ("fwd_0.2cm", "move forward 0.2 cm, open gripper"),
        ("fwd_0.5cm", "move forward 0.5 cm, open gripper"),
        ("fwd_1.0cm", "move forward 1.0 cm, open gripper"),
        ("fwd_2.0cm", "move forward 2.0 cm, open gripper"),
        ("fwd_4.0cm", "move forward 4.0 cm, open gripper"),
        ("fwd_6.0cm", "move forward 6.0 cm, open gripper"),
        ("fwd_8.0cm", "move forward 8.0 cm, open gripper"),
        # ── Magnitude sweep: move_left at different magnitudes ──
        ("left_0.5cm", "move left 0.5 cm, open gripper"),
        ("left_2.0cm", "move left 2.0 cm, open gripper"),
        ("left_4.0cm", "move left 4.0 cm, open gripper"),
        ("left_8.0cm", "move left 8.0 cm, open gripper"),
        # ── Magnitude sweep: move_up at different magnitudes ──
        ("up_0.5cm", "move up 0.5 cm, open gripper"),
        ("up_2.0cm", "move up 2.0 cm, open gripper"),
        ("up_4.0cm", "move up 4.0 cm, open gripper"),
        ("up_8.0cm", "move up 8.0 cm, open gripper"),
        # ── Magnitude sweep: rotation ──
        ("tilt_right_10deg", "tilt right 10 degrees, open gripper"),
        ("tilt_right_20deg", "tilt right 20 degrees, open gripper"),
        # ── 2-axis combos ──
        ("combo_fwd_up_close", "move forward 3.0 cm, move up 2.0 cm, close gripper"),
        ("combo_left_down_open", "move left 4.0 cm, move down 2.0 cm, open gripper"),
        ("combo_up_tilt_right", "move up 3.0 cm, tilt right 10 degrees, open gripper"),
        ("combo_fwd_rotate_cw_close", "move forward 3.0 cm, rotate clockwise 10 degrees, close gripper"),
        ("combo_up_tilt_right_rotate_ccw", "move up 2.0 cm, tilt right 10 degrees, rotate counterclockwise 10 degrees, open gripper"),
        ("combo_back_tilt_left_close", "move back 3.0 cm, tilt left 10 degrees, close gripper"),
        # ── Data-sampled multi-axis combos (from actual training data) ──
        ("data_down_right_tilt_close", "move down 4.8 cm, move right 1.3 cm, tilt right 10 degrees, tilt back 10 degrees, rotate counterclockwise 10 degrees, open gripper"),
        ("data_fwd_right_up_close", "move forward 2.5 cm, move right 1.0 cm, move up 3.0 cm, tilt left 10 degrees, tilt back 10 degrees, rotate clockwise 10 degrees, close gripper"),
        ("data_fwd_left_down_open", "move forward 1.5 cm, move left 2.0 cm, move down 3.0 cm, tilt right 10 degrees, tilt forward 10 degrees, rotate counterclockwise 10 degrees, open gripper"),
        # ── Opposite-direction pairs (same magnitudes, flipped directions) ──
        ("pair_a_fwd_right_up", "move forward 3.0 cm, move right 2.0 cm, move up 2.0 cm, open gripper"),
        ("pair_a_back_left_down", "move back 3.0 cm, move left 2.0 cm, move down 2.0 cm, open gripper"),
        ("pair_b_fwd_left", "move forward 4.0 cm, move left 3.0 cm, close gripper"),
        ("pair_b_back_right", "move back 4.0 cm, move right 3.0 cm, close gripper"),
    ]
)

STEER_PROMPTS_NONUM = OrderedDict(
    [
        # ── Translation (single axis) ──
        ("move_forward", "move forward, open gripper"),
        ("move_back", "move back, open gripper"),
        ("move_left", "move left, open gripper"),
        ("move_right", "move right, open gripper"),
        ("move_up", "move up, open gripper"),
        ("move_down", "move down, open gripper"),
        # ── Rotation (single axis) ──
        ("tilt_left", "tilt left, open gripper"),
        ("tilt_right", "tilt right, open gripper"),
        ("tilt_back", "tilt back, open gripper"),
        ("tilt_forward", "tilt forward, open gripper"),
        ("rotate_ccw", "rotate counterclockwise, open gripper"),
        ("rotate_cw", "rotate clockwise, open gripper"),
        # ── Gripper ──
        ("close_gripper", "close gripper"),
        ("open_gripper", "open gripper"),
        # ── 2-axis combos ──
        ("combo_fwd_up_close", "move forward, move up, close gripper"),
        ("combo_left_down_open", "move left, move down, open gripper"),
        ("combo_up_tilt_right", "move up, tilt right, open gripper"),
        ("combo_fwd_rotate_cw_close", "move forward, rotate clockwise, close gripper"),
        ("combo_up_tilt_right_rotate_ccw", "move up, tilt right, rotate counterclockwise, open gripper"),
        ("combo_back_tilt_left_close", "move back, tilt left, close gripper"),
        # ── Data-sampled: 3-motion (frequent in training data) ──
        ("data_3m_a", "tilt right, tilt forward, rotate clockwise, close gripper"),
        ("data_3m_b", "tilt right, tilt back, rotate clockwise, close gripper"),
        ("data_3m_c", "tilt left, tilt back, rotate counterclockwise, open gripper"),
        # ── Data-sampled: 4-motion ──
        ("data_4m_a", "move up, tilt right, tilt forward, rotate counterclockwise, open gripper"),
        ("data_4m_b", "move left, tilt right, tilt forward, rotate clockwise, open gripper"),
        ("data_4m_c", "move forward, tilt right, tilt forward, rotate clockwise, open gripper"),
        ("data_4m_d", "move down, tilt right, tilt forward, rotate clockwise, open gripper"),
        # ── Data-sampled: 5-motion ──
        ("data_5m_a", "move forward, move down, tilt right, tilt forward, rotate clockwise, open gripper"),
        ("data_5m_b", "move back, move down, tilt right, tilt forward, rotate clockwise, open gripper"),
        ("data_5m_c", "move back, move down, tilt right, tilt back, rotate clockwise, open gripper"),
        # ── Data-sampled: 6-motion (top-1 most frequent = 224x) ──
        ("data_6m_top1", "move forward, move right, move up, tilt left, tilt back, rotate clockwise, close gripper"),
        ("data_6m_top2", "move forward, move left, move down, tilt right, tilt back, rotate counterclockwise, open gripper"),
        ("data_6m_top3", "move forward, move left, move down, tilt right, tilt forward, rotate counterclockwise, open gripper"),
        ("data_6m_top4", "move forward, move right, move down, tilt right, tilt forward, rotate clockwise, open gripper"),
        # ── Opposite-direction pairs (from data, for steerability comparison) ──
        # Pair A: top-1 vs its translation-flipped version
        ("pair_a_orig", "move forward, move right, move up, tilt left, tilt back, rotate clockwise, close gripper"),
        ("pair_a_flip", "move back, move left, move down, tilt left, tilt back, rotate clockwise, close gripper"),
        # Pair B: top-2 vs its translation-flipped version
        ("pair_b_orig", "move forward, move left, move down, tilt right, tilt back, rotate counterclockwise, open gripper"),
        ("pair_b_flip", "move back, move right, move up, tilt right, tilt back, rotate counterclockwise, open gripper"),
        # Pair C: top-3 vs its translation-flipped version
        ("pair_c_orig", "move forward, move left, move down, tilt right, tilt forward, rotate counterclockwise, open gripper"),
        ("pair_c_flip", "move back, move right, move up, tilt right, tilt forward, rotate counterclockwise, open gripper"),
    ]
)

STEER_PROMPTS_ALL = OrderedDict()
for k, v in STEER_PROMPTS_VERBOSE.items():
    STEER_PROMPTS_ALL[f"{k}_v"] = v
for k, v in STEER_PROMPTS_NONUM.items():
    STEER_PROMPTS_ALL[f"{k}_nn"] = v

PROMPT_SETS = {
    "nonum": STEER_PROMPTS_NONUM,
    "verbose": STEER_PROMPTS_VERBOSE,
    "all": STEER_PROMPTS_ALL,
}


# ─── Config ──────────────────────────────────────────────────────────────────────


@dataclass
class SteerConfig:
    model_family: str = "openvla"
    pretrained_checkpoint: str = ""
    use_l1_regression: bool = True
    use_diffusion: bool = False
    num_diffusion_steps_train: int = 50
    num_diffusion_steps_inference: int = 50
    use_film: bool = False
    num_images_in_input: int = 2
    use_proprio: bool = True
    center_crop: bool = True
    num_open_loop_steps: int = 8
    lora_rank: int = 32
    unnorm_key: str = ""
    load_in_8bit: bool = False
    load_in_4bit: bool = False
    task_suite_name: str = "libero_goal"
    num_steps_wait: int = 10
    env_img_res: int = 256
    seed: int = 7


# ─── Helper functions ────────────────────────────────────────────────────────────


def prepare_observation(obs, resize_size):
    """Prepare observation for policy input."""
    img = get_libero_image(obs)
    wrist_img = get_libero_wrist_image(obs)
    img_resized = resize_image_for_policy(img, resize_size)
    wrist_img_resized = resize_image_for_policy(wrist_img, resize_size)
    observation = {
        "full_image": img_resized,
        "wrist_image": wrist_img_resized,
        "state": np.concatenate(
            (obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"])
        ),
    }
    return observation, img


def process_action(action):
    """Normalize gripper and invert for environment."""
    action = normalize_gripper_action(action, binarize=True)
    action = invert_gripper_action(action)
    return action


def get_eef_state(obs):
    """Extract full EEF state from observation."""
    return {
        "pos": obs["robot0_eef_pos"].copy(),
        "quat": obs["robot0_eef_quat"].copy(),
        "axang": quat2axisangle(obs["robot0_eef_quat"]),
        "gripper": obs["robot0_gripper_qpos"].copy(),
    }


def check_unnorm_key(cfg, model):
    """Check and set the action un-normalization key."""
    # If already set via --unnorm_key, verify it exists
    if cfg.unnorm_key:
        if cfg.unnorm_key in model.norm_stats:
            return
        assert False, (
            f"Explicitly provided unnorm_key '{cfg.unnorm_key}' not found! "
            f"Available keys: {list(model.norm_stats.keys())}"
        )

    base = cfg.task_suite_name  # e.g. "libero_goal"
    # Try progressively more specific keys until one matches
    candidates = [
        base,
        f"{base}_no_noops",
        f"{base}_no_noops_langact_verbose",
        f"{base}_no_noops_langact_nonum",
    ]
    for key in candidates:
        if key in model.norm_stats:
            cfg.unnorm_key = key
            return
    # If none matched, try any key that starts with "libero" (handles combined datasets)
    for key in model.norm_stats:
        if key.startswith(base) or key.startswith("libero"):
            cfg.unnorm_key = key
            print(f"  Auto-detected unnorm_key: {key}")
            return
    assert False, (
        f"Action un-norm key not found for '{base}'! "
        f"Available keys: {list(model.norm_stats.keys())}. "
        f"For combined models, pass --unnorm_key explicitly."
    )


def initialize_model(cfg):
    """Load model and all components."""
    model = get_model(cfg)
    proprio_projector = None
    if cfg.use_proprio:
        proprio_projector = get_proprio_projector(cfg, model.llm_dim, proprio_dim=8)
    action_head = None
    if cfg.use_l1_regression or cfg.use_diffusion:
        action_head = get_action_head(cfg, model.llm_dim)
    noisy_action_projector = None
    if cfg.use_diffusion:
        noisy_action_projector = get_noisy_action_projector(cfg, model.llm_dim)
    processor = get_processor(cfg)
    check_unnorm_key(cfg, model)
    return model, action_head, proprio_projector, noisy_action_projector, processor


def reset_env_to_initial(env, initial_state, cfg):
    """Reset environment to a LIBERO initial state (from task_suite.get_task_init_states).

    Uses LIBERO's set_init_state API which properly initializes everything
    and returns a valid observation.
    """
    env.reset()
    obs = env.set_init_state(initial_state)
    for _ in range(cfg.num_steps_wait):
        obs, _, _, _ = env.step(get_libero_dummy_action(cfg.model_family))
    return obs


def reset_env_to_sim_state(env, sim_state, cfg):
    """Reset environment to a saved MuJoCo sim state (for mid-rollout snapshots).

    Uses low-level sim state restore. Steps one dummy action to produce a valid
    observation (OffScreenRenderEnv has no get_observation method). The dummy
    action is all-zeros with gripper=-1 (no movement), so the effect is negligible.
    """
    env.reset()
    env.sim.set_state_from_flattened(sim_state)
    env.sim.forward()
    obs, _, _, _ = env.step(get_libero_dummy_action(cfg.model_family))
    for _ in range(cfg.num_steps_wait - 1):
        obs, _, _, _ = env.step(get_libero_dummy_action(cfg.model_family))
    return obs


def _compute_langact_from_obs(obs_before, obs_after, window_size=8):
    """Compute verbose language-action text from actual EEF state changes.

    Takes the observations before and after an action chunk and produces
    the language-action text describing what happened.
    """
    from langact_utils import axis_angle_to_euler, euler_diff, summarize_numeric_actions

    # Extract EEF states: [pos(3), axis_angle(3)]
    state_before = np.concatenate([
        obs_before["robot0_eef_pos"],
        quat2axisangle(obs_before["robot0_eef_quat"]),
    ])
    state_after = np.concatenate([
        obs_after["robot0_eef_pos"],
        quat2axisangle(obs_after["robot0_eef_quat"]),
    ])

    # Position delta
    delta_pos = state_after[:3] - state_before[:3]

    # Rotation delta via euler
    euler_before = axis_angle_to_euler(state_before[3:6].reshape(1, 3))[0]
    euler_after = axis_angle_to_euler(state_after[3:6].reshape(1, 3))[0]
    delta_rot = euler_diff(euler_after.reshape(1, 3), euler_before.reshape(1, 3))[0]

    # Gripper: use after state (1=open, 0=close after inversion)
    gripper = 1.0 - np.clip(obs_after["robot0_gripper_qpos"][0], 0, 1)

    # Build 7D vector and convert to text
    lang_action = np.concatenate([delta_pos, delta_rot, [gripper]])
    text = summarize_numeric_actions(lang_action, sum_decimal="1f", include_rotation=True)
    return text or "close gripper"


def advance_with_auto_langact(
    cfg, env, obs, initial_prompt, model, resize_size, processor,
    action_head, proprio_projector, noisy_action_projector, num_chunks,
):
    """Advance the environment using auto-computed language-action prompts.

    For the first chunk, uses initial_prompt. After each chunk, computes
    the actual language-action from the observed EEF state change and uses
    that as the prompt for the next chunk. This ensures the model always
    receives in-distribution language-action prompts.

    Returns the observation after advancing, all collected images, and
    the list of prompts used at each chunk.
    """
    images = []
    prompts_used = []
    current_obs = obs
    current_prompt = initial_prompt

    for chunk_idx in range(num_chunks):
        observation, img = prepare_observation(current_obs, resize_size)
        images.append(img)
        prompts_used.append(current_prompt)

        obs_before_chunk = current_obs

        with torch.no_grad():
            actions = get_action(
                cfg=cfg, model=model, obs=observation, task_label=current_prompt,
                processor=processor, action_head=action_head,
                proprio_projector=proprio_projector,
                noisy_action_projector=noisy_action_projector,
                use_film=cfg.use_film,
            )
        for action in actions:
            action = process_action(action)
            current_obs, _, _, _ = env.step(action.tolist())
            images.append(get_libero_image(current_obs))

        # Auto-compute language-action from what actually happened
        current_prompt = _compute_langact_from_obs(obs_before_chunk, current_obs)
        print(f"      chunk {chunk_idx}: executed with \"{prompts_used[-1][:60]}...\"")
        print(f"        -> actual motion: \"{current_prompt}\"")

    return current_obs, images, prompts_used


def run_steering_test(
    cfg, env, obs, prompt, model, resize_size, processor,
    action_head, proprio_projector, noisy_action_projector,
):
    """Run a single steering test: get actions for a prompt, execute them, return images + state."""
    observation, before_img = prepare_observation(obs, resize_size)

    with torch.no_grad():
        actions = get_action(
            cfg=cfg, model=model, obs=observation, task_label=prompt,
            processor=processor, action_head=action_head,
            proprio_projector=proprio_projector,
            noisy_action_projector=noisy_action_projector,
            use_film=cfg.use_film,
        )

    rollout_images = [before_img]
    current_obs = obs
    for action in actions:
        action = process_action(action)
        current_obs, _, _, _ = env.step(action.tolist())
        rollout_images.append(get_libero_image(current_obs))

    after_img = get_libero_image(current_obs)
    eef_after = get_eef_state(current_obs)

    return before_img, after_img, rollout_images, eef_after


def save_video(images, path, fps=30):
    """Save a list of images as an MP4 video."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    writer = imageio.get_writer(path, fps=fps)
    for img in images:
        writer.append_data(img)
    writer.close()


def format_eef_delta(eef_before, eef_after):
    """Format EEF state change as a readable string."""
    dp = eef_after["pos"] - eef_before["pos"]
    da = eef_after["axang"] - eef_before["axang"]
    lines = [
        f"pos:  dx={dp[0]:+.4f} dy={dp[1]:+.4f} dz={dp[2]:+.4f}",
        f"rot:  dr={da[0]:+.4f} dp={da[1]:+.4f} dy={da[2]:+.4f}",
        f"grip: {eef_after['gripper'][0]:.3f},{eef_after['gripper'][1]:.3f}",
    ]
    return "\n".join(lines)


def make_comparison_grid(results, title, output_path):
    """Create a comparison grid showing before/after for all prompts."""
    n = len(results)
    fig, axes = plt.subplots(n, 3, figsize=(16, 3.5 * n))
    if n == 1:
        axes = axes[None, :]

    for i, (prompt_name, data) in enumerate(results.items()):
        axes[i, 0].imshow(data["before"])
        axes[i, 0].set_title("Before", fontsize=9)
        axes[i, 0].axis("off")

        axes[i, 1].imshow(data["after"])
        axes[i, 1].set_title("After (8 steps)", fontsize=9)
        axes[i, 1].axis("off")

        axes[i, 2].axis("off")
        wrapped = "\n".join(textwrap.wrap(data["prompt"], width=45))
        delta_str = format_eef_delta(data["eef_before"], data["eef_after"])

        axes[i, 2].text(
            0.05, 0.80, f"Prompt:\n{wrapped}",
            transform=axes[i, 2].transAxes, fontsize=8, va="top", fontfamily="monospace",
            bbox=dict(boxstyle="round", facecolor="lightcyan", alpha=0.8),
        )
        axes[i, 2].text(
            0.05, 0.30, f"EEF delta:\n{delta_str}",
            transform=axes[i, 2].transAxes, fontsize=8, va="top", fontfamily="monospace",
            bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8),
        )

    plt.suptitle(title, fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved grid: {output_path}")


def run_steer_battery(
    cfg, env, obs, sim_state, prompts, model, resize_size, processor,
    action_head, proprio_projector, noisy_action_projector,
    output_dir, title,
):
    """Run the full battery of steering prompts from a given state, save all outputs."""
    os.makedirs(output_dir, exist_ok=True)
    eef_before = get_eef_state(obs)
    results = OrderedDict()

    for prompt_name, prompt_text in prompts.items():
        print(f"    {prompt_name} -> \"{prompt_text}\"")

        # Reset to the same state for each prompt
        obs_reset = reset_env_to_sim_state(env, sim_state, cfg)

        before_img, after_img, rollout_images, eef_after = run_steering_test(
            cfg, env, obs_reset, prompt_text, model, resize_size, processor,
            action_head, proprio_projector, noisy_action_projector,
        )

        # Save per-prompt outputs
        prompt_dir = os.path.join(output_dir, prompt_name)
        os.makedirs(prompt_dir, exist_ok=True)
        plt.imsave(os.path.join(prompt_dir, "before.png"), before_img)
        plt.imsave(os.path.join(prompt_dir, "after.png"), after_img)
        save_video(rollout_images, os.path.join(prompt_dir, "rollout.mp4"))

        dp = eef_after["pos"] - eef_before["pos"]
        print(f"      EEF delta: dx={dp[0]:+.4f} dy={dp[1]:+.4f} dz={dp[2]:+.4f}")

        results[prompt_name] = {
            "before": before_img,
            "after": after_img,
            "prompt": prompt_text,
            "eef_before": eef_before,
            "eef_after": eef_after,
        }

    make_comparison_grid(results, title, os.path.join(output_dir, "grid.png"))
    return results


# ─── Main ────────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Test language steerability of language-action VLA.")
    parser.add_argument("--pretrained_checkpoint", type=str, required=True)
    parser.add_argument("--task_suite_name", type=str, default="libero_goal",
                        choices=["libero_spatial", "libero_object", "libero_goal", "libero_10"])
    parser.add_argument("--num_tasks", type=int, default=3,
                        help="Number of tasks to test (starting from 0). Ignored if --task_ids is set.")
    parser.add_argument("--task_ids", type=int, nargs="+", default=None,
                        help="Specific task IDs to test. E.g., --task_ids 0 3 7")
    parser.add_argument("--output_dir", type=str, default="./steerability_results")
    parser.add_argument("--prompt_mode", type=str, default="nonum",
                        choices=["nonum", "verbose", "all"],
                        help="Which prompt set to use: 'nonum' (no numbers), 'verbose' (with numbers), 'all' (both).")
    parser.add_argument("--prompts", nargs="+", default=None,
                        help="Subset of prompt keys to test (overrides --prompt_mode). Default: use prompt_mode.")
    parser.add_argument("--mid_rollout_chunks", type=int, nargs="+", default=[3, 6],
                        help="Number of task-prompt action chunks to execute before mid-rollout tests. "
                             "E.g., '--mid_rollout_chunks 3 6' tests at chunk 3 and chunk 6.")
    parser.add_argument("--task_prompt_override", type=str, default=None,
                        help="Override the task description used for mid-rollout advancement. "
                             "Default: auto-compute language-action from actual EEF state changes.")
    parser.add_argument("--initial_advance_prompt", type=str,
                        default="move forward 3.0 cm, move down 2.0 cm, close gripper",
                        help="Language-action prompt for the first advance chunk. "
                             "Subsequent chunks auto-compute from actual motion.")
    parser.add_argument("--precomputed_states_dir", type=str, default=None,
                        help="Path to directory of saved sim states (from generate_mid_traj_states.py). "
                             "Skips on-the-fly advancement and loads states from .npy files instead.")
    parser.add_argument("--unnorm_key", type=str, default="")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--lora_rank", type=int, default=32)
    args = parser.parse_args()

    cfg = SteerConfig(
        pretrained_checkpoint=args.pretrained_checkpoint,
        task_suite_name=args.task_suite_name,
        unnorm_key=args.unnorm_key,
        seed=args.seed,
        lora_rank=args.lora_rank,
    )

    set_seed_everywhere(cfg.seed)

    # Select prompts
    prompt_set = PROMPT_SETS[args.prompt_mode]
    if args.prompts:
        prompts = OrderedDict((k, prompt_set[k]) for k in args.prompts if k in prompt_set)
    else:
        prompts = prompt_set

    print(f"Testing {len(prompts)} prompts on {args.num_tasks} tasks from {args.task_suite_name}")
    print(f"Mid-rollout test points: after chunks {args.mid_rollout_chunks}")
    print(f"Checkpoint: {args.pretrained_checkpoint}")
    print(f"Output: {args.output_dir}\n")

    # Load model
    print("Loading model...")
    model, action_head, proprio_projector, noisy_action_projector, processor = initialize_model(cfg)
    resize_size = get_image_resize_size(cfg)

    # Get task suite
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_name]()
    if args.task_ids is not None:
        task_ids = [t for t in args.task_ids if t < task_suite.n_tasks]
    else:
        task_ids = list(range(min(args.num_tasks, task_suite.n_tasks)))

    os.makedirs(args.output_dir, exist_ok=True)

    for task_id in task_ids:
        task = task_suite.get_task(task_id)
        env, task_description = get_libero_env(task, cfg.model_family, resolution=cfg.env_img_res)
        task_label = task_description.replace(" ", "_").replace("/", "_")

        print(f"\n{'='*70}")
        print(f"Task {task_id}: {task_description}")
        print(f"{'='*70}")

        # Get LIBERO initial state
        initial_states = task_suite.get_task_init_states(task_id)
        initial_state = initial_states[0]  # Use first initial state

        # Determine which task prompt to use for advancing
        advance_prompt = args.task_prompt_override or task_description

        # ── Test at initial state ──
        print(f"\n  [initial] Testing steering from initial state...")
        obs_init = reset_env_to_initial(env, initial_state, cfg)
        init_sim_state = env.sim.get_state().flatten()

        run_steer_battery(
            cfg, env, obs_init, init_sim_state, prompts,
            model, resize_size, processor,
            action_head, proprio_projector, noisy_action_projector,
            output_dir=os.path.join(args.output_dir, f"task_{task_id}_{task_label}", "initial"),
            title=f"Task {task_id}: {task_description}\n(initial state)",
        )

        # ── Test at mid-rollout states ──
        for num_chunks in args.mid_rollout_chunks:
            chunk_dir = os.path.join(args.output_dir, f"task_{task_id}_{task_label}", f"after_chunk_{num_chunks}")
            os.makedirs(chunk_dir, exist_ok=True)

            # Check for precomputed states
            precomputed_path = None
            if args.precomputed_states_dir:
                candidate = os.path.join(
                    args.precomputed_states_dir,
                    f"task_{task_id}_{task_label}",
                    f"sim_state_chunk_{num_chunks}.npy",
                )
                if os.path.exists(candidate):
                    precomputed_path = candidate

            if precomputed_path:
                # Load precomputed sim state (generated by generate_mid_traj_states.py)
                print(f"\n  [chunk_{num_chunks}] Loading precomputed sim state from {precomputed_path}")
                mid_sim_state = np.load(precomputed_path)
                obs_mid = reset_env_to_sim_state(env, mid_sim_state, cfg)

                with open(os.path.join(chunk_dir, "advance_prompts.txt"), "w") as f:
                    f.write(f"precomputed from: {precomputed_path}\n")

            elif args.task_prompt_override:
                # Legacy mode: use a fixed prompt for all chunks
                print(f"\n  [chunk_{num_chunks}] Advancing {num_chunks} chunks with fixed prompt: \"{args.task_prompt_override}\"...")
                obs_mid = reset_env_to_initial(env, initial_state, cfg)
                images = []
                current_obs = obs_mid
                for ci in range(num_chunks):
                    observation, img = prepare_observation(current_obs, resize_size)
                    images.append(img)
                    with torch.no_grad():
                        actions = get_action(
                            cfg=cfg, model=model, obs=observation, task_label=args.task_prompt_override,
                            processor=processor, action_head=action_head,
                            proprio_projector=proprio_projector,
                            noisy_action_projector=noisy_action_projector,
                            use_film=cfg.use_film,
                        )
                    for action in actions:
                        action = process_action(action)
                        current_obs, _, _, _ = env.step(action.tolist())
                        images.append(get_libero_image(current_obs))
                obs_mid = current_obs
                mid_sim_state = env.sim.get_state().flatten()
                save_video(images, os.path.join(chunk_dir, "advance_rollout.mp4"))
                with open(os.path.join(chunk_dir, "advance_prompts.txt"), "w") as f:
                    for ci in range(num_chunks):
                        f.write(f"chunk {ci}: {args.task_prompt_override}\n")

            else:
                # Auto-langact mode: compute language-action from actual motion
                print(f"\n  [chunk_{num_chunks}] Advancing {num_chunks} chunks with auto language-action...")
                obs_mid = reset_env_to_initial(env, initial_state, cfg)
                obs_mid, advance_images, prompts_used = advance_with_auto_langact(
                    cfg, env, obs_mid, args.initial_advance_prompt, model, resize_size, processor,
                    action_head, proprio_projector, noisy_action_projector,
                    num_chunks=num_chunks,
                )
                mid_sim_state = env.sim.get_state().flatten()
                save_video(advance_images, os.path.join(chunk_dir, "advance_rollout.mp4"))
                with open(os.path.join(chunk_dir, "advance_prompts.txt"), "w") as f:
                    for ci, p in enumerate(prompts_used):
                        f.write(f"chunk {ci}: {p}\n")

            print(f"  [chunk_{num_chunks}] Testing steering from mid-rollout state...")
            run_steer_battery(
                cfg, env, obs_mid, mid_sim_state, prompts,
                model, resize_size, processor,
                action_head, proprio_projector, noisy_action_projector,
                output_dir=chunk_dir,
                title=f"Task {task_id}: {task_description}\n(after {num_chunks} chunks, precomputed={'yes' if precomputed_path else 'no'})",
            )

        env.close()

    print(f"\nAll done! Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
