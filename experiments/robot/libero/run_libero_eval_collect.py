"""
run_libero_eval_collect.py

Runs OpenVLA-OFT evaluation in LIBERO and collects data at every action chunk
query point for offline GPT correction experiments.

Saves per-episode pickle files containing:
  - task_description, success flag
  - For each chunk: prev_obs image, curr_obs image, wrist image,
    OpenVLA action chunk (8x7), proprio state, step number

Usage:
    cd openvla-oft
    python experiments/robot/libero/run_libero_eval_collect.py \
        --pretrained_checkpoint <path_or_hf_id> \
        --task_suite_name libero_spatial \
        --num_trials_per_task 5 \
        --collect_dir ./collected_data
"""

import json
import logging
import os
import pickle
import sys
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import draccus
import numpy as np
import tqdm
from libero.libero import benchmark

# Append current directory so that interpreter can find experiments.robot
sys.path.append("../..")
from experiments.robot.libero.libero_utils import (
    get_libero_dummy_action,
    get_libero_env,
    get_libero_image,
    get_libero_wrist_image,
    quat2axisangle,
    save_rollout_video,
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


# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


@dataclass
class CollectConfig(BaseGenerateConfig):
    # Directory to save collected data
    collect_dir: str = "./collected_data"


def run_episode_collect(
    cfg,
    env,
    task_description,
    model,
    resize_size,
    processor=None,
    action_head=None,
    proprio_projector=None,
    noisy_action_projector=None,
    initial_state=None,
    log_file=None,
):
    """Run a single episode and collect data at every action chunk query."""
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

    # Data collection
    chunks = []
    prev_obs_img = None
    prev_wrist_img = None
    chunk_idx = 0

    try:
        while t < max_steps + cfg.num_steps_wait:
            # Stabilization phase
            if t < cfg.num_steps_wait:
                obs, reward, done, info = env.step(get_libero_dummy_action(cfg.model_family))
                t += 1
                continue

            # Prepare observation
            observation, img = prepare_observation(obs, resize_size)
            replay_images.append(img)

            # Also grab wrist image (raw, before resizing) for GPT
            wrist_img = get_libero_wrist_image(obs)

            if len(action_queue) == 0:
                # Query OpenVLA for action chunk
                actions = get_action(
                    cfg,
                    model,
                    observation,
                    task_description,
                    processor=processor,
                    action_head=action_head,
                    proprio_projector=proprio_projector,
                    noisy_action_projector=noisy_action_projector,
                    use_film=cfg.use_film,
                )

                # Collect data for this chunk
                chunk_data = {
                    "chunk_idx": chunk_idx,
                    "step": t,
                    "prev_obs_img": prev_obs_img.copy() if prev_obs_img is not None else None,
                    "prev_wrist_img": prev_wrist_img.copy() if prev_wrist_img is not None else None,
                    "curr_obs_img": img.copy(),
                    "curr_wrist_img": wrist_img.copy(),
                    "openvla_actions": [a.copy() for a in actions],
                    "proprio_state": observation["state"].copy(),
                }
                chunks.append(chunk_data)
                chunk_idx += 1

                # Update prev images for next chunk
                prev_obs_img = img.copy()
                prev_wrist_img = wrist_img.copy()

                action_queue.extend(actions)

            # Execute action
            action = action_queue.popleft()
            action = process_action(action, cfg.model_family)
            obs, reward, done, info = env.step(action.tolist())
            if done:
                success = True
                break
            t += 1

    except Exception as e:
        log_message(f"Episode error: {e}", log_file)

    episode_data = {
        "task_description": task_description,
        "success": success,
        "num_chunks": len(chunks),
        "total_steps": t,
        "chunks": chunks,
    }

    return success, replay_images, episode_data


@draccus.wrap()
def collect_libero(cfg: CollectConfig) -> None:
    """Run LIBERO eval and collect data for offline GPT correction."""
    validate_config(cfg)
    set_seed_everywhere(cfg.seed)

    # Initialize model
    model, action_head, proprio_projector, noisy_action_projector, processor = initialize_model(cfg)
    resize_size = get_image_resize_size(cfg)

    # Setup logging
    os.makedirs(cfg.local_log_dir, exist_ok=True)
    run_id = f"COLLECT-{cfg.task_suite_name}-{DATE_TIME}"
    local_log_filepath = os.path.join(cfg.local_log_dir, run_id + ".txt")
    log_file = open(local_log_filepath, "w")
    logger.info(f"Logging to: {local_log_filepath}")

    # Setup collection directory
    collect_suite_dir = os.path.join(cfg.collect_dir, cfg.task_suite_name)
    os.makedirs(collect_suite_dir, exist_ok=True)
    logger.info(f"Saving collected data to: {collect_suite_dir}")

    # Initialize LIBERO task suite
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[cfg.task_suite_name]()
    num_tasks = task_suite.n_tasks

    # Determine task range
    task_start = cfg.task_id_start
    task_end = num_tasks if cfg.task_id_end == -1 else min(cfg.task_id_end, num_tasks)

    log_message(f"Task suite: {cfg.task_suite_name}", log_file)
    log_message(f"Task range: [{task_start}, {task_end})", log_file)
    log_message(f"Trials per task: {cfg.num_trials_per_task}", log_file)

    total_episodes, total_successes = 0, 0

    for task_id in tqdm.tqdm(range(task_start, task_end), desc="Tasks"):
        task = task_suite.get_task(task_id)
        initial_states = task_suite.get_task_init_states(task_id)
        env, task_description = get_libero_env(task, cfg.model_family, resolution=cfg.env_img_res)

        task_episodes, task_successes = 0, 0

        for episode_idx in tqdm.tqdm(range(cfg.num_trials_per_task), desc=f"Task {task_id}", leave=False):
            initial_state = initial_states[episode_idx]

            log_message(f"\nTask {task_id}: {task_description}, Episode {episode_idx}", log_file)

            success, replay_images, episode_data = run_episode_collect(
                cfg,
                env,
                task_description,
                model,
                resize_size,
                processor,
                action_head,
                proprio_projector,
                noisy_action_projector,
                initial_state,
                log_file,
            )

            # Add episode metadata
            episode_data["task_id"] = task_id
            episode_data["episode_idx"] = episode_idx

            # Save episode data
            save_path = os.path.join(
                collect_suite_dir,
                f"task{task_id:02d}_ep{episode_idx:03d}.pkl",
            )
            with open(save_path, "wb") as f:
                pickle.dump(episode_data, f)

            # Update counters
            task_episodes += 1
            total_episodes += 1
            if success:
                task_successes += 1
                total_successes += 1

            # Save replay video
            save_rollout_video(
                replay_images, total_episodes, success=success,
                task_description=task_description, log_file=log_file,
            )

            log_message(f"Success: {success} | Chunks collected: {episode_data['num_chunks']}", log_file)
            log_message(
                f"Running: {total_successes}/{total_episodes} "
                f"({total_successes / total_episodes * 100:.1f}%)",
                log_file,
            )

        task_rate = task_successes / task_episodes if task_episodes > 0 else 0
        log_message(f"Task {task_id} success rate: {task_rate:.3f}", log_file)

    final_rate = total_successes / total_episodes if total_episodes > 0 else 0
    log_message(f"\nFinal success rate: {final_rate:.4f} ({final_rate * 100:.1f}%)", log_file)
    log_message(f"Data saved to: {collect_suite_dir}", log_file)
    log_file.close()


if __name__ == "__main__":
    collect_libero()
