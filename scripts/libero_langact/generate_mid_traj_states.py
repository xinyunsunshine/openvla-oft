"""
Generate mid-trajectory sim states by rolling out a nonum checkpoint with task prompts.

Runs the nonum model using the original LIBERO task description (e.g. "open the middle
drawer") for N action chunks, then saves the MuJoCo sim state at each checkpoint.
These saved states can later be loaded by test_steerability.py via --precomputed_states_dir.

Usage:
    cd /proj/inf-scaling/zwhong/idan/vla-rl/openvla-oft

    python scripts/libero_langact/generate_mid_traj_states.py \
        --pretrained_checkpoint /proj/inf-scaling/zwhong/idan/vla-rl/openvla-oft/runs/openvla-7b+libero_goal_no_noops_langact_nonum+b8+lr-0.0005+lora-r32+dropout-0.0--image_aug--langact_verbose--libero_goal--8_acts_chunk--L1_regression--3rd_person_img--wrist_img--proprio_state--150000_chkpt \
        --task_suite_name libero_goal \
        --num_tasks 10 \
        --mid_rollout_chunks 3 6 \
        --output_dir ./mid_traj_states/libero_goal_nonum150k
"""

import argparse
import os
import sys

import imageio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "experiments", "robot"))

from experiments.robot.libero.libero_utils import (
    get_libero_dummy_action,
    get_libero_env,
    get_libero_image,
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

from test_steerability import (
    SteerConfig,
    check_unnorm_key,
    get_libero_wrist_image,
    prepare_observation,
    process_action,
    reset_env_to_initial,
    save_video,
)


def main():
    parser = argparse.ArgumentParser(description="Generate mid-trajectory sim states using a nonum checkpoint.")
    parser.add_argument("--pretrained_checkpoint", type=str, required=True,
                        help="Path to nonum checkpoint that follows task descriptions well.")
    parser.add_argument("--task_suite_name", type=str, default="libero_goal",
                        choices=["libero_spatial", "libero_object", "libero_goal", "libero_10"])
    parser.add_argument("--num_tasks", type=int, default=10,
                        help="Number of tasks (0 to N-1). Ignored if --task_ids is set.")
    parser.add_argument("--task_ids", type=int, nargs="+", default=None,
                        help="Specific task IDs to generate states for.")
    parser.add_argument("--mid_rollout_chunks", type=int, nargs="+", default=[3, 6],
                        help="Save sim states after these many chunks.")
    parser.add_argument("--output_dir", type=str, default="./mid_traj_states/libero_goal_nonum150k")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--lora_rank", type=int, default=32)
    parser.add_argument("--unnorm_key", type=str, default="")
    args = parser.parse_args()

    cfg = SteerConfig(
        pretrained_checkpoint=args.pretrained_checkpoint,
        task_suite_name=args.task_suite_name,
        unnorm_key=args.unnorm_key,
        seed=args.seed,
        lora_rank=args.lora_rank,
    )
    set_seed_everywhere(cfg.seed)

    print("Loading model...")
    model = get_model(cfg)
    proprio_projector = get_proprio_projector(cfg, model.llm_dim, proprio_dim=8) if cfg.use_proprio else None
    action_head = get_action_head(cfg, model.llm_dim) if (cfg.use_l1_regression or cfg.use_diffusion) else None
    noisy_action_projector = get_noisy_action_projector(cfg, model.llm_dim) if cfg.use_diffusion else None
    processor = get_processor(cfg)
    check_unnorm_key(cfg, model)
    resize_size = get_image_resize_size(cfg)

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_name]()
    if args.task_ids is not None:
        task_ids = [t for t in args.task_ids if t < task_suite.n_tasks]
    else:
        task_ids = list(range(min(args.num_tasks, task_suite.n_tasks)))

    max_chunks = max(args.mid_rollout_chunks)
    os.makedirs(args.output_dir, exist_ok=True)

    for task_id in task_ids:
        task = task_suite.get_task(task_id)
        env, task_description = get_libero_env(task, cfg.model_family, resolution=cfg.env_img_res)
        task_label = task_description.replace(" ", "_").replace("/", "_")

        print(f"\n{'='*70}")
        print(f"Task {task_id}: {task_description}")
        print(f"{'='*70}")

        initial_states = task_suite.get_task_init_states(task_id)
        initial_state = initial_states[0]

        # Reset and run forward with task description
        obs = reset_env_to_initial(env, initial_state, cfg)
        all_images = [get_libero_image(obs)]

        # Also save initial state
        task_dir = os.path.join(args.output_dir, f"task_{task_id}_{task_label}")
        os.makedirs(task_dir, exist_ok=True)

        init_sim_state = env.sim.get_state().flatten()
        np.save(os.path.join(task_dir, "initial_sim_state.npy"), init_sim_state)
        plt.imsave(os.path.join(task_dir, "initial_snapshot.png"), get_libero_image(obs))
        print(f"  Saved initial state")

        for chunk_idx in range(max_chunks):
            observation, img = prepare_observation(obs, resize_size)

            with torch.no_grad():
                actions = get_action(
                    cfg=cfg, model=model, obs=observation, task_label=task_description,
                    processor=processor, action_head=action_head,
                    proprio_projector=proprio_projector,
                    noisy_action_projector=noisy_action_projector,
                    use_film=cfg.use_film,
                )

            for action in actions:
                action = process_action(action)
                obs, _, _, _ = env.step(action.tolist())
                all_images.append(get_libero_image(obs))

            current_chunk = chunk_idx + 1
            eef_pos = obs["robot0_eef_pos"]
            print(f"  Chunk {current_chunk}/{max_chunks} done, EEF: [{eef_pos[0]:.3f}, {eef_pos[1]:.3f}, {eef_pos[2]:.3f}]")

            if current_chunk in args.mid_rollout_chunks:
                sim_state = env.sim.get_state().flatten()
                np.save(os.path.join(task_dir, f"sim_state_chunk_{current_chunk}.npy"), sim_state)
                plt.imsave(os.path.join(task_dir, f"snapshot_chunk_{current_chunk}.png"), get_libero_image(obs))
                save_video(all_images, os.path.join(task_dir, f"advance_rollout_chunk_{current_chunk}.mp4"))
                print(f"  -> Saved sim state & snapshot at chunk {current_chunk}")

        env.close()

    # Save metadata
    import json
    meta = {
        "checkpoint": args.pretrained_checkpoint,
        "task_suite": args.task_suite_name,
        "task_ids": task_ids,
        "mid_rollout_chunks": args.mid_rollout_chunks,
        "seed": args.seed,
        "num_open_loop_steps": cfg.num_open_loop_steps,
    }
    with open(os.path.join(args.output_dir, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nDone! States saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
