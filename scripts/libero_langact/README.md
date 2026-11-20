# Generate LIBERO RLDS Datasets with Language-Action Prompts

## Context

We want to replace the task-description prompts in LIBERO RLDS datasets (e.g., "pick up the red cup") with per-step **language-action** text that describes the robot's motion (e.g., "move forward 3.0 cm, move up 1.0 cm, rotate clockwise 15 degrees, close gripper"). This lets us SFT OpenVLA-OFT using language-action prompts instead of task descriptions.

We generate **two variants** per dataset:
- **Verbose**: "move forward 3.0 cm, move up 1.0 cm, close gripper" (with numbers + units)
- **No-numbers**: "move forward, move up, close gripper" (direction only)

**Temporal horizon**: Sum deltas over next 8 steps (matching OpenVLA-OFT's `NUM_ACTIONS_CHUNK=8`).
**Rotation**: Included (6DoF + gripper).
**No normalization mismatch**: We compute language-actions from raw RLDS data before any normalization.

## Language-Action Computation (per step)

```
Raw state[t]:  [eef_x, eef_y, eef_z, ax_ang_x, ax_ang_y, ax_ang_z, grip_qpos_l, grip_qpos_r]
                ─────── pos ────────   ──────── axis-angle ─────────   ──── gripper ────────

Step 1: euler[t] = Rotation.from_rotvec(axis_angle[t]).as_euler('xyz')
Step 2: delta_pos[t] = pos[t+1] - pos[t]
Step 3: delta_euler[t] = euler_diff(euler[t+1], euler[t])  # via rotation matrices
Step 4: gripper[t] = 1 - clip(action[t, -1], 0, 1)        # invert: 1=open, 0=close
Step 5: lang_act[t] = [delta_pos(3), delta_euler(3), gripper(1)]  # 7D
Step 6: summed = sum(lang_act[t:t+8])                      # 8-step aggregation
         gripper = lang_act[t+7, 6]                         # last step's gripper
Step 7: text = summarize_numeric_actions(summed, "1f", include_rotation=True)
```

## Scripts

### 1. `generate_langact_datasets.py` — Per-Suite Clean Datasets

Generates clean (no noise) language-action datasets for each LIBERO suite.

```bash
/proj/inf-scaling/zwhong/idan/vla-rl/lap/.venv/bin/python \
    scripts/libero_langact/generate_langact_datasets.py \
    --data_dir /proj/inf-scaling/SWE_sim/modified_libero_rlds_new \
    --output_dir /proj/inf-scaling/SWE_sim/modified_libero_rlds_new \
    --datasets libero_spatial_no_noops libero_object_no_noops libero_goal_no_noops libero_10_no_noops
```

### 2. `generate_combined_noisy.py` — Combined Multi-Task Noisy Dataset

Combines all 4 suites (40 tasks), samples N trajectories per task, and adds text-level Gaussian noise to the verbose prompts.

```bash
# 5 trajectories per task (200 total), with noise
/proj/inf-scaling/zwhong/idan/vla-rl/lap/.venv/bin/python \
    scripts/libero_langact/generate_combined_noisy.py \
    --data_dir /proj/inf-scaling/SWE_sim/modified_libero_rlds_new \
    --output_dir /proj/inf-scaling/SWE_sim/modified_libero_rlds_new \
    --output_name libero_combined_langact_verbose_noisy_v2 \
    --trajs_per_task 5 --seed 42

# 1 trajectory per task (40 total), with noise
/proj/inf-scaling/zwhong/idan/vla-rl/lap/.venv/bin/python \
    scripts/libero_langact/generate_combined_noisy.py \
    --data_dir /proj/inf-scaling/SWE_sim/modified_libero_rlds_new \
    --output_dir /proj/inf-scaling/SWE_sim/modified_libero_rlds_new \
    --output_name libero_combined_langact_verbose_noisy_v2_1traj \
    --trajs_per_task 1 --seed 42

# Without noise (for comparison)
/proj/inf-scaling/zwhong/idan/vla-rl/lap/.venv/bin/python \
    scripts/libero_langact/generate_combined_noisy.py \
    --data_dir /proj/inf-scaling/SWE_sim/modified_libero_rlds_new \
    --output_dir /proj/inf-scaling/SWE_sim/modified_libero_rlds_new \
    --output_name libero_combined_langact_verbose_clean \
    --trajs_per_task 5 --no_noise --seed 42
```

### 3. `visualize_langact.py` — Visualize Datasets

```bash
# Show before/after images with prompt and actual EEF delta
/proj/inf-scaling/zwhong/idan/vla-rl/lap/.venv/bin/python \
    scripts/libero_langact/visualize_langact.py \
    --data_dir /proj/inf-scaling/SWE_sim/modified_libero_rlds_new \
    --dataset libero_combined_langact_verbose_noisy_v2 \
    --num_episodes 1 --steps_per_episode 4 \
    --output viz.png

# Compare noisy prompts against clean verbose (3-column: original, noisy prompt, actual delta)
/proj/inf-scaling/zwhong/idan/vla-rl/lap/.venv/bin/python \
    scripts/libero_langact/visualize_langact.py \
    --data_dir /proj/inf-scaling/SWE_sim/modified_libero_rlds_new \
    --dataset libero_combined_langact_verbose_noisy_v2 \
    --original_dataset libero_goal_no_noops_langact_verbose \
    --num_episodes 1 --steps_per_episode 3 \
    --output viz_compare.png
```

## Noise Design (v2, Text-Level)

Noise is applied **at the text level** after converting the 7D language-action vector to verbose text. This ensures all numeric perturbations are visible in the final prompt (unlike v1 which applied noise to the raw vector, where rounding often hid the noise).

### How It Works

For each numeric value in the text string:
1. **30% chance**: keep the value unchanged (the model still sees exact ground-truth sometimes)
2. **70% chance**: add Gaussian noise `N(0, std)` where:
   - **Translation**: `std = max(0.2 cm, 10% * |value|)` — floor of 0.2 cm ensures even small values (0.1-0.5 cm) get visible noise
   - **Rotation**: `std = max(3 degrees, 10% * |value|)` — floor of 3 degrees allows crossing the 10/20 degree boundary
3. **Gripper** ("open gripper" / "close gripper"): never changed

### Examples

```
Clean:  "move forward 5.0 cm, move up 2.0 cm, tilt right 10 degrees, close gripper"
Noisy:  "move forward 5.4 cm, move up 2.0 cm, tilt right 13 degrees, close gripper"
                       ^^^                                   ^^
                     changed (10% of 5.0 = 0.5 std)       changed (floor 3 deg std)
                                     kept (30% chance)

Clean:  "move forward 0.3 cm, open gripper"
Noisy:  "move forward 0.1 cm, open gripper"
                       ^^^
                     changed (floor 0.2 cm std on small value)
```

### Why Text-Level Noise?

The v1 approach added noise to the raw 7D vector (in meters/radians) before text conversion. This failed because:
- **Translation rounding**: Text uses 0.1 cm precision. For small values like 0.3 cm, 10% noise (0.03 cm) was below the 0.05 cm rounding threshold — invisible ~90% of the time.
- **Rotation rounding**: `_round_to_nearest_n(degrees, 10)` snaps to {0, 10, 20}. Noise of 1-2 degrees never crosses the ±5 degree boundary needed to change the text.

Result: only ~63% of translation values and ~0% of rotation values visibly changed in v1. With v2 text-level noise, all perturbed values are guaranteed to be visible.

### Noise Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--noise_frac` | 0.1 | Proportional noise (10% of value) |
| `--keep_prob` | 0.3 | Probability of keeping value unchanged |
| `--trans_floor_cm` | 0.2 | Minimum translation noise std (cm) |
| `--rot_floor_deg` | 3.0 | Minimum rotation noise std (degrees) |

## Output Dataset Names

### Per-Suite Clean Datasets

| Original | Verbose | No-numbers |
|----------|---------|------------|
| `libero_spatial_no_noops` | `libero_spatial_no_noops_langact_verbose` | `libero_spatial_no_noops_langact_nonum` |
| `libero_object_no_noops` | `libero_object_no_noops_langact_verbose` | `libero_object_no_noops_langact_nonum` |
| `libero_goal_no_noops` | `libero_goal_no_noops_langact_verbose` | `libero_goal_no_noops_langact_nonum` |
| `libero_10_no_noops` | `libero_10_no_noops_langact_verbose` | `libero_10_no_noops_langact_nonum` |

### Combined Multi-Task Datasets

| Name | Tasks | Trajs/Task | Noise |
|------|-------|------------|-------|
| `libero_combined_langact_verbose_noisy` | 40 | 5 | v1 (raw vector, mostly invisible) |
| `libero_combined_langact_verbose_noisy_v2` | 40 | 5 | v2 (text-level, visible) |
| `libero_combined_langact_verbose_noisy_v2_1traj` | 40 | 1 | v2 (text-level, visible) |

## Training

```bash
torchrun --standalone --nnodes 1 --nproc-per-node 8 vla-scripts/finetune.py \
    --data_root_dir /proj/inf-scaling/SWE_sim/modified_libero_rlds_new \
    --dataset_name libero_combined_langact_verbose_noisy_v2 \
    --run_root_dir /proj/inf-scaling/zwhong/idan/vla-rl/openvla-oft/runs \
    --use_l1_regression True \
    --use_diffusion False \
    --use_film False \
    --num_images_in_input 2 \
    --use_proprio True \
    --batch_size 8 \
    --learning_rate 2e-4 \
    --num_steps_before_decay 30000 \
    --max_steps 50005 \
    --save_freq 2500 \
    --save_latest_checkpoint_only False \
    --image_aug True \
    --lora_rank 32 \
    ...
```

Note: For smaller datasets (200 or 40 trajectories), reduce `max_steps` and `learning_rate` to avoid overfitting. See STEERABILITY_EVAL.md for details.
