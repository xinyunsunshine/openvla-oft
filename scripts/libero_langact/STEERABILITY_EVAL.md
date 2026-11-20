# Evaluating Language Steerability of OpenVLA-OFT with Action-Language

## Overview

After fine-tuning OpenVLA-OFT with language-action prompts (e.g., "move forward 3.0 cm, tilt right 10 degrees, close gripper"), we evaluate whether the model is **steerable** — i.e., does it produce different actions when given different language-action prompts from the same visual state?

There are three evaluation tools:

| Script | Purpose |
|--------|---------|
| `generate_mid_traj_states.py` | Generate mid-trajectory sim states using a nonum checkpoint |
| `test_steerability.py` | Run full prompt battery and save before/after images, videos, grids |
| `check_action_directions.py` | Verify raw action signs match prompted directions (numerical) |
| `eval_combined_steerability.sh` | Batch eval script for combined (all-task) models |

## Quick Start

### Single-suite model (e.g. libero_goal only)

```bash
cd /proj/inf-scaling/zwhong/idan/vla-rl/openvla-oft

# Step 1: Generate mid-trajectory states (one-time, using nonum checkpoint)
python scripts/libero_langact/generate_mid_traj_states.py \
    --pretrained_checkpoint runs/<nonum_checkpoint> \
    --task_suite_name libero_goal \
    --num_tasks 10 \
    --mid_rollout_chunks 3 6 \
    --output_dir ./mid_traj_states/libero_goal_nonum150k

# Step 2a: Visual steerability test (with precomputed mid-traj states)
python scripts/libero_langact/test_steerability.py \
    --pretrained_checkpoint /path/to/verbose/checkpoint \
    --prompt_mode verbose \
    --precomputed_states_dir ./mid_traj_states/libero_goal_nonum150k \
    --task_suite_name libero_goal \
    --num_tasks 3 \
    --output_dir ./steerability_verbose

# Step 2b: Numerical direction check (with precomputed mid-traj states)
python scripts/libero_langact/check_action_directions.py \
    --pretrained_checkpoint /path/to/verbose/checkpoint \
    --precomputed_states_dir ./mid_traj_states/libero_goal_nonum150k \
    --output_dir ./direction_check_results/verbose_150k_task0

# Or without precomputed states (initial state only)
python scripts/libero_langact/check_action_directions.py \
    --pretrained_checkpoint /path/to/verbose/checkpoint
```

### Combined all-task models

For models trained on `libero_combined_langact_*` (all 4 suites, 40 tasks), use the batch eval script:

```bash
# Run on a GPU node
bash /proj/inf-scaling/zwhong/idan/vla-rl/bash_scripts/260303/eval_combined_steerability.sh
```

This script:
1. Generates mid-traj states for all 4 suites using the nonum_5traj_40k checkpoint
2. Runs direction checks (task 0 per suite, initial + chunk 3 + chunk 6)
3. Runs visual steerability tests (task 0 per suite, full prompt battery)
4. Prints a summary of all results

Checkpoints are configured in the `CKPTS` array. Currently evaluates:

| Short name | Dataset | Prompt mode | Unnorm key |
|---|---|---|---|
| `nonum_1traj_10k` | `libero_combined_langact_nonum_v2` (1 traj/task) | nonum | `libero_combined_langact_nonum_v2` |
| `nonum_5traj_40k` | `libero_combined_langact_nonum_v2` (5 traj/task) | nonum | `libero_combined_langact_nonum_v2` |
| `verbose_noisy_15steps_10k` | `libero_combined_langact_verbose_noisy_v2` (5 traj/task) | verbose | `libero_combined_langact_verbose_noisy_v2` |
| `verbose_noisy_1traj_2k` | `libero_combined_langact_verbose_noisy_v2` (1 traj/task) | verbose | `libero_combined_langact_verbose_noisy_v2` |

All steps skip if results already exist, so the script is safe to re-run.

**Important for combined models**: You must pass `--unnorm_key` explicitly because the norm stats key (`libero_combined_langact_*`) doesn't match any single `--task_suite_name`. The batch script handles this automatically.

**Checkpoint preparation**: If a checkpoint is missing `config.json` and model shards (only has LoRA adapter), symlink from a working checkpoint trained on the same base model:

```bash
CKPT="runs/<incomplete_checkpoint>"
BASE="runs/<complete_checkpoint_same_base>"

for f in config.json generation_config.json model.safetensors.index.json \
         model-00001-of-00004.safetensors model-00002-of-00004.safetensors \
         model-00003-of-00004.safetensors model-00004-of-00004.safetensors; do
    ln -sf "$BASE/$f" "$CKPT/$f"
done
```

## Generating Mid-Trajectory States

`generate_mid_traj_states.py` runs a **nonum checkpoint** (which follows LIBERO task descriptions well) to advance the robot to mid-trajectory positions, then saves the MuJoCo sim states to disk. These saved states can be loaded by both `test_steerability.py` and `check_action_directions.py`.

This decouples the "advance to a mid-traj state" step from the "evaluate steerability" step, so you can:
- Generate states once, test many checkpoints against them
- Use a nonum checkpoint for advancement (which follows task descriptions) while testing a verbose checkpoint for steerability

### Output Structure

```
mid_traj_states/libero_goal_nonum150k/
  metadata.json                           # Checkpoint, seed, task info
  task_0_open_the_middle_drawer/
    initial_sim_state.npy                 # MuJoCo sim state at t=0
    initial_snapshot.png                  # Visual snapshot
    sim_state_chunk_3.npy                 # Sim state after 3 chunks (24 steps)
    snapshot_chunk_3.png                  # Visual snapshot
    advance_rollout_chunk_3.mp4           # Video of advancement
    sim_state_chunk_6.npy                 # Sim state after 6 chunks (48 steps)
    snapshot_chunk_6.png
    advance_rollout_chunk_6.mp4
  task_1_.../
    ...
```

### Command-Line Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--pretrained_checkpoint` | (required) | Path to nonum checkpoint |
| `--task_suite_name` | `libero_goal` | LIBERO suite |
| `--num_tasks` | 10 | Number of tasks |
| `--task_ids` | None | Specific task IDs |
| `--mid_rollout_chunks` | `3 6` | Save states at these chunk counts |
| `--output_dir` | `./mid_traj_states/libero_goal_nonum150k` | Output directory |
| `--seed` | 7 | Random seed |

## Visual Steerability Test (`test_steerability.py`)

Runs a battery of language-action prompts from fixed states, executes the predicted 8-step action chunk, and saves before/after images, rollout videos, and comparison grids.

### Prompt Modes

#### `--prompt_mode verbose` (47 prompts)

For models trained on `langact_verbose` datasets. Includes:

| Category | Count | Examples |
|----------|-------|---------|
| Single-axis translation | 6 | "move forward 5.0 cm, open gripper" |
| Single-axis rotation | 6 | "tilt right 20 degrees, open gripper" |
| Gripper | 2 | "close gripper", "open gripper" |
| Magnitude sweep (forward) | 7 | "move forward 0.2/0.5/1.0/2.0/4.0/6.0/8.0 cm, open gripper" |
| Magnitude sweep (left) | 4 | "move left 0.5/2.0/4.0/8.0 cm, open gripper" |
| Magnitude sweep (up) | 4 | "move up 0.5/2.0/4.0/8.0 cm, open gripper" |
| Magnitude sweep (rotation) | 2 | "tilt right 10/20 degrees, open gripper" |
| Multi-axis combos | 6 | "move forward 3.0 cm, move up 2.0 cm, close gripper" |
| Data-sampled combos | 3 | Full 5-6 axis prompts from training data |
| Opposite-direction pairs | 4 | "move forward 3.0 cm, move right 2.0 cm" vs "move back 3.0 cm, move left 2.0 cm" |

#### `--prompt_mode nonum` (41 prompts)

For models trained on `langact_nonum` datasets. Includes:

| Category | Count | Examples |
|----------|-------|---------|
| Single-axis translation | 6 | "move forward, open gripper" |
| Single-axis rotation | 6 | "tilt right, open gripper", "tilt back, open gripper" |
| Gripper | 2 | "close gripper", "open gripper" |
| 2-axis combos | 6 | "move forward, move up, close gripper" |
| Data-sampled 3-6 motion | 14 | Frequent prompts from training data |
| Opposite-direction pairs | 6 | Translation-flipped versions of top-frequency prompts |

#### `--prompt_mode all` (88 prompts)

Both verbose and nonum combined (suffixed `_v` and `_nn`).

### Mid-Rollout Testing

By default, the script tests steerability at three states per task:

1. **Initial state** — robot's starting position
2. **After 3 action chunks** (24 steps) — robot partway through a trajectory
3. **After 6 action chunks** (48 steps) — robot further along

There are three ways to reach mid-rollout states:

#### 1. Precomputed States (Recommended)

Load states generated by `generate_mid_traj_states.py`:

```bash
python scripts/libero_langact/test_steerability.py \
    --pretrained_checkpoint /path/to/checkpoint \
    --precomputed_states_dir ./mid_traj_states/libero_goal_nonum150k \
    ...
```

Falls back to auto-langact if a state file is not found for a particular task/chunk.

#### 2. Auto Language-Action Advancement (Default without precomputed)

Advances the robot using auto-computed language-action prompts:
- Chunk 0: uses `--initial_advance_prompt` (default: "move forward 3.0 cm, move down 2.0 cm, close gripper")
- Chunk 1+: computes the actual EEF state change from the previous chunk, converts it to verbose language-action text, and uses that as the next prompt

This keeps all prompts in-distribution for verbose models.

#### 3. Legacy Fixed-Prompt Mode

```bash
python scripts/libero_langact/test_steerability.py \
    --pretrained_checkpoint /path/to/checkpoint \
    --task_prompt_override "move forward 3.0 cm, close gripper" \
    ...
```

### Command-Line Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--pretrained_checkpoint` | (required) | Path to trained checkpoint directory |
| `--task_suite_name` | `libero_goal` | LIBERO suite: `libero_spatial`, `libero_object`, `libero_goal`, `libero_10` |
| `--num_tasks` | 3 | Number of tasks to test (0 to N-1). Ignored if `--task_ids` set |
| `--task_ids` | None | Specific task IDs, e.g., `--task_ids 0 3 7` |
| `--prompt_mode` | `nonum` | Prompt set: `nonum`, `verbose`, `all` |
| `--prompts` | None | Subset of prompt keys (overrides `--prompt_mode`) |
| `--mid_rollout_chunks` | `3 6` | Advance N chunks before mid-rollout test |
| `--precomputed_states_dir` | None | Path to saved sim states from `generate_mid_traj_states.py` |
| `--initial_advance_prompt` | "move forward 3.0 cm, move down 2.0 cm, close gripper" | First chunk prompt for auto-langact advancement |
| `--task_prompt_override` | None | Fixed prompt for all advance chunks (disables auto-langact) |
| `--unnorm_key` | (auto) | Action un-normalization key |
| `--seed` | 7 | Random seed |
| `--lora_rank` | 32 | LoRA rank (must match training) |

### Output Structure

```
output_dir/
  task_0_open_the_middle_drawer/
    initial/
      grid.png                              # Comparison grid for all prompts
      move_forward/
        before.png                          # Observation before action chunk
        after.png                           # Observation after 8 steps
        rollout.mp4                         # 9-frame video (before + 8 steps)
      move_back/
        before.png / after.png / rollout.mp4
      ...
    after_chunk_3/
      advance_prompts.txt                   # How the mid-traj state was reached
      grid.png
      move_forward/
        before.png / after.png / rollout.mp4
      ...
    after_chunk_6/
      ...
  task_1_.../
    ...
```

## Directional Accuracy Check (`check_action_directions.py`)

Verifies whether the model's **raw predicted actions** have the correct sign for each prompted direction. This is more precise than pixel-diff analysis — it checks e.g. "does 'tilt left' actually produce positive droll?"

### Action Vector Convention

```
[0] dx      + = forward,    - = back
[1] dy      + = left,       - = right
[2] dz      + = up,         - = down
[3] droll   + = tilt left,  - = tilt right
[4] dpitch  + = tilt back,  - = tilt forward
[5] dyaw    + = rotate CCW, - = rotate CW
[6] gripper >= 0.5 = open,  < 0.5 = close
```

### Prompt Modes (`--prompt_mode`)

#### `verbose` (default, 29 prompts) — for models trained on `langact_verbose` data

| Category | Count | Example prompt | Expected conditions |
|----------|-------|---------------|-------------------|
| Single-axis translation | 6 | "move forward 5.0 cm, open gripper" | `dx > 0` |
| Single-axis rotation | 6 | "tilt left 20 degrees, open gripper" | `droll > 0` |
| Gripper | 2 | "close gripper" | `grip < 0.5` |
| Translation combos | 5 | "move forward 3.0 cm, move up 2.0 cm, close gripper" | `dx > 0, dz > 0` |
| Translation + rotation combos | 5 | "move up 3.0 cm, tilt right 10 degrees, open gripper" | `dz > 0, droll < 0` |
| Rotation combos | 3 | "tilt left 10 degrees, tilt back 10 degrees, open gripper" | `droll > 0, dpitch > 0` |
| Full 6-axis combos | 2 | All 6 motion dims + gripper | All 6 dims checked |

#### `nonum` (29 prompts) — for models trained on `langact_nonum` data

Same structure as verbose but without magnitudes:

| Category | Count | Example prompt | Expected conditions |
|----------|-------|---------------|-------------------|
| Single-axis translation | 6 | "move forward, open gripper" | `dx > 0` |
| Single-axis rotation | 6 | "tilt left, open gripper" | `droll > 0` |
| Gripper | 2 | "close gripper" | `grip < 0.5` |
| Translation combos | 5 | "move forward, move up, close gripper" | `dx > 0, dz > 0` |
| Translation + rotation combos | 5 | "move up, tilt right, open gripper" | `dz > 0, droll < 0` |
| Rotation combos | 3 | "tilt left, tilt back, open gripper" | `droll > 0, dpitch > 0` |
| Full 6-axis combos | 2 | All 6 motion dims + gripper | All 6 dims checked |

### Output

Saves two files to `--output_dir`:

- **`results.json`** — structured data with per-prompt summed actions, per-condition pass/fail, per-dimension accuracy
- **`log.txt`** — full text log (same as terminal output)

Example output:

```
  up_tilt_right                       [OK]
    prompt:  move up 3.0 cm, tilt right 10 degrees, open gripper
    checks:  dz>0:+2.199[ok], droll<0:-0.248[ok]
    all_dims: dx=-0.940 dy=-1.383 dz=+2.199 droll=-0.248 dpitch=+0.408 dyaw=+0.082 grip=+6.699

  Prompt-level accuracy:    17/29 (59%)
  Condition-level accuracy: 39/55 (71%)

  Per-dimension accuracy:
        dx: 9/10 (90%)
        dy: 6/9 (67%)
        dz: 10/10 (100%)
     droll: 5/10 (50%)
    dpitch: 4/7 (57%)
      dyaw: 4/7 (57%)
      grip: 1/2 (50%)
```

### Command-Line Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--pretrained_checkpoint` | (required) | Path to checkpoint |
| `--prompt_mode` | `verbose` | `verbose` (with magnitudes) or `nonum` (directions only) |
| `--task_suite_name` | `libero_goal` | LIBERO suite |
| `--task_id` | 0 | Single task ID to test |
| `--precomputed_states_dir` | None | Path to saved sim states (tests at chunk 3/6 too) |
| `--mid_rollout_chunks` | `3 6` | Which chunk states to test |
| `--output_dir` | (auto) | Output directory for results.json + log.txt |
| `--seed` | 7 | Random seed |
| `--lora_rank` | 32 | LoRA rank |

## How to Analyze Results

### Visual Inspection (from `test_steerability.py`)

1. **grid.png**: Each row shows before image, after image, and prompt text with EEF delta. Look for:
   - Do opposite prompts (move_left vs move_right) produce visibly different after-images?
   - Do magnitude sweeps (0.5 cm vs 8.0 cm) show proportionally different movement?
   - Does the gripper open/close as instructed?

2. **rollout.mp4**: Watch the 8-step execution. Does the robot move in the prompted direction?

### Pixel-Diff Analysis (from `test_steerability.py`)

```python
import os, numpy as np
from PIL import Image

base = "steerability_results/task_0_.../initial"

# Compute before→after pixel diff for each prompt
for prompt_dir in sorted(os.listdir(base)):
    if not os.path.isdir(os.path.join(base, prompt_dir)):
        continue
    before = np.array(Image.open(f"{base}/{prompt_dir}/before.png")).astype(float)
    after = np.array(Image.open(f"{base}/{prompt_dir}/after.png")).astype(float)
    diff = np.mean(np.abs(after - before))
    print(f"{prompt_dir:35s} | {diff:.2f}")
```

### Key Metrics

1. **Coefficient of Variation (CV)**: `std(diffs) / mean(diffs)` across all prompts. Higher CV = more diverse responses = better steerability. Target: CV > 0.3.

2. **Opposite-pair differentiation**: Pixel diff between after-images of opposite prompts (e.g., move_left vs move_right). Should be significantly larger than individual before→after diffs.

3. **Magnitude sensitivity** (verbose only): Do pixel diffs increase monotonically with magnitude? Plot diff vs. magnitude for the sweep prompts.

4. **Per-dimension directional accuracy** (from `check_action_directions.py`): What percentage of conditions are correct per action dimension? Reveals which axes the model controls vs ignores.

## Key Findings

### Per-Dimension Directional Accuracy (verbose 150k, task 0)

| Dimension | Initial | Chunk 3 | Chunk 6 | Verdict |
|-----------|:---:|:---:|:---:|---------|
| dx (fwd/back) | 90% | 90% | 100% | Strong |
| dy (left/right) | 67% | 78% | 100% | Strong |
| dz (up/down) | 100% | 100% | 90% | Strong |
| droll (tilt L/R) | 50% | 50% | 40% | Broken |
| dpitch (tilt B/F) | 57% | 57% | 43% | Weak |
| dyaw (rotate) | 57% | 43% | 43% | Broken |
| grip (open/close) | 50% | 50% | 50% | Broken |

**Translation is reliably steerable. Rotation and gripper are not.**

The model outputs a fixed default rotation pattern regardless of the rotation prompt — e.g., "tilt left" and "tilt right" produce nearly identical droll values (+0.23 vs +0.26). Training longer (50k vs 150k) does not improve this.

### Pixel-Diff Steerability (verbose 150k, with precomputed nonum mid-traj states)

| State | CV | Mean diff | Magnitude corr (fwd sweep) |
|-------|:---:|:---:|:---:|
| Initial | 0.39 | 1.12 | 1.00 (monotonic) |
| Chunk 3 | 0.39 | 2.01 | 1.00 (monotonic) |
| Chunk 6 | 0.26 | 2.92 | 0.98 |

CV > 0.3 at initial and chunk 3. Magnitude sensitivity is near-perfect (monotonic forward sweep). CV drops at chunk 6 due to larger baseline motion masking prompt differences.

### Opposite-Pair Differentiation

| Pair | Initial | Chunk 3 | Chunk 6 |
|------|:---:|:---:|:---:|
| up vs down | 4.25-4.35 | 5.32-6.39 | 5.78-6.90 |
| fwd vs back | 3.31-3.51 | 4.34-5.22 | 5.63-6.96 |
| left vs right | 2.94-3.27 | 4.48-5.57 | 5.64-6.77 |
| tilt L vs R | 3.13-3.55 | 3.09-3.10 | 1.41-1.59 |

Translation opposites get stronger at mid-rollout. Rotation (tilt L/R) degrades significantly at chunk 6.

### 50k vs 150k Checkpoint Comparison

No meaningful difference — both achieve ~60% prompt-level accuracy and ~71% condition-level accuracy with the same per-dimension breakdown. The rotation failure is structural, not a matter of more training.

## Checkpoint Preparation

If your checkpoint doesn't have `config.json` (merged LoRA failed during training), symlink the base model files:

```bash
CKPT="/path/to/your/checkpoint--XXXXX_chkpt"
BASE="/path/to/openvla-7b"

for f in config.json generation_config.json model.safetensors.index.json \
         model-00001-of-00003.safetensors model-00002-of-00003.safetensors \
         model-00003-of-00003.safetensors; do
    ln -sf "$BASE/$f" "$CKPT/$f"
done
```

## Resume Training from Checkpoint

```bash
torchrun --standalone --nnodes 1 --nproc-per-node 8 vla-scripts/finetune.py \
    --vla_path /path/to/checkpoint--XXXXX_chkpt \
    --resume True \
    --resume_step XXXXX \
    --data_root_dir /proj/inf-scaling/SWE_sim/modified_libero_rlds_new \
    --dataset_name libero_goal_no_noops_langact_verbose \
    ... (same hyperparameters as original run)
```
