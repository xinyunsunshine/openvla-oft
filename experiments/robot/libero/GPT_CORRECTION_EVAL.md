# GPT Action Chunk Correction for OpenVLA-OFT

Closed-loop evaluation that uses GPT to correct OpenVLA-OFT's predicted action chunks at inference time. For each episode, the robot runs with the same initial state under multiple modes so rollout videos are directly comparable.

## Overview

At each action-chunk boundary (every 8 sim steps), the system:
1. Queries the OpenVLA-OFT policy for a proposed 8-step action chunk
2. Sends the current observation image, previous observation image, task description, and proposed actions to GPT
3. GPT reasons per degree of freedom (forward/back, left/right, up/down, rotations, gripper), evaluates the proposed actions, and outputs corrected actions
4. The corrected actions are executed in the environment

### Correction Modes

| Mode | Description |
|------|-------------|
| `openvla_only` | Baseline — no GPT correction, just run the policy |
| `gpt_normal` | GPT outputs a **full replacement** action chunk |
| `gpt_residual` | GPT outputs a **residual** (delta) added to the proposed chunk: `corrected = proposed + residual` |
| `all` | Run all three modes per episode for comparison |

### GPT Reasoning Structure

GPT follows a structured per-DOF reasoning chain:

1. **Effect of last chunk** — compare previous vs current observation
2. **Plan each degree of freedom** — for each of the 7 dimensions, decide direction and magnitude:
   - Forward (+dx) or Back (-dx)? By how many cm?
   - Left (+dy) or Right (-dy)? By how many cm?
   - Up (+dz) or Down (-dz)? By how many cm?
   - Tilt left (+droll) or right (-droll)? By how many degrees?
   - Tilt back (+dpitch) or forward (-dpitch)? By how many degrees?
   - Rotate CCW (+dyaw) or CW (-dyaw)? By how many degrees?
   - Gripper: Open (1.0) or Close (0.0)?
3. **Evaluate proposed actions** — check sign and magnitude against the plan
4. **Convert & output** — corrected action chunk as JSON

The system prompt includes 5 in-context examples (approach, grasp, lift, transport, place) so GPT understands how natural-language plans map to action values.

## Prerequisites

1. **OpenAI API key** with access to GPT-4o (or whichever model you want to use)
2. **GPU** — the OpenVLA-OFT model runs on GPU
3. **Environment** — openvla-oft dependencies installed (the `.venv-oft` venv or conda env)

## Quick Start

```bash
cd /proj/inf-scaling/zwhong/idan/vla-rl/openvla-oft

# Set your OpenAI API key
export OPENAI_API_KEY='sk-...'

# Run all three modes on libero_goal task 0, 3 episodes, with custom prompt
python experiments/robot/libero/run_libero_eval_with_gpt.py \
    --pretrained_checkpoint /path/to/checkpoint \
    --task_suite_name libero_goal \
    --task_id_start 0 \
    --task_id_end 1 \
    --num_trials_per_task 3 \
    --custom_task_description "put the wine bottle on the plate" \
    --gpt_mode all \
    --gpt_model gpt-4o \
    --video_dir ./rollout_videos
```

## Command-Line Arguments

All arguments from the base `run_libero_eval.py` are inherited, plus GPT-specific ones.

### GPT-Specific Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--gpt_mode` | `all` | Which modes to run: `openvla_only`, `gpt_normal`, `gpt_residual`, or `all` |
| `--gpt_model` | `gpt-4o` | OpenAI model ID |
| `--video_dir` | `./rollout_videos` | Root directory for saving videos and logs |
| `--video_fps` | `30` | FPS for saved videos |

### Model Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--pretrained_checkpoint` | (required) | Path to OpenVLA-OFT checkpoint |
| `--unnorm_key` | `""` | Action un-normalization key (must match the model's `dataset_statistics.json`) |
| `--lora_rank` | `32` | LoRA rank (must match training) |
| `--use_l1_regression` | `True` | Use L1 regression action head |
| `--use_diffusion` | `False` | Use diffusion action head |
| `--num_images_in_input` | `2` | Number of images in VLA input |
| `--use_proprio` | `True` | Include proprioceptive state |

### LIBERO Environment Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--task_suite_name` | `libero_spatial` | LIBERO suite: `libero_spatial`, `libero_object`, `libero_goal`, `libero_10` |
| `--task_id_start` | `0` | First task ID (inclusive) |
| `--task_id_end` | `-1` | Last task ID (exclusive); `-1` = all tasks |
| `--num_trials_per_task` | `50` | Number of episodes per task |
| `--custom_task_description` | `""` | Override task prompt (e.g., `"put the wine bottle on the plate"`) |
| `--seed` | `7` | Random seed |

## Usage Examples

### Quick test — single task, 1 episode, GPT-4o normal correction only

```bash
python experiments/robot/libero/run_libero_eval_with_gpt.py \
    --pretrained_checkpoint runs/my_checkpoint \
    --task_suite_name libero_goal \
    --task_id_start 0 --task_id_end 1 \
    --num_trials_per_task 1 \
    --gpt_mode gpt_normal \
    --gpt_model gpt-4o
```

### Full comparison — all modes on all libero_goal tasks

```bash
python experiments/robot/libero/run_libero_eval_with_gpt.py \
    --pretrained_checkpoint runs/my_checkpoint \
    --task_suite_name libero_goal \
    --num_trials_per_task 10 \
    --gpt_mode all \
    --gpt_model gpt-4o \
    --video_dir ./rollout_videos
```

### Custom task description (prompt steering + GPT correction)

```bash
python experiments/robot/libero/run_libero_eval_with_gpt.py \
    --pretrained_checkpoint runs/my_checkpoint \
    --task_suite_name libero_goal \
    --task_id_start 0 --task_id_end 1 \
    --num_trials_per_task 5 \
    --custom_task_description "put the wine bottle on the plate" \
    --gpt_mode all \
    --gpt_model gpt-4o
```

### Use a cheaper/faster model

```bash
python experiments/robot/libero/run_libero_eval_with_gpt.py \
    --pretrained_checkpoint runs/my_checkpoint \
    --task_suite_name libero_goal \
    --task_id_start 0 --task_id_end 1 \
    --num_trials_per_task 3 \
    --gpt_mode gpt_normal \
    --gpt_model gpt-4o-mini
```

## Output Structure

```
rollout_videos/
  libero_goal/
    YYYYMMDD_HHMMSS__open_the_middle_drawer_of_the_cabinet/
      ep000/
        openvla_only--success=False.mp4       # Baseline rollout video
        gpt_normal--success=True.mp4          # GPT normal correction rollout
        gpt_residual--success=False.mp4       # GPT residual correction rollout
        side_by_side.mp4                      # All modes side-by-side comparison
        gpt_normal--gpt_logs_full.json        # Full GPT logs (reasoning + actions)
        gpt_normal--gpt_reasoning.txt         # Human-readable reasoning only
        gpt_residual--gpt_logs_full.json
        gpt_residual--gpt_reasoning.txt
      ep001/
        ...
    YYYYMMDD_HHMMSS__put_the_bowl_on_the_stove/
      ...

experiments/logs/
  GPT-EVAL-libero_goal-YYYYMMDD_HHMMSS.txt          # Run log
  GPT-EVAL-libero_goal-YYYYMMDD_HHMMSS_results.json  # Success rates JSON
```

### Key Output Files

| File | Contents |
|------|----------|
| `*--success=True/False.mp4` | Per-mode rollout video (filename encodes success) |
| `side_by_side.mp4` | All modes stacked horizontally with colored labels |
| `*--gpt_logs_full.json` | Complete GPT logs: reasoning, raw response, original actions, corrected actions per chunk |
| `*--gpt_reasoning.txt` | GPT's reasoning text per chunk (easy to read) |
| `*_results.json` | Aggregated success rates per mode and per task |

## Aggregating Results

After evaluation, use `aggregate_gpt_results.py` to produce a summary report:

```bash
# Print report to terminal
python experiments/robot/libero/aggregate_gpt_results.py \
    --video_dir ./rollout_videos/libero_goal

# Also save report to file
python experiments/robot/libero/aggregate_gpt_results.py \
    --video_dir ./rollout_videos/libero_goal \
    --save_report
```

This shows:
- **Overall success rates** per mode (openvla_only vs gpt_normal vs gpt_residual)
- **Per-task success rates** broken down by mode
- **GPT reasoning summaries** for each episode
- **Correction statistics** — per-dimension mean/max correction magnitude, % of chunks left unchanged

## Action Convention

The action space is a 7D vector per step. Each action chunk contains 8 steps.

```
[0] dx (delta_x)      → +forward  / -back
[1] dy (delta_y)      → +left     / -right
[2] dz (delta_z)      → +up       / -down
[3] droll (delta_roll)   → +tilt left  / -tilt right
[4] dpitch (delta_pitch) → +tilt back  / -tilt forward
[5] dyaw (delta_yaw)    → +rotate CCW / -rotate CW
[6] gripper             → >= 0.5 = open / < 0.5 = close
```

Position values (dims 0-2) are in **meters per step**. Rotation values (dims 3-5) are in **radians per step**.

| Desired motion | Per-step value |
|----------------|----------------|
| 1 cm over 8 steps | ~0.00125 |
| 3 cm over 8 steps | ~0.00375 |
| 5 cm over 8 steps | ~0.00625 |
| 10° over 8 steps | ~0.022 |
| 20° over 8 steps | ~0.044 |

## Cost Estimate

Each GPT call sends 2 images (previous + current observation) plus the action chunk. With GPT-4o:
- ~1,000-1,500 tokens per call (input images + system prompt)
- A typical episode has ~30-50 action chunks → 30-50 GPT calls
- Running `--gpt_mode all` doubles GPT calls (normal + residual)

For 10 tasks x 10 episodes x ~40 chunks x 2 modes = ~8,000 GPT calls per full eval.

## Troubleshooting

### "OPENAI_API_KEY not set" or authentication error

```bash
export OPENAI_API_KEY='sk-...'
```

### GPT parse errors (falls back to original actions)

Check `*--gpt_logs_full.json` for the raw GPT response. Parse failures are logged as `PARSE_ERROR` in the reasoning field. The system retries up to 3 times before falling back.

### Out of memory

The OpenVLA model needs ~16GB VRAM. If you're running out of memory:
```bash
# Use 8-bit quantization
python experiments/robot/libero/run_libero_eval_with_gpt.py \
    --pretrained_checkpoint runs/my_checkpoint \
    --load_in_8bit True \
    ...
```

### Videos are empty or very short

Make sure `--num_steps_wait` (default 10) is not too large relative to the episode length. The first `num_steps_wait` steps are dummy actions for object stabilization.

## Related Files

| File | Description |
|------|-------------|
| `run_libero_eval_with_gpt.py` | Main evaluation script (this README) |
| `run_libero_eval_gpt_langact.py` | GPT→langact→VLA closed-loop eval (see section below) |
| `aggregate_gpt_results.py` | Aggregate results into a report |
| `run_libero_eval.py` | Base evaluation script (no GPT) |
| `libero_utils.py` | LIBERO environment utilities |
| `../robot_utils.py` | Action processing utilities |

---

## GPT→Langact→VLA Closed-Loop Evaluation

`run_libero_eval_gpt_langact.py` implements a different approach from the action-correction
scripts above. Instead of GPT correcting raw 7D action values, GPT generates **language-action
prompts** (e.g., "move forward 3.0 cm, move down 2.0 cm, close gripper") that are fed to a
langact-finetuned OpenVLA-OFT model as the task instruction.

### Architecture

```
observe → GPT plans next motion → langact prompt → VLA executes 8-step chunk → observe → repeat
```

### Eval Modes

| Mode | Description |
|------|-------------|
| `gpt_langact` | GPT generates langact prompts, VLA executes |
| `openvla_langact` | Auto-computed langact (no GPT) — uses observed motion as next prompt |
| `openvla_only` | Standard VLA with task description (baseline) |
| `all` | Run all three modes per episode for comparison |

### Features

| Feature | Description | Flag |
|---------|-------------|------|
| **Proprioceptive state** | EEF position, euler orientation, gripper state sent as text | always on |
| **In-context learning** | Text-based trajectory examples in system prompt | `--use_icl` (default: True) |
| **Task decomposition** | GPT decomposes task into subtasks before execution | `--use_decomposition` (default: True) |
| **Verification** | GPT sees observed motion vs intended, self-corrects | always on (via history) |
| **Structured output** | JSON with observation, subtask, reasoning, prompt, confidence | always on |

### Quick Start

```bash
cd /proj/inf-scaling/zwhong/idan/vla-rl/openvla-oft

export OPENAI_API_KEY='sk-...'

# GPT-steered langact eval on libero_goal task 0
python experiments/robot/libero/run_libero_eval_gpt_langact.py \
    --pretrained_checkpoint runs/<langact_verbose_checkpoint> \
    --task_suite_name libero_goal \
    --task_id_start 0 --task_id_end 1 \
    --num_trials_per_task 3 \
    --eval_mode gpt_langact \
    --gpt_model gpt-4o \
    --video_dir ./rollout_videos_langact

# Compare all modes
python experiments/robot/libero/run_libero_eval_gpt_langact.py \
    --pretrained_checkpoint runs/<langact_verbose_checkpoint> \
    --task_suite_name libero_goal \
    --task_id_start 0 --task_id_end 1 \
    --num_trials_per_task 5 \
    --eval_mode all \
    --gpt_model gpt-4o

# Disable ICL and decomposition for ablation
python experiments/robot/libero/run_libero_eval_gpt_langact.py \
    --pretrained_checkpoint runs/<langact_verbose_checkpoint> \
    --task_suite_name libero_goal \
    --task_id_start 0 --task_id_end 1 \
    --num_trials_per_task 3 \
    --eval_mode gpt_langact \
    --use_icl False \
    --use_decomposition False
```

### Langact-Specific Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--eval_mode` | `gpt_langact` | `gpt_langact`, `openvla_langact`, `openvla_only`, or `all` |
| `--gpt_model` | `gpt-4o` | OpenAI model ID |
| `--use_icl` | `True` | Include trajectory examples in system prompt |
| `--use_decomposition` | `True` | Ask GPT to decompose task into subtasks first |
| `--video_dir` | `./rollout_videos_langact` | Output directory |
| `--initial_langact_prompt` | `"move forward 3.0 cm, ..."` | First chunk prompt for openvla_langact mode |

### Output Structure

```
rollout_videos_langact/
  libero_goal/
    YYYYMMDD_HHMMSS__open_the_middle_drawer/
      ep000/
        gpt_langact--success=True.mp4
        gpt_langact/
          chunk_000/
            current_obs.png
            gpt_response.json          # Full GPT reasoning + parsed JSON
          chunk_001/
            ...
        gpt_langact--gpt_logs.json     # Full log with all chunks
        gpt_langact--reasoning.txt     # Human-readable reasoning per chunk
        openvla_langact--success=False.mp4
        openvla_only--success=False.mp4
        side_by_side.mp4
```

### How It Differs from GPT Action Correction

| Aspect | GPT Action Correction | GPT→Langact |
|--------|----------------------|-------------|
| GPT output | Raw 7D action values (8×7 array) | Language-action prompt string |
| VLA role | Proposes actions; GPT corrects | Executes langact prompt directly |
| Requires | Standard VLA checkpoint | Langact-finetuned VLA checkpoint |
| Action space | Numeric (GPT must produce valid float arrays) | Natural language (GPT's strength) |
| Verification | Implicit (GPT sees before/after) | Explicit (observed motion vs prompt logged) |
