# GPT → Langact → VLA Closed-Loop Evaluation

Closed-loop evaluation where GPT acts as a high-level planner, generating
language-action prompts that steer a langact-finetuned OpenVLA-OFT model.

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         Episode Loop                                │
│                                                                     │
│   ┌───────────┐    langact prompt     ┌───────────┐    8 actions    │
│   │  GPT-4o   │ ──────────────────►   │  OpenVLA  │ ──────────►  env│
│   │ (planner) │  "move fwd 3.0 cm,    │  (langact │                 │
│   │           │   move down 2.0 cm,   │  finetuned│                 │
│   │           │   close gripper"      │   policy) │                 │
│   └─────▲─────┘                       └───────────┘                 │
│         │                                                           │
│         │  observation image + proprio state + history               │
│         └───────────────────────────────────────────────────────────┘
```

At each action-chunk boundary (every 8 sim steps):
1. **Observe**: capture RGB image + proprioceptive state (EEF pos/orient/gripper)
2. **GPT plans**: given task, observation, proprio, and history of past prompts + outcomes
3. **Generate langact prompt**: e.g., `"move forward 3.0 cm, move down 2.0 cm, close gripper"`
4. **VLA executes**: the langact-finetuned model treats the prompt as its task instruction
5. **Verify**: compute actual EEF motion, add to history for next GPT call

## How It Differs from GPT Action Correction

| Aspect | GPT Action Correction (`run_libero_eval_with_gpt.py`) | GPT → Langact (`run_libero_eval_gpt_langact.py`) |
|--------|-------------------------------------------------------|--------------------------------------------------|
| GPT output | Raw 7D action values (8×7 float array) | Natural language prompt string |
| VLA role | Proposes actions; GPT corrects them | Directly executes the langact prompt |
| Requires | Standard VLA checkpoint | **Langact-finetuned** VLA checkpoint |
| GPT strength | Must produce precise floats | Operates in natural language (GPT's strength) |
| Calibration | Action-unit scale reference | Exact cm/degrees ↔ meters/radians conversion |
| Verification | Implicit (before/after images) | Explicit: observed motion vs commanded motion |

## Eval Modes

| Mode | Description |
|------|-------------|
| `gpt_langact` | Reactive: GPT generates ONE langact prompt per chunk, VLA executes, repeat |
| `gpt_multistep` | Open-loop: GPT generates N prompts at once, VLA executes all, then GPT verifies + corrects |
| `openvla_langact` | Auto-computed langact baseline (no GPT) — uses observed motion as next prompt |
| `openvla_only` | Standard VLA with task description (no langact, no GPT) |
| `all` | Run all four modes per episode for side-by-side comparison |

### `gpt_multistep` — Open-Loop Multi-Step Planning with Verification

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         Planning Cycle                                  │
│                                                                         │
│   ┌───────────┐   N prompts    ┌───────────┐    N×8 actions            │
│   │  GPT      │ ────────────►  │  VLA      │ ──────────►  env          │
│   │ (planner) │   open-loop    │  (langact │                           │
│   └─────▲─────┘                │   policy) │                           │
│         │                      └───────────┘                           │
│         │                                                               │
│   ┌─────┴─────┐   before/after images + execution log                  │
│   │  GPT      │ ◄────────────────────────────────────────              │
│   │ (verifier)│ ── correction prompt (if needed) ──► VLA ──► env       │
│   └───────────┘                                                         │
│                                                                         │
│   Then repeat: plan → execute → verify → (correct) → plan → ...        │
└─────────────────────────────────────────────────────────────────────────┘
```

Each planning cycle:
1. **Plan**: GPT generates `plan_horizon` (default 3) consecutive langact prompts
2. **Execute**: VLA runs all N chunks open-loop without re-querying GPT
3. **Verify**: GPT compares before/after images + commanded vs observed motions
4. **Correct**: If a major error is detected (missed grasp, overshoot), GPT generates ONE corrective prompt

Key design choices:
- GPT is told NOT to plan past gripper transitions (grasp/place) — these are critical re-observation points
- Verification distinguishes minor deviations (< 1 cm, handled by next plan) from major errors (need immediate correction)
- Reduces GPT API calls by ~3× compared to reactive mode (N chunks per call instead of 1)

## Features

| Feature | Description | Flag |
|---------|-------------|------|
| **Proprioceptive state** | EEF position (meters + cm), euler orientation (degrees), gripper state, delta since last chunk | always on |
| **Calibration info** | Exact unit conversions (1 cm = 0.01 m, 10° = 0.1745 rad), workspace bounds, verification guidance | always on |
| **Text ICL** | 3 text-only trajectory examples in system prompt | `--use_icl` (default: True) |
| **Image ICL** | Few-shot image+prompt pairs extracted from real training data (3 suites × ~6 key chunks each) | `--use_image_icl` (default: False) |
| **Task decomposition** | GPT decomposes task into subtasks (approach/grasp/lift/...) before execution | `--use_decomposition` (default: True) |
| **Multi-step planning** | GPT plans N chunks at once (open-loop), reducing API calls | `--plan_horizon 3` (gpt_multistep mode) |
| **Verification / self-correction** | After plan execution, GPT verifies outcome and generates corrective prompt if needed | `--use_verification` (default: True, gpt_multistep mode) |
| **Structured output** | JSON: observation, progress, subtask, reasoning, prompt, confidence, task_complete | always on |

## Prerequisites

1. **OpenAI API key** with access to GPT-4o (or `gpt-4o-mini` for cheaper runs)
2. **GPU** — OpenVLA-OFT model needs ~16 GB VRAM
3. **Langact-finetuned checkpoint** — trained on `libero_combined_langact_verbose_noisy_v2`
4. **Environment** — openvla-oft conda env with LIBERO installed

## Quick Start

```bash
cd /proj/inf-scaling/zwhong/idan/vla-rl/openvla-oft
export OPENAI_API_KEY='sk-...'

# Reactive mode: GPT generates 1 prompt per chunk
python experiments/robot/libero/run_libero_eval_gpt_langact.py \
    --pretrained_checkpoint runs/<langact_verbose_checkpoint> \
    --unnorm_key libero_combined_langact_verbose_noisy_v2 \
    --task_suite_name libero_goal \
    --task_id_start 0 --task_id_end 1 \
    --num_trials_per_task 1 \
    --eval_mode gpt_langact \
    --gpt_model gpt-4o

# Multi-step mode: GPT plans 3 chunks at once + verification
python experiments/robot/libero/run_libero_eval_gpt_langact.py \
    --pretrained_checkpoint runs/<langact_verbose_checkpoint> \
    --unnorm_key libero_combined_langact_verbose_noisy_v2 \
    --task_suite_name libero_goal \
    --task_id_start 0 --task_id_end 1 \
    --num_trials_per_task 1 \
    --eval_mode gpt_multistep \
    --plan_horizon 3 \
    --use_verification True \
    --gpt_model gpt-4o

# Compare all 4 modes side-by-side
python experiments/robot/libero/run_libero_eval_gpt_langact.py \
    --pretrained_checkpoint runs/<langact_verbose_checkpoint> \
    --unnorm_key libero_combined_langact_verbose_noisy_v2 \
    --task_suite_name libero_goal \
    --task_id_start 0 --task_id_end 1 \
    --num_trials_per_task 5 \
    --eval_mode all \
    --gpt_model gpt-4o

# Ablation: no ICL, no decomposition
python experiments/robot/libero/run_libero_eval_gpt_langact.py \
    --pretrained_checkpoint runs/<langact_verbose_checkpoint> \
    --unnorm_key libero_combined_langact_verbose_noisy_v2 \
    --task_suite_name libero_goal \
    --task_id_start 0 --task_id_end 1 \
    --num_trials_per_task 3 \
    --eval_mode gpt_langact \
    --use_icl False \
    --use_decomposition False
```

## Image-Based ICL (from real training data)

Instead of text-only trajectory examples, you can provide GPT with actual
observation images paired with the correct langact prompts, extracted from
the RLDS training datasets.

### 1. Extract ICL trajectories

```bash
# Requires the venv with tensorflow + tensorflow_datasets
/proj/inf-scaling/zwhong/idan/vla-rl/lap/.venv/bin/python \
    scripts/libero_langact/extract_icl_trajectories.py \
    --data_dir /proj/inf-scaling/SWE_sim/modified_libero_rlds_new \
    --output_dir ./icl_langact_trajectories \
    --image_every 8
```

This extracts 3 trajectories (one from `libero_goal`, `libero_object`, `libero_10`):
- Saves observation images every 8 steps (= every action chunk boundary)
- Records the full sequence of langact prompts with EEF state
- Retrieves the original task description from the non-langact dataset
- Output: `icl_langact_trajectories/icl_langact_metadata.json` + per-trajectory PNG images

### 2. Run eval with image ICL

```bash
python experiments/robot/libero/run_libero_eval_gpt_langact.py \
    --pretrained_checkpoint runs/<langact_verbose_checkpoint> \
    --unnorm_key libero_combined_langact_verbose_noisy_v2 \
    --task_suite_name libero_goal \
    --task_id_start 0 --task_id_end 1 \
    --num_trials_per_task 3 \
    --eval_mode gpt_langact \
    --use_image_icl True \
    --gpt_model gpt-4o
```

When `--use_image_icl True`, the system:
1. Loads the metadata + images from `icl_langact_trajectories/`
2. Selects ~6 key chunks per trajectory (first, grasp transition, mid-transport, place, last)
3. Inserts them as few-shot user/assistant message pairs before the real query
4. Text ICL in the system prompt is automatically disabled to avoid redundancy

**Note**: Image ICL increases API cost significantly (~18 images total). Use
`gpt-4o-mini` for cheaper development runs.

---

## GPT Prompts (Exact Text)

This section documents the exact prompts sent to GPT in each mode.

### 1. Langact Vocabulary (included in all system prompts)

```
The low-level policy understands language-action prompts that describe the
desired motion over the next 8 simulation steps. Prompts are comma-separated
phrases composed from the vocabulary below.

## Direction vocabulary

Translation (in centimeters):
  move forward / move back       (robot's x-axis)
  move left    / move right      (robot's y-axis)
  move up      / move down       (robot's z-axis)

Rotation (in degrees, rounded to nearest 10):
  tilt left    / tilt right      (roll)
  tilt back    / tilt forward    (pitch)
  rotate counterclockwise / rotate clockwise  (yaw)

Gripper:
  open gripper / close gripper

## Magnitude ranges (cumulative over 8 steps)

Translation:
  Very small:  0.1 – 0.5 cm   (barely visible nudge)
  Small:       0.5 – 2.0 cm   (fine adjustment)
  Medium:      2.0 – 5.0 cm   (typical reaching)
  Large:       5.0 – 8.0 cm   (fast repositioning)

Rotation:
  10 degrees   (small correction)
  20 degrees   (noticeable rotation)

## Prompt format

Verbose (with magnitudes): "<motion> <value> <unit>, ..., <gripper>"
Examples:
  "move forward 3.0 cm, move down 2.0 cm, close gripper"
  "move left 1.5 cm, tilt right 10 degrees, open gripper"
  "close gripper"
  "move forward 5.0 cm, move up 3.0 cm, tilt back 10 degrees, open gripper"

Translation values use 1 decimal place in cm (e.g. 3.0, 0.5, 1.2).
Rotation values are integers in degrees, rounded to nearest 10 (e.g. 10, 20).
Omit any axis with negligible motion. Always end with a gripper command.

## Important notes

- Translation is reliably steerable. Rotation is less reliable — use sparingly.
- The gripper should be OPEN while approaching, CLOSED while grasping/carrying,
  and OPEN to release/place.
- Each prompt covers only 8 steps (~0.5-8 cm of motion). Plan one chunk at a time.
- Magnitudes are cumulative over the 8-step chunk, NOT per-step.
```

### 2. Calibration Info (included in all system prompts)

```
## Unit calibration (exact conversions)

The proprioceptive state you receive uses SI units (meters, radians).
The langact prompts use centimeters and degrees.

Position:
  1 cm in prompt  =  0.01 meters of EEF delta
  "move forward 3.0 cm" means the EEF x-coordinate should increase by 0.030 m
  "move left 1.5 cm"    means the EEF y-coordinate should increase by 0.015 m
  "move up 2.0 cm"      means the EEF z-coordinate should increase by 0.020 m

Rotation:
  10 degrees in prompt  =  0.1745 radians of Euler angle change
  20 degrees in prompt  =  0.3491 radians of Euler angle change

Gripper:
  gripper qpos > 0.03  →  gripper is currently open
  gripper qpos ≤ 0.03  →  gripper is currently closed

## How to use proprioception for planning

You receive the EEF position in meters. To decide how far to move:
  1. Estimate the target position from the image.
  2. Compare with current EEF position (e.g., EEF x=0.40).
  3. Compute the delta: 0.45 - 0.40 = 0.05 m = 5.0 cm.
  4. Generate: "move forward 5.0 cm"

The LIBERO tabletop workspace is approximately:
  x (forward/back):  0.15 – 0.75 m   (robot facing +x)
  y (left/right):   -0.40 – 0.40 m   (robot centered at y≈0)
  z (up/down):       0.75 – 1.15 m   (table surface ≈ 0.81 m)

## Verifying your commands

After each chunk executes, you will see the "observed motion":
  - If commanded "move forward 3.0 cm" and observed "move forward 2.8 cm",
    the execution was accurate — proceed normally.
  - If commanded "move forward 3.0 cm" and observed "move forward 0.5 cm",
    there may be a collision or undershoot — try a different approach.
  - If commanded "move left 2.0 cm" and observed "move right 1.0 cm",
    the policy may be ignoring — try a smaller magnitude or different direction.
```

### 3. Task Decomposition Prompt (`query_gpt_decompose`)

Sent once at episode start (if `--use_decomposition True`). Receives the initial observation image.

```
You are a robot task planner. Given a task description and the initial
observation of a tabletop scene, decompose the task into a sequence of
high-level subtasks.

Each subtask should be one of:
  - approach: move gripper toward the target object (open gripper)
  - align: fine-adjust position to be directly above/beside the object
  - descend: move down to reach the object
  - grasp: close gripper around the object
  - lift: raise the grasped object
  - transport: move the object horizontally toward the target location
  - lower: move the object down toward the target surface
  - place: release the object (open gripper)
  - retract: move gripper away after placing
  - push/pull: for tasks involving pushing or pulling (e.g., opening drawers)

Output a JSON object:
{
  "scene_description": "What you see in the image",
  "task_analysis": "How to accomplish this task",
  "subtasks": ["approach", "descend", "grasp", "lift", "transport", "lower", "place", "retract"]
}
```

### 4. Reactive Planner Prompt (`gpt_langact` mode — `build_system_prompt`)

System prompt for per-chunk reactive planning. Includes vocabulary + calibration + ICL examples, then:

```
## Your task

At each step you receive:
1. The task description (what the robot should accomplish)
2. The current RGB observation image
3. The previous RGB observation image (if available)
4. The robot's proprioceptive state (EEF position, orientation, gripper)
5. History of previous prompts and their observed outcomes

Follow these steps:

1. **OBSERVE**: Describe what you see. Where is the gripper? Where are the
   relevant objects? What is the gripper holding (if anything)?

2. **ASSESS PROGRESS**: If history is provided, what happened in the last
   chunk? Did the robot move as expected? Is it closer to completing the task?

3. **IDENTIFY SUBTASK**: What is the current subtask?
   (e.g., approach, align, descend, grasp, lift, transport, lower, place, retract)

4. **PLAN MOTION**: For each degree of freedom, decide direction and magnitude
   for this ONE chunk (8 steps). Think about:
   - Translation: which direction and how far? (0.5–8.0 cm)
   - Gripper: should it be open or closed?
   - Rotation: only if clearly needed

5. **GENERATE PROMPT**: Compose the language-action prompt string.

6. **OUTPUT**: Respond with a JSON object inside a ```json code block:
{
  "observation": "Brief description of what you see",
  "progress": "What happened since last step (or 'first step' if no history)",
  "subtask": "current subtask name",
  "reasoning": "Why this motion is needed",
  "langact_prompt": "move forward 3.0 cm, move down 2.0 cm, close gripper",
  "confidence": 0.8,
  "task_complete": false
}

IMPORTANT RULES:
- The `langact_prompt` MUST follow the exact vocabulary and format.
- Always include a gripper command (open gripper / close gripper) at the end.
- Use translation magnitudes with 1 decimal (e.g., 3.0 cm, not 3 cm).
- Rotation must be 10 or 20 degrees (rounded to nearest 10).
- Plan for only ONE 8-step chunk. Do NOT plan the entire trajectory.
- Set `task_complete` to true only if you can see the task is fully done.
```

**User message structure** (sent each chunk):

```
Task: <task_description>
Task plan (subtasks): approach, descend, grasp, lift, transport, place, retract
Previous observation (before last chunk): [IMAGE]
Current observation: [IMAGE]
Proprioceptive state:
  EEF position (meters): x=0.4200, y=0.1500, z=0.8500
  EEF position (cm):     x=42.0,   y=15.0,   z=85.0
  EEF orientation (degrees): roll=-2.3°, pitch=15.1°, yaw=0.5°
  Gripper: open (qpos: 0.040, 0.040)
  Delta since last chunk (cm): dx=+2.8, dy=-0.3, dz=-1.5
Recent history (last 5 chunks):
  Chunk 0: prompt="move forward 3.0 cm, open gripper" → observed="move forward 2.8 cm, open gripper"
  Chunk 1: prompt="move forward 2.0 cm, move down 1.0 cm, open gripper" → observed="..."
Generate the next language-action prompt. Output JSON:
```

### 5. Multi-Step Planner Prompt (`gpt_multistep` mode — `build_multistep_system_prompt`)

System prompt for open-loop planning. Includes vocabulary + calibration + ICL, then:

```
## Your task — Multi-Step Planning (horizon = 3)

At each planning step you receive:
1. The task description
2. The current RGB observation image
3. The previous RGB observation image (if available)
4. Robot proprioceptive state (EEF position, orientation, gripper)
5. History of previous prompts and their observed outcomes
6. Task decomposition (if available)

Follow these steps:

1. **OBSERVE**: Describe the scene — gripper position, object locations,
   what the gripper holds.

2. **ASSESS PROGRESS**: Review history. Did previous commands achieve
   their intended effect?

3. **IDENTIFY PHASE**: Where are we in the task?
   (approach / align / descend / grasp / lift / transport / lower / place / retract)

4. **PLAN 3 CHUNKS**: Generate 3 consecutive langact prompts. Think about
   the cumulative motion — each chunk moves ~1–8 cm, so 3 chunks cover
   ~3–24 cm total. Consider:
   - What trajectory shape is needed? (e.g., descend then close gripper)
   - Where should the gripper state change?
   - Do NOT plan past a grasp or place transition — those are critical
     moments where re-observation is needed.

5. **OUTPUT**: Respond with a JSON object inside a ```json code block:
{
  "observation": "Brief description of what you see",
  "progress": "Assessment of recent history",
  "phase": "current phase",
  "reasoning": "Why this plan is appropriate",
  "plan": [
    {"prompt": "move forward 3.0 cm, move down 2.0 cm, open gripper",
     "intent": "approach the target object"},
    {"prompt": "move forward 2.0 cm, move down 4.0 cm, open gripper",
     "intent": "descend to grasp height"},
    {"prompt": "move forward 1.0 cm, move down 1.0 cm, close gripper",
     "intent": "close gripper to grasp"}
  ],
  "expect_grasp_or_place": true,
  "confidence": 0.8,
  "task_complete": false
}

IMPORTANT RULES:
- The `plan` array MUST contain exactly 3 entries.
- Each `prompt` MUST follow the exact langact vocabulary and format.
- Always include a gripper command at the end of each prompt.
- Set `expect_grasp_or_place` to true if ANY prompt changes the gripper state.
- Set `task_complete` to true only if the task is fully done.
- If fewer than 3 chunks remain, pad with small motions.
```

### 6. Verification / Self-Correction Prompt (`query_gpt_verify`)

Sent after executing a multi-step plan. Receives before/after images + execution log.

```
You are a robot motion verifier. After a sequence of action chunks was
executed, you compare the commanded motions against the observed motions
and the resulting scene.

Your job:
1. Check if each commanded motion was executed accurately.
2. Assess whether the overall goal was advanced.
3. If something went wrong (e.g., missed a grasp, overshot, wrong direction),
   generate ONE corrective langact prompt to fix it.
4. If execution was satisfactory, indicate no correction is needed.

[LANGACT_VOCABULARY]
[CALIBRATION_INFO]

Respond with JSON inside a ```json code block:
{
  "assessment": "Brief assessment of plan execution",
  "errors_detected": [
    {"chunk": 0, "issue": "description of problem", "severity": "minor|major"}
  ],
  "needs_correction": true,
  "correction_prompt": "move back 2.0 cm, move down 1.0 cm, close gripper",
  "correction_reasoning": "Why this correction is needed",
  "confidence": 0.7
}

Rules:
- Set `needs_correction` to false if execution was acceptable (even if not perfect).
- If `needs_correction` is false, set `correction_prompt` to null.
- Only generate corrections for MAJOR errors (missed grasp, wrong direction,
  significant overshoot/undershoot).
- Minor deviations (< 1 cm) do not need correction — the next planning cycle
  will handle them.
- The correction prompt follows the same langact format.
```

**User message structure** (sent after plan execution):

```
Task: <task_description>
Task plan (subtasks): approach, descend, grasp, lift, transport, place
Observation BEFORE plan execution: [IMAGE]
Observation AFTER plan execution: [IMAGE]
Current proprioceptive state:
  EEF position (meters): ...
  EEF position (cm): ...
  ...
Execution log (3 chunks):
  Chunk 0: commanded="move forward 3.0 cm, open gripper" | intent="approach" | observed="move forward 2.8 cm, open gripper"
  Chunk 1: commanded="move down 4.0 cm, open gripper" | intent="descend" | observed="move down 3.5 cm, open gripper"
  Chunk 2: commanded="close gripper" | intent="grasp" | observed="close gripper"
Verify execution and output JSON:
```

### 7. Text-Based ICL Examples (included when `--use_icl True`)

```
## In-context trajectory examples

### Example A — "open the middle drawer of the cabinet"
Step  0: move forward 1.8 cm, move down 1.5 cm, open gripper     (approach the drawer handle)
Step  1: move forward 2.3 cm, move down 0.8 cm, open gripper     (continue approaching)
Step  2: move forward 1.0 cm, move down 0.3 cm, close gripper    (reached handle, grasp it)
Step  3: move back 3.5 cm, close gripper                          (pull drawer open)
Step  4: move back 4.0 cm, close gripper                          (continue pulling)
Step  5: move back 2.0 cm, open gripper                           (release handle)
Outcome: drawer is open. Total chunks: 6.

### Example B — "put the bowl on the plate"
Step  0: move forward 2.0 cm, move down 1.0 cm, move left 1.5 cm, open gripper   (approach)
Step  1: move forward 1.5 cm, move down 2.0 cm, open gripper                      (descend)
Step  2: move forward 0.5 cm, move down 0.5 cm, close gripper                     (grasp)
Step  3: move up 3.0 cm, close gripper                                            (lift)
Step  4: move back 2.0 cm, move right 3.0 cm, close gripper                       (transport)
Step  5: move forward 1.0 cm, move down 2.5 cm, close gripper                     (lower)
Step  6: move down 1.0 cm, open gripper                                           (release)
Step  7: move up 2.0 cm, move back 1.0 cm, open gripper                           (retract)
Outcome: bowl is on the plate. Total chunks: 8.

### Example C — "pick up the red mug and place it on the shelf"
Step  0: move forward 3.0 cm, move down 1.0 cm, open gripper            (approach)
Step  1: move forward 1.5 cm, move down 1.5 cm, move right 0.5 cm, open gripper  (align)
Step  2: move down 0.5 cm, close gripper                                (grasp)
Step  3: move up 4.0 cm, close gripper                                  (lift)
Step  4: move up 3.0 cm, move back 1.0 cm, close gripper                (lift higher)
Step  5: move forward 2.0 cm, move left 2.0 cm, close gripper           (transport)
Step  6: move forward 1.5 cm, move down 1.0 cm, close gripper           (position over shelf)
Step  7: move down 1.5 cm, open gripper                                 (place)
Step  8: move up 2.0 cm, move back 1.5 cm, open gripper                 (retract)
Outcome: mug is on the shelf. Total chunks: 9.

### Example D — "put the wine bottle on the rack" (from real training data)
Step  0: move back 0.3 cm, move down 0.9 cm, open gripper           (begin approach)
Step  1: move forward 0.2 cm, move left 0.6 cm, open gripper        (lateral adjustment)
Step  2: move down 1.3 cm, open gripper                              (descending)
Step  3: move back 0.3 cm, move down 2.4 cm, open gripper           (descending toward bottle)
Step  4: move back 1.0 cm, move down 3.1 cm, open gripper           (reaching bottle)
Step  5: move back 0.8 cm, move down 2.3 cm, move right 0.4 cm, open gripper  (final descent)
Step  6: move forward 2.3 cm, move down 1.1 cm, move right 1.8 cm, open gripper  (aligning)
Step  7: move forward 0.7 cm, move down 0.5 cm, move right 1.8 cm, open gripper  (fine alignment)
Step  8: move back 2.6 cm, move down 1.8 cm, move right 2.3 cm, open gripper  (positioning)
Step  9: move back 0.8 cm, move down 3.2 cm, move right 2.0 cm, open gripper  (final approach)
Step 10: move forward 0.2 cm, move down 0.8 cm, close gripper       (grasp bottle)
Step 11: move forward 1.9 cm, move up 4.7 cm, close gripper         (lift)
Step 12: move forward 0.8 cm, move up 8.9 cm, move right 1.8 cm, close gripper  (lift high)
Step 13: move back 0.7 cm, move up 7.3 cm, move right 4.4 cm, close gripper  (transport right)
Step 14: move back 0.5 cm, move up 4.5 cm, move right 6.5 cm, close gripper  (transport right)
Step 15: move back 0.4 cm, move down 1.6 cm, move right 6.3 cm, close gripper  (approach rack)
Step 16: move forward 0.3 cm, move down 3.7 cm, close gripper       (lower onto rack)
Step 17: move forward 1.0 cm, open gripper                           (release)
Step 18: move back 0.2 cm, move up 0.2 cm, open gripper             (retract)
Outcome: wine bottle on rack. Total chunks: 19.
Note: long approach (10 chunks), large lateral transport (right), bottle lifted high.

### Example E — "put the bowl on the plate" (from real training data)
Step  0: move forward 3.9 cm, move down 0.9 cm, move left 1.5 cm, open gripper   (approach)
Step  1: move forward 9.1 cm, move down 2.1 cm, move left 2.1 cm, open gripper   (large reach)
Step  2: move forward 5.6 cm, move down 7.4 cm, open gripper                      (descend rapidly)
Step  3: move back 3.0 cm, move down 8.6 cm, open gripper                         (continue descending)
Step  4: move back 5.7 cm, move down 6.0 cm, move right 1.5 cm, close gripper     (grasp bowl)
Step  5: move back 0.5 cm, move up 0.9 cm, close gripper                          (slight lift)
Step  6: move forward 2.9 cm, move up 6.9 cm, close gripper                       (lift bowl)
Step  7: move forward 7.5 cm, move up 4.6 cm, move right 3.0 cm, close gripper    (transport)
Step  8: move forward 4.3 cm, move down 6.0 cm, close gripper                     (lower toward plate)
Step  9: move back 0.5 cm, move down 5.7 cm, close gripper                        (lower onto plate)
Step 10: move forward 0.1 cm, move up 2.4 cm, open gripper                        (release + retract)
Outcome: bowl on plate. Total chunks: 11.
Note: large magnitudes (5-9 cm) during approach/transport, grasp at chunk 4-5.

### Key patterns:
1. Approach phase: forward + down, gripper OPEN (2-3 chunks)
2. Grasp: small/no motion, gripper changes from OPEN → CLOSE
3. Lift: primarily up, gripper CLOSED (1-2 chunks)
4. Transport: lateral/forward motion, gripper CLOSED (1-3 chunks)
5. Place: down motion, then gripper OPEN
6. Retract: up + back, gripper OPEN (1 chunk)
7. Wine bottle tasks: more chunks (longer approach + careful alignment)
8. Real trajectories use magnitudes from 0.1 cm to 9+ cm depending on distance
```

---

## Command-Line Arguments

### GPT Langact-Specific

| Argument | Default | Description |
|----------|---------|-------------|
| `--eval_mode` | `gpt_langact` | `gpt_langact`, `gpt_multistep`, `openvla_langact`, `openvla_only`, or `all` |
| `--gpt_model` | `gpt-4o` | OpenAI model ID |
| `--use_icl` | `True` | Include text-only trajectory examples in system prompt |
| `--use_image_icl` | `False` | Include image-based ICL from extracted trajectories (replaces text ICL) |
| `--icl_trajectories_dir` | `""` | Path to extracted ICL directory (empty = default `icl_langact_trajectories/`) |
| `--use_decomposition` | `True` | GPT decomposes task into subtasks before execution |
| `--plan_horizon` | `3` | Number of chunks GPT plans at once (`gpt_multistep` mode) |
| `--use_verification` | `True` | Verify + self-correct after plan execution (`gpt_multistep` mode) |
| `--video_dir` | `./rollout_videos_langact` | Root output directory |
| `--video_fps` | `30` | FPS for saved videos |
| `--initial_langact_prompt` | `"move forward 3.0 cm, ..."` | First-chunk prompt for `openvla_langact` mode |
| `--custom_task_description` | `""` | Override task description for GPT planning |

### Model (inherited from base config)

| Argument | Default | Description |
|----------|---------|-------------|
| `--pretrained_checkpoint` | (required) | Path to langact-finetuned checkpoint |
| `--unnorm_key` | `""` | Action un-normalization key (must match checkpoint's `dataset_statistics.json`) |
| `--lora_rank` | `32` | LoRA rank (must match training) |
| `--use_l1_regression` | `True` | L1 regression action head |
| `--use_diffusion` | `False` | Diffusion action head |
| `--num_images_in_input` | `2` | Number of images in VLA input |
| `--use_proprio` | `True` | Include proprio state |

### LIBERO Environment (inherited)

| Argument | Default | Description |
|----------|---------|-------------|
| `--task_suite_name` | `libero_spatial` | Suite: `libero_spatial`, `libero_object`, `libero_goal`, `libero_10` |
| `--task_id_start` | `0` | First task ID (inclusive) |
| `--task_id_end` | `-1` | Last task ID (exclusive); `-1` = all |
| `--num_trials_per_task` | `50` | Episodes per task |
| `--custom_task_description` | `""` | Override task description |
| `--seed` | `7` | Random seed |

## Output Structure

### `gpt_langact` mode

```
rollout_videos_langact/<ckpt>_<suite>/
  <task_description>/
    ep000/
      gpt_langact--success=True.mp4
      gpt_langact/
        episode_summary.json              # Full episode: task, decomposition, all chunk logs
        chunk_000/
          current_obs.png                 # Image sent to GPT
          previous_obs.png                # Previous image sent to GPT (if any)
          post_action_obs.png             # Image after VLA executed chunk
          gpt_input.json                  # Everything sent to GPT: task, proprio, history
          gpt_response.json               # GPT's full response: parsed JSON + raw text
          chunk_outcome.json              # What happened: commanded vs observed, post-proprio
        chunk_001/
          ...
      openvla_langact--success=False.mp4
      openvla_only--success=False.mp4
      side_by_side.mp4
```

### `gpt_multistep` mode

```
gpt_multistep/
  episode_summary.json                    # Full episode: plans, chunks, corrections
  plan_000/
    obs_before_plan.png                   # Observation when plan was generated
    obs_previous.png                      # Previous observation
    plan_input.json                       # Input to GPT: task, proprio, history
    plan_response.json                    # GPT's plan: N prompts with intents
    obs_after_plan.png                    # Observation after all chunks executed
    verify_response.json                  # Verification assessment + errors
    chunk_000/
      obs_before.png
      obs_after.png
      chunk_outcome.json                  # commanded, intent, observed, post-proprio
    chunk_001/
      ...
    correction_003/                       # Only if correction was needed
      obs_before.png
      obs_after.png
      correction.json
  plan_001/
    ...
```

## GPT Response Formats

### Reactive mode (`gpt_langact`)

```json
{
  "observation": "The gripper is above and behind the drawer handle, open",
  "progress": "Last chunk moved forward 2.8 cm and down 1.2 cm as intended",
  "subtask": "descend",
  "reasoning": "Need to descend further to reach the handle before grasping",
  "langact_prompt": "move forward 1.0 cm, move down 1.5 cm, open gripper",
  "confidence": 0.85,
  "task_complete": false
}
```

### Multi-step mode (`gpt_multistep`)

```json
{
  "observation": "Gripper is 5 cm above and behind the bowl, open",
  "progress": "First planning step, no history yet",
  "phase": "approach",
  "reasoning": "Need to approach the bowl, descend, then grasp",
  "plan": [
    {"prompt": "move forward 3.0 cm, move down 2.0 cm, open gripper",
     "intent": "approach the bowl"},
    {"prompt": "move forward 1.0 cm, move down 3.0 cm, open gripper",
     "intent": "descend to grasp height"},
    {"prompt": "move forward 0.5 cm, move down 0.5 cm, close gripper",
     "intent": "grasp the bowl"}
  ],
  "expect_grasp_or_place": true,
  "confidence": 0.8,
  "task_complete": false
}
```

### Verification response

```json
{
  "assessment": "Plan executed well. Approach and descent were accurate. Grasp appears successful — object is held.",
  "errors_detected": [],
  "needs_correction": false,
  "correction_prompt": null,
  "correction_reasoning": null,
  "confidence": 0.9
}
```

### Verification with correction

```json
{
  "assessment": "Approach was accurate but gripper closed too early — object was not grasped.",
  "errors_detected": [
    {"chunk": 2, "issue": "Gripper closed but object not in grasp — EEF was 1.5 cm too high", "severity": "major"}
  ],
  "needs_correction": true,
  "correction_prompt": "move down 1.5 cm, close gripper",
  "correction_reasoning": "Need to descend 1.5 cm more and re-attempt grasp",
  "confidence": 0.7
}
```

## Cost Estimate

Each GPT call sends 2 images (prev + current) + text. With GPT-4o:

| Scenario | GPT Calls | Approx Cost (GPT-4o) |
|----------|-----------|---------------------|
| 1 task × 1 episode (reactive) | ~30 | ~$0.15 |
| 1 task × 1 episode (multistep, horizon=3) | ~12 | ~$0.08 |
| 1 task × 10 episodes | ~300 / ~120 | ~$1.50 / ~$0.80 |
| 10 tasks × 10 episodes | ~3,000 / ~1,200 | ~$15 / ~$8 |
| 10 tasks × 50 episodes | ~15,000 / ~6,000 | ~$75 / ~$40 |

Use `--gpt_model gpt-4o-mini` for ~10× cheaper runs.

## Troubleshooting

### "OPENAI_API_KEY not set"

```bash
export OPENAI_API_KEY='sk-...'
```

### GPT parse errors (falls back to default prompt)

Check `gpt_langact/chunk_NNN/gpt_response.json` for the raw GPT response.
If GPT consistently produces invalid prompts, it falls back to `"move forward 1.0 cm, open gripper"`.

### `max_tokens` error with newer models

Newer models (e.g., `gpt-5.4`) require `max_completion_tokens` instead of `max_tokens`.
This is already handled in the code.

### unnorm_key mismatch

For combined (all-task) models, the norm stats key is `libero_combined_langact_verbose_noisy_v2`,
not `libero_goal`. Pass `--unnorm_key libero_combined_langact_verbose_noisy_v2` explicitly.

### Model loads but actions are wrong

Make sure:
- `--lora_rank 32` matches training
- `--use_l1_regression True` matches training
- `--num_images_in_input 2` matches training
- The checkpoint is langact-finetuned (not a standard task-description checkpoint)

## Related Files

| File | Description |
|------|-------------|
| `run_libero_eval_gpt_langact.py` | Main evaluation script (this README) |
| `run_libero_eval_with_gpt.py` | GPT action correction (raw 7D actions, different approach) |
| `run_libero_eval.py` | Base evaluation script (no GPT) |
| `../../scripts/libero_langact/extract_icl_trajectories.py` | Extract image-based ICL from RLDS training data |
| `../../scripts/libero_langact/langact_utils.py` | Unit conversion: `summarize_numeric_actions`, rotation helpers |
| `../../scripts/libero_langact/test_steerability.py` | Steerability tests (prompt battery, no GPT) |
| `libero_utils.py` | LIBERO environment utilities |
| `../robot_utils.py` | Action processing utilities |
