"""
run_libero_eval_subgoal.py

Evaluates a subgoal-finetuned OpenVLA-OFT model on LIBERO-Goal tasks.

Three modes:
  oracle_subgoal  -- Per-subgoal test using RLDS action replay to initialize the
                     sim exactly at the start frame of each subgoal. The model
                     receives only that subgoal's prompt and runs until done/timeout.
                     Tests pure subgoal-following ability in the correct context.
  gpt_subgoal     -- GPT watches the scene and generates the current subgoal string
                     at each action-chunk boundary. Tests GPT's ability to plan and
                     steer the robot via subgoal language.
  openvla_only    -- Baseline: full task description, no subgoal conditioning.

Oracle setup:
  For each task, RLDS episodes are loaded from --rlds_results_dir
  (outputs/subgoal_results_rlds/ep{XXXXX}/results.json).  For each episode,
  demo actions are replayed to reach the start frame of each subgoal k.  The
  model then gets prompt subgoals[k] and runs for the subgoal's allotted steps.

Usage:
    cd /proj/inf-scaling/zwhong/idan/vla-rl/openvla-oft

    export OPENAI_API_KEY='sk-...'

    python experiments/robot/libero/run_libero_eval_subgoal.py \\
        --pretrained_checkpoint runs/<subgoal_checkpoint> \\
        --unnorm_key libero_subgoal \\
        --task_suite_name libero_goal \\
        --task_id_start 0 --task_id_end -1 \\
        --num_trials_per_task 3 \\
        --eval_mode oracle_subgoal \\
        --rlds_dataset libero_goal_no_noops \\
        --rlds_data_dir /proj/inf-scaling/SWE_sim/modified_libero_rlds_new \\
        --rlds_results_dir ../outputs/subgoal_results_rlds \\
        --video_dir ./rollout_videos_subgoal
"""

import base64
import io
import json
import logging
import os
import re
import sys
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import draccus
import imageio
import numpy as np
import torch
import tqdm
from libero.libero import benchmark
from PIL import Image

sys.path.append("../..")
from experiments.robot.libero.libero_utils import (
    get_libero_dummy_action,
    get_libero_env,
    get_libero_image,
    get_libero_wrist_image,
    quat2axisangle,
)
from experiments.robot.libero.run_libero_eval import (
    GenerateConfig as BaseGenerateConfig,
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


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


# =============================================================================
# Config
# =============================================================================

@dataclass
class SubgoalEvalConfig(BaseGenerateConfig):
    # fmt: off
    eval_mode: str = "oracle_subgoal"   # oracle_subgoal | gpt_subgoal | openvla_only | all

    # ── Oracle mode settings ──────────────────────────────────────────────────
    # RLDS dataset used to load demo actions for replay-based initialization
    rlds_dataset: str = "libero_goal_no_noops"
    rlds_data_dir: str = "/proj/inf-scaling/SWE_sim/modified_libero_rlds_new"
    # Directory of per-episode subgoal results (outputs/subgoal_results_rlds/)
    # Layout: <dir>/ep{XXXXX}/results.json  (contains episode_index, subgoals, boundaries)
    rlds_results_dir: str = "/proj/inf-scaling/zwhong/idan/vla-rl/outputs/subgoal_results_rlds"
    # How many extra steps to give beyond the demo duration for each subgoal
    # (demo_duration * slack_factor = allotted eval steps per subgoal)
    oracle_slack_factor: float = 2.0
    # Hard cap on steps per subgoal (0 = no cap)
    oracle_max_steps: int = 0
    # Which field(s) from steering_commands.json to use as the VLA prompt.
    # "all"  → try every key in STEERING_COMMAND_KEYS (one fresh replay per key)
    # comma-separated list → e.g. "task,subgoal,low_level_motion,low_level_motion_with_numbers"
    # single key → e.g. "subgoal"
    # Falls back to results.json subgoal string if steering_commands.json is missing.
    # Per-checkpoint recommendations:
    #   ml6: "task,subgoal,low_level_motion,low_level_motion_with_numbers,spatial_command,object_affordance"
    #   ml4: "task,subgoal,low_level_motion,low_level_motion_with_numbers"
    #   basic subgoal: "subgoal"
    oracle_command_key: str = "all"

    # ── GPT mode settings ─────────────────────────────────────────────────────
    gpt_model: str = "gpt-4o"
    # Provide the pre-computed subgoal list to GPT as a menu of choices
    use_precomputed_subgoals: bool = True
    # Re-query GPT every N action chunks (1 = every 8 steps)
    gpt_replan_every_n_chunks: int = 1

    # ── Output / logging ──────────────────────────────────────────────────────
    video_dir: str = "./rollout_videos_subgoal"
    local_log_dir: str = "./experiments/logs"
    # fmt: on


# =============================================================================
# RLDS episode loading and action replay
# =============================================================================

_rlds_cache: Dict[int, np.ndarray] = {}  # episode_index → actions array


def load_rlds_actions(rlds_dataset: str, rlds_data_dir: str, episode_index: int) -> np.ndarray:
    """Load demo actions for a single RLDS episode.  Returns (T, 7) array.

    Actions are stored in the format expected by the LIBERO env:
      [dx, dy, dz, droll, dpitch, dyaw, gripper]  with gripper ∈ {-1, +1}.
    """
    if episode_index in _rlds_cache:
        return _rlds_cache[episode_index]

    import tensorflow as tf
    import tensorflow_datasets as tfds
    tf.config.set_visible_devices([], "GPU")

    ds = tfds.load(
        rlds_dataset,
        data_dir=rlds_data_dir,
        split=f"train[{episode_index}:{episode_index + 1}]",
    )
    ep = next(iter(ds))
    steps = list(ep["steps"])
    actions = np.stack([s["action"].numpy() for s in steps])  # (T, 7)
    _rlds_cache[episode_index] = actions
    return actions


def replay_to_frame(env, initial_state, actions: np.ndarray, target_frame: int):
    """Reset env and step with demo actions to reach target_frame.

    Returns the observation at target_frame.  Actions are applied raw
    (no process_action) since RLDS already stores env-format actions.
    """
    env.reset()
    obs = env.set_init_state(initial_state)
    for t in range(min(target_frame, len(actions))):
        obs, _, done, _ = env.step(actions[t].tolist())
        if done:
            break
    return obs


# =============================================================================
# RLDS → (task_id, initial_state_idx) mapping
# =============================================================================

def build_episode_map(rlds_results_dir: str, task_suite) -> Dict[str, List[dict]]:
    """Scan rlds_results_dir and build a mapping from task description → episode list.

    Returns:
        { task_description: [ {episode_index, ep_dir, subgoal_data}, ... ] }
        sorted by episode_index within each task.
    """
    ep_map: Dict[str, list] = defaultdict(list)

    for ep_name in sorted(os.listdir(rlds_results_dir)):
        ep_dir = os.path.join(rlds_results_dir, ep_name)
        results_path = os.path.join(ep_dir, "results.json")
        if not os.path.exists(results_path):
            continue
        with open(results_path) as f:
            data = json.load(f)

        # Load steering_commands.json if available (provides richer per-subgoal prompts)
        steering_path = os.path.join(ep_dir, "steering_commands.json")
        if os.path.exists(steering_path):
            with open(steering_path) as f:
                data["steering_commands"] = json.load(f).get("steering_commands", [])

        task_str = data.get("task", "")
        ep_idx = data.get("episode_index", -1)
        ep_map[task_str].append({
            "episode_index": ep_idx,
            "ep_dir": ep_dir,
            "subgoal_data": data,
        })

    # Sort by episode_index within each task
    for key in ep_map:
        ep_map[key].sort(key=lambda x: x["episode_index"])

    return dict(ep_map)


def get_task_description(task_suite, task_id: int) -> str:
    """Return the canonical task description string for a libero_goal task."""
    task = task_suite.get_task(task_id)
    env, desc = get_libero_env(task, "openvla")
    env.close()
    return desc


# =============================================================================
# Image utilities (for GPT mode)
# =============================================================================

def encode_image_to_base64(img: np.ndarray) -> str:
    pil_img = Image.fromarray(img)
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def format_proprio(obs: dict) -> str:
    eef_pos = obs["robot0_eef_pos"]
    grip = obs["robot0_gripper_qpos"]
    grip_state = "open" if grip[0] > 0.03 else "closed"
    return (
        f"EEF position (m): x={eef_pos[0]:.3f}, y={eef_pos[1]:.3f}, z={eef_pos[2]:.3f}\n"
        f"Gripper: {grip_state}"
    )


# =============================================================================
# GPT subgoal generation
# =============================================================================

GPT_SYSTEM_PROMPT = """\
You are a high-level robot planner for a tabletop manipulation task.
You observe the scene through an RGB image and determine the current subgoal
that a low-level visuomotor policy should execute.

The low-level policy is an OpenVLA model fine-tuned to follow natural-language
subgoal instructions like "reach for the bowl", "grasp the bottle", or
"carry the object to the target". It executes 8 simulation steps per subgoal
chunk before re-querying you.

Your job: at each step, output the ONE subgoal string the low-level policy
should receive as its instruction right now.

Guidelines:
- A subgoal is a short, imperative natural-language phrase (3–12 words).
- Track progress: the robot advances through a sequence of subgoals to complete
  the task. Only advance to the next subgoal when the current one appears done.
- If a pre-computed subgoal list is provided, prefer strings from that list.
  You may also rephrase or generate new subgoals if the list doesn't cover
  the current situation.

Output ONLY a JSON object:
{"subgoal": "...", "reasoning": "...", "task_complete": false}
"""


def query_gpt_subgoal(
    client,
    task_description: str,
    obs_img: np.ndarray,
    obs_raw: dict,
    prev_subgoal: Optional[str],
    subgoal_history: List[str],
    precomputed_subgoals: Optional[List[str]],
    gpt_model: str = "gpt-4o",
    max_retries: int = 3,
) -> Tuple[str, dict]:
    """Query GPT to get the current subgoal string. Returns (subgoal, parsed_json)."""
    content: list = [{"type": "text", "text": f"Task: {task_description}"}]

    if precomputed_subgoals:
        sg_list = "\n".join(f"  {i}: \"{sg}\"" for i, sg in enumerate(precomputed_subgoals))
        content.append({"type": "text", "text": f"Pre-computed subgoal sequence:\n{sg_list}"})

    if subgoal_history:
        recent = subgoal_history[-5:]
        content.append({
            "type": "text",
            "text": "Recent subgoal history:\n" +
                    "\n".join(f"  [{i}] \"{sg}\"" for i, sg in enumerate(recent))
        })

    content.append({
        "type": "text",
        "text": f"Current subgoal: \"{prev_subgoal}\"" if prev_subgoal else "(First step.)"
    })
    content.append({"type": "text", "text": "Current observation:"})
    content.append({
        "type": "image_url",
        "image_url": {"url": f"data:image/png;base64,{encode_image_to_base64(obs_img)}"},
    })
    content.append({"type": "text", "text": f"Proprioceptive state:\n{format_proprio(obs_raw)}"})
    content.append({"type": "text", "text": "Output the subgoal for right now. Respond with JSON:"})

    messages = [
        {"role": "system", "content": GPT_SYSTEM_PROMPT},
        {"role": "user", "content": content},
    ]

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=gpt_model, messages=messages, temperature=0.0, max_completion_tokens=256,
            )
            raw = response.choices[0].message.content.strip()
            json_match = re.search(r"```(?:json)?\s*\n?([\s\S]*?)```", raw)
            if json_match:
                parsed = json.loads(json_match.group(1).strip())
            else:
                parsed = json.loads(re.search(r"(\{[\s\S]*\})", raw).group(1).strip())
            subgoal = parsed.get("subgoal", "").strip()
            if not subgoal:
                raise ValueError("Empty subgoal")
            return subgoal, parsed
        except Exception as e:
            if attempt < max_retries - 1:
                logger.warning(f"GPT subgoal attempt {attempt+1} failed: {e}. Retrying...")
                time.sleep(2)
            else:
                logger.error(f"GPT subgoal failed: {e}")
                fallback = prev_subgoal or task_description
                return fallback, {"subgoal": fallback, "error": str(e)}

    return prev_subgoal or task_description, {}


# =============================================================================
# Core action execution helper
# =============================================================================

def query_and_execute_chunk(
    cfg, model, obs, resize_size, prompt, processor,
    action_head, proprio_projector, noisy_action_projector,
) -> Tuple[List[dict], List[np.ndarray], bool]:
    """Query VLA with prompt and execute one action chunk.

    Returns (actions_taken, step_images, done).
    """
    observation, img = prepare_observation(obs, resize_size)
    with torch.no_grad():
        actions = get_action(
            cfg=cfg, model=model, obs=observation, task_label=prompt,
            processor=processor, action_head=action_head,
            proprio_projector=proprio_projector,
            noisy_action_projector=noisy_action_projector,
            use_film=cfg.use_film,
        )
    return actions, img


# =============================================================================
# Oracle per-subgoal episode runner
# =============================================================================

# All steering command fields available in steering_commands.json
STEERING_COMMAND_KEYS = [
    "task",
    "subgoal",
    "low_level_motion",
    "low_level_motion_with_numbers",
    "spatial_command",
    "object_affordance",
]


def _run_one_subgoal(
    cfg: SubgoalEvalConfig,
    env,
    model,
    resize_size: int,
    initial_state,
    demo_actions: np.ndarray,
    start_frame: int,
    allotted_steps: int,
    prompt: str,
    processor=None,
    action_head=None,
    proprio_projector=None,
    noisy_action_projector=None,
) -> Tuple[bool, int, List[np.ndarray]]:
    """Replay to start_frame, stabilize, then run VLA with prompt for allotted_steps.

    Construction:
      1. env.reset() clears any terminated state from the previous run.
      2. env.set_init_state(initial_state) restores the episode's starting physics
         (qpos, qvel, object poses) — same state used for the full episode eval.
      3. replay_to_frame steps the env with ground-truth RLDS demo actions from
         frame 0 up to (but not including) start_frame, placing the sim in the
         exact physical state at the beginning of subgoal k.
      4. num_steps_wait dummy no-op actions let the physics engine settle after
         the instantaneous state set.
      5. The VLA is queried with `prompt` and executes action chunks (8 sim steps
         each) for up to allotted_steps chunks. Task success is detected by the
         env's own done signal (robosuite success detector).

    Returns (success, steps_taken, frame_images).
    """
    # ── 1–3: Replay demo to the correct starting frame ──────────────────────
    obs = replay_to_frame(env, initial_state, demo_actions, start_frame)

    # ── 4: Physics stabilization ─────────────────────────────────────────────
    for _ in range(cfg.num_steps_wait):
        obs, _, _, _ = env.step(get_libero_dummy_action(cfg.model_family))

    # ── 5: VLA execution loop ─────────────────────────────────────────────────
    replay_images = []
    success = False
    steps_taken = 0

    try:
        for _ in range(allotted_steps):
            observation, img = prepare_observation(obs, resize_size)
            replay_images.append(img)

            with torch.no_grad():
                actions = get_action(
                    cfg=cfg, model=model, obs=observation, task_label=prompt,
                    processor=processor, action_head=action_head,
                    proprio_projector=proprio_projector,
                    noisy_action_projector=noisy_action_projector,
                    use_film=cfg.use_film,
                )

            for action in actions:
                action = process_action(action, cfg.model_family)
                obs, _, done, _ = env.step(action.tolist())
                _, step_img = prepare_observation(obs, resize_size)
                replay_images.append(step_img)
                steps_taken += 1
                if done:
                    success = True
                    break
            if success:
                break

    except Exception as e:
        logger.error(f"    VLA execution error: {e}")

    return success, steps_taken, replay_images


def run_oracle_per_subgoal(
    cfg: SubgoalEvalConfig,
    env,
    task_description: str,
    model,
    resize_size: int,
    initial_state,
    demo_actions: np.ndarray,
    subgoal_data: dict,
    processor=None,
    action_head=None,
    proprio_projector=None,
    noisy_action_projector=None,
    save_dir: Optional[str] = None,
) -> List[dict]:
    """Oracle per-subgoal evaluation for one episode.

    For every subgoal k and every command key (or just the configured key):
      - Replay demo actions to place the sim at the exact start frame of subgoal k
      - Run the VLA with that command key's prompt string
      - Record task success

    This means each (subgoal, command_key) pair gets its own independent rollout
    from the correct physical starting state.

    Episode structure (from results.json + steering_commands.json):
      subgoals     : ["reach for handle", "pull drawer open"]
      boundaries   : [70, 135]   ← last frame (inclusive) of each subgoal
      starts       : [0, 71]     ← first frame of each subgoal
      durations    : [71, 65]    ← demo frame count per subgoal
      allotted     : [142, 130]  ← eval budget (duration × slack_factor VLA queries)

    steering_commands.json adds per-subgoal:
      subgoal           : full verbose description (same as results.json)
      spatial_command   : directional description ("Move down, backward, right...")
      low_level_motion  : minimal motion words ("move down, backward, and right")
      outcome           : goal state ("gripper closed around handle")
      object_affordance : grasp hint ("hook fingers around handle from the front")

    Returns list of result dicts, one per (subgoal_idx, command_key) combination.
    """
    subgoals   = subgoal_data["subgoals"]
    boundaries = subgoal_data["boundaries"]
    n_subgoals = len(subgoals)

    # steering_commands indexed by subgoal_index
    steering_cmds = {
        sc["subgoal_index"]: sc
        for sc in subgoal_data.get("steering_commands", [])
    }

    # start frame and demo duration for each subgoal
    starts    = [0] + [boundaries[k] + 1 for k in range(n_subgoals - 1)]
    durations = [boundaries[k] - starts[k] + 1 for k in range(n_subgoals)]

    # which command keys to test
    # oracle_command_key can be:
    #   "all"                       → all STEERING_COMMAND_KEYS
    #   "subgoal"                   → single key
    #   "task,subgoal,low_level_motion,low_level_motion_with_numbers"  → comma-separated list
    if cfg.oracle_command_key == "all":
        keys_to_test = STEERING_COMMAND_KEYS
    else:
        keys_to_test = [k.strip() for k in cfg.oracle_command_key.split(",") if k.strip()]

    results = []

    for k in range(n_subgoals):
        start_frame    = starts[k]
        allotted_steps = max(int(durations[k] * cfg.oracle_slack_factor), 8)
        if cfg.oracle_max_steps > 0:
            allotted_steps = min(allotted_steps, cfg.oracle_max_steps)
        sc             = steering_cmds.get(k)

        logger.info(
            f"  Subgoal {k}/{n_subgoals-1}  "
            f"start_frame={start_frame}  demo_dur={durations[k]}  "
            f"allotted={allotted_steps} chunks  "
            f'canonical="{subgoals[k][:60]}"'
        )

        for cmd_key in keys_to_test:
            # Resolve prompt for this command key
            if sc and cmd_key in sc and sc[cmd_key]:
                prompt = sc[cmd_key]
            elif cmd_key == "subgoal":
                prompt = subgoals[k]   # fallback: results.json string
            else:
                logger.info(f"    [{cmd_key}] no steering_commands — skip")
                continue

            logger.info(f"    [{cmd_key}] \"{prompt}\"")

            # ── Fresh replay to start frame for this (subgoal, key) pair ──
            success, steps_taken, replay_images = _run_one_subgoal(
                cfg=cfg, env=env, model=model, resize_size=resize_size,
                initial_state=initial_state, demo_actions=demo_actions,
                start_frame=start_frame, allotted_steps=allotted_steps,
                prompt=prompt,
                processor=processor, action_head=action_head,
                proprio_projector=proprio_projector,
                noisy_action_projector=noisy_action_projector,
            )

            logger.info(f"      → success={success}  steps={steps_taken}")

            # ── Save video ────────────────────────────────────────────────
            if save_dir:
                slug = re.sub(r"[^a-z0-9]+", "_", prompt.lower()).strip("_")[:35]
                fname = f"sg{k:02d}_{cmd_key}_{slug}_{'success' if success else 'fail'}.mp4"
                vpath = os.path.join(save_dir, fname)
                os.makedirs(save_dir, exist_ok=True)
                try:
                    imageio.mimwrite(vpath, [np.asarray(x) for x in replay_images], fps=10)
                except Exception:
                    pass

            results.append({
                "subgoal_idx":   k,
                "command_key":   cmd_key,
                "prompt":        prompt,
                "subgoal":       subgoals[k],
                "start_frame":   start_frame,
                "demo_duration": durations[k],
                "allotted_steps": allotted_steps,
                "steps_taken":   steps_taken,
                "task_success":  success,
            })

    return results


# =============================================================================
# GPT-steered episode runner
# =============================================================================

def run_episode_gpt_subgoal(
    cfg: SubgoalEvalConfig,
    env,
    task_description: str,
    model,
    resize_size: int,
    gpt_client,
    precomputed_subgoals: Optional[List[str]] = None,
    initial_state=None,
    processor=None,
    action_head=None,
    proprio_projector=None,
    noisy_action_projector=None,
) -> Tuple[bool, List[np.ndarray], List[dict]]:
    """Run a closed-loop episode where GPT generates subgoal instructions."""
    env.reset()
    if initial_state is not None:
        obs = env.set_init_state(initial_state)
    else:
        obs = env.get_observation()

    max_steps = TASK_MAX_STEPS[cfg.task_suite_name]
    t = 0
    replay_images = []
    gpt_logs = []
    subgoal_history: List[str] = []
    current_subgoal: Optional[str] = None
    chunk_idx = 0
    success = False

    for _ in range(cfg.num_steps_wait):
        obs, _, _, _ = env.step(get_libero_dummy_action(cfg.model_family))
        t += 1

    try:
        while t < max_steps + cfg.num_steps_wait:
            observation, img = prepare_observation(obs, resize_size)
            replay_images.append(img)

            if chunk_idx % cfg.gpt_replan_every_n_chunks == 0:
                new_subgoal, parsed = query_gpt_subgoal(
                    client=gpt_client,
                    task_description=task_description,
                    obs_img=img,
                    obs_raw=obs,
                    prev_subgoal=current_subgoal,
                    subgoal_history=subgoal_history,
                    precomputed_subgoals=precomputed_subgoals if cfg.use_precomputed_subgoals else None,
                    gpt_model=cfg.gpt_model,
                )
                current_subgoal = new_subgoal
                subgoal_history.append(current_subgoal)
                gpt_logs.append({"chunk": chunk_idx, "step": t - cfg.num_steps_wait,
                                  "subgoal": current_subgoal, "parsed": parsed})
                logger.info(f"  [chunk {chunk_idx}] GPT subgoal: \"{current_subgoal}\"")
                if parsed.get("task_complete", False):
                    break

            effective_prompt = current_subgoal or task_description

            with torch.no_grad():
                actions = get_action(
                    cfg=cfg, model=model, obs=observation, task_label=effective_prompt,
                    processor=processor, action_head=action_head,
                    proprio_projector=proprio_projector,
                    noisy_action_projector=noisy_action_projector,
                    use_film=cfg.use_film,
                )

            chunk_idx += 1
            for action in actions:
                action = process_action(action, cfg.model_family)
                obs, _, done, _ = env.step(action.tolist())
                _, step_img = prepare_observation(obs, resize_size)
                replay_images.append(step_img)
                t += 1
                if done:
                    success = True
                    break
            if success:
                break

    except Exception as e:
        logger.error(f"GPT subgoal episode error: {e}")

    return success, replay_images, gpt_logs


# =============================================================================
# Baseline episode runner
# =============================================================================

def run_episode_openvla_only(
    cfg: SubgoalEvalConfig,
    env,
    task_description: str,
    model,
    resize_size: int,
    initial_state=None,
    processor=None,
    action_head=None,
    proprio_projector=None,
    noisy_action_projector=None,
) -> Tuple[bool, List[np.ndarray]]:
    """Baseline: run episode with full task description, no subgoal."""
    env.reset()
    if initial_state is not None:
        obs = env.set_init_state(initial_state)
    else:
        obs = env.get_observation()

    max_steps = TASK_MAX_STEPS[cfg.task_suite_name]
    action_queue = deque()
    t = 0
    replay_images = []
    success = False

    try:
        while t < max_steps + cfg.num_steps_wait:
            if t < cfg.num_steps_wait:
                obs, _, _, _ = env.step(get_libero_dummy_action(cfg.model_family))
                t += 1
                continue

            observation, img = prepare_observation(obs, resize_size)
            replay_images.append(img)

            if len(action_queue) == 0:
                with torch.no_grad():
                    actions = get_action(
                        cfg=cfg, model=model, obs=observation, task_label=task_description,
                        processor=processor, action_head=action_head,
                        proprio_projector=proprio_projector,
                        noisy_action_projector=noisy_action_projector,
                        use_film=cfg.use_film,
                    )
                action_queue.extend(actions)

            action = action_queue.popleft()
            action = process_action(action, cfg.model_family)
            obs, _, done, _ = env.step(action.tolist())
            if done:
                success = True
                break
            t += 1

    except Exception as e:
        logger.error(f"openvla_only episode error: {e}")

    return success, replay_images


# =============================================================================
# Main evaluation loop
# =============================================================================

def run_evaluation(cfg: SubgoalEvalConfig) -> None:
    set_seed_everywhere(cfg.seed)
    validate_config(cfg)

    model, action_head, proprio_projector, noisy_action_projector, processor = initialize_model(cfg)
    resize_size = get_image_resize_size(cfg)

    gpt_client = None
    if cfg.eval_mode in ("gpt_subgoal", "all"):
        from openai import OpenAI
        gpt_client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    os.makedirs(cfg.video_dir, exist_ok=True)
    os.makedirs(cfg.local_log_dir, exist_ok=True)

    modes = ["oracle_subgoal", "gpt_subgoal", "openvla_only"] if cfg.eval_mode == "all" else [cfg.eval_mode]

    # Set up logging
    ckpt_name = os.path.basename(cfg.pretrained_checkpoint.rstrip("/"))
    run_id = f"SUBGOAL-EVAL-{cfg.task_suite_name}-{ckpt_name}-{DATE_TIME}"
    log_path = os.path.join(cfg.local_log_dir, f"{run_id}.txt")
    log_file = open(log_path, "w")
    logger.info(f"Logging to {log_path}")

    # Initialize task suite
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[cfg.task_suite_name]()
    num_tasks = task_suite.n_tasks
    task_id_end = cfg.task_id_end if cfg.task_id_end > 0 else num_tasks

    # ── Build RLDS episode map for oracle mode ────────────────────────────────
    episode_map: Dict[str, list] = {}
    if "oracle_subgoal" in modes:
        logger.info(f"Building RLDS episode map from {cfg.rlds_results_dir} ...")
        episode_map = build_episode_map(cfg.rlds_results_dir, task_suite)
        logger.info(f"  Found {sum(len(v) for v in episode_map.values())} episodes "
                    f"across {len(episode_map)} tasks.")

    # ── Aggregate results ─────────────────────────────────────────────────────
    # oracle_sg_stats keyed by "sg{k}_{cmd_key}" → {"total": N, "successes": M}
    oracle_sg_stats: Dict[str, dict] = defaultdict(lambda: {"total": 0, "successes": 0})
    oracle_task_stats: Dict[int, dict] = {}

    all_results: Dict[str, dict] = {m: {"total_episodes": 0, "total_successes": 0} for m in modes}

    # ── Loop over tasks ───────────────────────────────────────────────────────
    for task_id in tqdm.tqdm(range(cfg.task_id_start, task_id_end), desc="Tasks"):
        task = task_suite.get_task(task_id)
        initial_states = task_suite.get_task_init_states(task_id)
        env, task_description = get_libero_env(task, cfg.model_family, resolution=cfg.env_img_res)
        if cfg.custom_task_description:
            task_description = cfg.custom_task_description

        log_message(f"\n{'='*60}", log_file)
        log_message(f"Task {task_id}: {task_description}", log_file)
        log_message(f"{'='*60}", log_file)

        oracle_task_stats[task_id] = {"task_description": task_description,
                                       "subgoal_results": []}

        # ── Oracle mode ───────────────────────────────────────────────────────
        if "oracle_subgoal" in modes:
            task_episodes = episode_map.get(task_description, [])
            if not task_episodes:
                log_message(f"  [oracle] No RLDS episodes found for: \"{task_description}\"", log_file)
            else:
                log_message(f"  [oracle] {len(task_episodes)} RLDS episodes", log_file)

            num_trials = min(cfg.num_trials_per_task, len(task_episodes), len(initial_states))

            for ep_ordinal in range(num_trials):
                ep_info = task_episodes[ep_ordinal]
                ep_idx = ep_info["episode_index"]
                subgoal_data = ep_info["subgoal_data"]
                initial_state = initial_states[ep_ordinal]

                log_message(f"\n  [oracle] ep_ordinal={ep_ordinal}, rlds_ep={ep_idx}", log_file)
                log_message(f"  subgoals: {subgoal_data['subgoals']}", log_file)

                # Load demo actions
                try:
                    demo_actions = load_rlds_actions(
                        cfg.rlds_dataset, cfg.rlds_data_dir, ep_idx
                    )
                except Exception as e:
                    log_message(f"  [oracle] Failed to load RLDS ep {ep_idx}: {e}", log_file)
                    continue

                # Run per-subgoal oracle eval
                save_dir = os.path.join(
                    cfg.video_dir, "oracle_subgoal",
                    f"task{task_id}_ep{ep_ordinal}"
                )
                sg_results = run_oracle_per_subgoal(
                    cfg=cfg, env=env, task_description=task_description,
                    model=model, resize_size=resize_size,
                    initial_state=initial_state, demo_actions=demo_actions,
                    subgoal_data=subgoal_data,
                    processor=processor, action_head=action_head,
                    proprio_projector=proprio_projector,
                    noisy_action_projector=noisy_action_projector,
                    save_dir=save_dir,
                )

                oracle_task_stats[task_id]["subgoal_results"].extend(sg_results)

                for r in sg_results:
                    stat_key = f"sg{r['subgoal_idx']}_{r['command_key']}"
                    oracle_sg_stats[stat_key]["total"] += 1
                    if r["task_success"]:
                        oracle_sg_stats[stat_key]["successes"] += 1

                    log_message(
                        f"    SG{r['subgoal_idx']} [{r['command_key']}] "
                        f"\"{r['prompt'][:50]}\": "
                        f"success={r['task_success']}, steps={r['steps_taken']}/{r['allotted_steps']}",
                        log_file,
                    )

        # ── GPT subgoal mode ──────────────────────────────────────────────────
        if "gpt_subgoal" in modes:
            # Load precomputed subgoals from any RLDS episode for this task
            task_episodes = episode_map.get(task_description, [])
            precomputed_subgoals = task_episodes[0]["subgoal_data"]["subgoals"] if task_episodes else None

            num_trials = min(cfg.num_trials_per_task, len(initial_states))
            for ep_idx in range(num_trials):
                initial_state = initial_states[ep_idx]
                success, replay_images, gpt_logs = run_episode_gpt_subgoal(
                    cfg, env, task_description, model, resize_size, gpt_client,
                    precomputed_subgoals, initial_state,
                    processor, action_head, proprio_projector, noisy_action_projector,
                )
                all_results["gpt_subgoal"]["total_episodes"] += 1
                if success:
                    all_results["gpt_subgoal"]["total_successes"] += 1

                suffix = "success" if success else "fail"
                vpath = os.path.join(cfg.video_dir, "gpt_subgoal",
                                     f"task{task_id}_ep{ep_idx}_{suffix}.mp4")
                os.makedirs(os.path.dirname(vpath), exist_ok=True)
                try:
                    imageio.mimwrite(vpath, [np.asarray(x) for x in replay_images], fps=10)
                except Exception:
                    pass
                log_message(f"  [gpt_subgoal] ep{ep_idx}: success={success}, "
                             f"subgoals={[g['subgoal'] for g in gpt_logs]}", log_file)

        # ── Baseline mode ─────────────────────────────────────────────────────
        if "openvla_only" in modes:
            num_trials = min(cfg.num_trials_per_task, len(initial_states))
            for ep_idx in range(num_trials):
                initial_state = initial_states[ep_idx]
                success, replay_images = run_episode_openvla_only(
                    cfg, env, task_description, model, resize_size,
                    initial_state, processor, action_head, proprio_projector, noisy_action_projector,
                )
                all_results["openvla_only"]["total_episodes"] += 1
                if success:
                    all_results["openvla_only"]["total_successes"] += 1

                suffix = "success" if success else "fail"
                vpath = os.path.join(cfg.video_dir, "openvla_only",
                                     f"task{task_id}_ep{ep_idx}_{suffix}.mp4")
                os.makedirs(os.path.dirname(vpath), exist_ok=True)
                try:
                    imageio.mimwrite(vpath, [np.asarray(x) for x in replay_images], fps=10)
                except Exception:
                    pass
                log_message(f"  [openvla_only] ep{ep_idx}: success={success}", log_file)

        env.close()

    # ── Final summary ─────────────────────────────────────────────────────────
    log_message(f"\n{'='*60}", log_file)
    log_message("FINAL RESULTS", log_file)
    log_message(f"{'='*60}", log_file)

    if "oracle_subgoal" in modes:
        log_message("\n[oracle_subgoal] Per-subgoal success rate by command key:", log_file)
        # Group by command_key for a clean table
        from collections import OrderedDict
        by_key: Dict[str, dict] = OrderedDict()
        for stat_key in sorted(oracle_sg_stats.keys()):
            # stat_key = "sg{k}_{cmd_key}"
            parts = stat_key.split("_", 1)  # ["sg0", "low_level_motion"]
            sg_label, cmd_key = parts[0], parts[1] if len(parts) > 1 else "subgoal"
            by_key.setdefault(cmd_key, {})[sg_label] = oracle_sg_stats[stat_key]
        for cmd_key, sg_dict in by_key.items():
            log_message(f"\n  [{cmd_key}]", log_file)
            for sg_label in sorted(sg_dict.keys()):
                s = sg_dict[sg_label]
                rate = s["successes"] / s["total"] if s["total"] > 0 else 0.0
                log_message(f"    {sg_label}: {s['successes']}/{s['total']} = {rate:.3f}", log_file)

    for mode in ["gpt_subgoal", "openvla_only"]:
        if mode in modes:
            r = all_results[mode]
            rate = r["total_successes"] / r["total_episodes"] if r["total_episodes"] > 0 else 0.0
            log_message(f"\n[{mode}] {r['total_successes']}/{r['total_episodes']} = {rate:.3f}", log_file)

    # Save results JSON
    results_path = os.path.join(cfg.local_log_dir, f"{run_id}_results.json")
    with open(results_path, "w") as f:
        json.dump({
            "oracle_sg_stats": dict(oracle_sg_stats),
            "oracle_task_stats": oracle_task_stats,
            "gpt_results": all_results.get("gpt_subgoal", {}),
            "openvla_results": all_results.get("openvla_only", {}),
        }, f, indent=2)
    log_message(f"\nResults saved to {results_path}", log_file)
    log_file.close()


# =============================================================================
# Entry point
# =============================================================================

@draccus.wrap()
def main(cfg: SubgoalEvalConfig) -> None:
    run_evaluation(cfg)


if __name__ == "__main__":
    main()
