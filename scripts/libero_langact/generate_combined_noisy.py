"""
Generate a combined LIBERO RLDS dataset with noisy verbose language-action prompts.

Combines all 4 LIBERO suites (spatial, object, goal, libero_10 = 40 tasks),
samples N trajectories per task, and adds 10% proportional Gaussian noise to
the numeric values in the verbose language-action text. This prevents the model
from memorizing exact magnitude-to-action mappings and improves generalization.

Noise is applied proportionally to each dimension of the aggregated 7D
language-action vector BEFORE text conversion:
  noisy_val = val + N(0, noise_frac * |val|)
  Default noise_frac = 0.1 (10%)

Usage:
    /proj/inf-scaling/zwhong/idan/vla-rl/lap/.venv/bin/python \
        scripts/libero_langact/generate_combined_noisy.py \
        --data_dir /proj/inf-scaling/SWE_sim/modified_libero_rlds_new \
        --output_dir /proj/inf-scaling/SWE_sim/modified_libero_rlds_new \
        --trajs_per_task 5 \
        --seed 42
"""

import argparse
import json
import os
import shutil
import sys
from collections import defaultdict

import numpy as np
import tensorflow as tf

tf.config.set_visible_devices([], "GPU")
import tensorflow_datasets as tfds

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

_lap_src = os.path.join(os.path.dirname(__file__), "..", "..", "..", "lap", "src")
if os.path.isdir(_lap_src):
    sys.path.insert(0, os.path.abspath(_lap_src))

from langact_utils import (
    aggregate_language_actions,
    compute_language_actions,
    summarize_numeric_actions,
)

try:
    from lap.datasets.utils.transforms import libero_dataset_transform as _lap_transform
    _USE_LAP = True
except ImportError:
    _USE_LAP = False


SOURCE_DATASETS = [
    "libero_spatial_no_noops",
    "libero_object_no_noops",
    "libero_goal_no_noops",
    "libero_10_no_noops",
]


def compute_lang_action_vectors(states, actions):
    """Compute 7D language-action vectors using LAP or numpy fallback."""
    if _USE_LAP:
        traj_dict = {
            "observation": {"state": tf.constant(states, dtype=tf.float32)},
            "action": tf.constant(actions, dtype=tf.float32),
            "language_instruction": tf.repeat("", len(states)),
        }
        result = _lap_transform(traj_dict)
        return result["language_action"].numpy()
    else:
        return compute_language_actions(states, actions)


def lang_actions_to_verbose_text(lang_actions, include_rotation=True):
    """Convert aggregated 7D language-action vectors to verbose text."""
    texts = []
    for t in range(len(lang_actions)):
        text = summarize_numeric_actions(
            lang_actions[t], sum_decimal="1f", include_rotation=include_rotation
        )
        texts.append(text or "close gripper")
    return texts


def lang_actions_to_nonum_text(lang_actions, include_rotation=True):
    """Convert aggregated 7D language-action vectors to no-number text."""
    texts = []
    for t in range(len(lang_actions)):
        text = summarize_numeric_actions(
            lang_actions[t], sum_decimal="no_number", include_rotation=include_rotation
        )
        texts.append(text or "close gripper")
    return texts


def add_text_level_noise(
    text: str,
    rng: np.random.Generator,
    noise_frac: float = 0.1,
    keep_prob: float = 0.3,
    trans_floor_cm: float = 0.2,
    rot_floor_deg: float = 3.0,
) -> str:
    """Add noise to numeric values in a verbose language-action text string.

    Each numeric value independently has:
      - keep_prob chance of being unchanged
      - (1-keep_prob) chance of noise: N(0, max(floor, noise_frac * |value|))

    Translation values are in cm (1 decimal), rotation in degrees (integer).
    Gripper text (open/close) is never changed.

    Args:
        text: Verbose language-action text, e.g. "move forward 3.0 cm, tilt right 10 degrees, close gripper"
        rng: numpy random generator.
        noise_frac: Proportional noise fraction (default 0.1 = 10%).
        keep_prob: Probability of keeping a value unchanged (default 0.3 = 30%).
        trans_floor_cm: Minimum noise std for translation in cm (default 0.2).
        rot_floor_deg: Minimum noise std for rotation in degrees (default 3.0).

    Returns:
        Noisy text string.
    """
    import re

    def _perturb_translation(match):
        direction = match.group(1)  # e.g. "move forward"
        value = float(match.group(2))  # e.g. 3.0
        unit = match.group(3)  # "cm"
        if rng.random() < keep_prob:
            return match.group(0)  # keep unchanged
        std = max(trans_floor_cm, noise_frac * value)
        noisy_val = max(0.1, value + rng.normal(0, std))  # clamp to >= 0.1 cm
        return f"{direction} {noisy_val:.1f} {unit}"

    def _perturb_rotation(match):
        direction = match.group(1)  # e.g. "tilt right"
        value = int(match.group(2))  # e.g. 10
        unit = match.group(3)  # "degrees"
        if rng.random() < keep_prob:
            return match.group(0)  # keep unchanged
        std = max(rot_floor_deg, noise_frac * value)
        noisy_val = max(1, int(round(value + rng.normal(0, std))))  # clamp to >= 1
        return f"{direction} {noisy_val} {unit}"

    # Apply noise to translation values
    text = re.sub(
        r"(move\s+(?:forward|back|left|right|up|down))\s+([\d.]+)\s*(cm)",
        _perturb_translation,
        text,
    )

    # Apply noise to rotation values
    text = re.sub(
        r"((?:tilt\s+(?:left|right|back|forward))|(?:rotate\s+(?:clockwise|counterclockwise)))\s+(\d+)\s*(degrees)",
        _perturb_rotation,
        text,
    )

    return text


def lang_actions_to_noisy_verbose_text(
    lang_actions,
    rng: np.random.Generator,
    include_rotation: bool = True,
    noise_frac: float = 0.1,
    keep_prob: float = 0.3,
    trans_floor_cm: float = 0.2,
    rot_floor_deg: float = 3.0,
) -> list[str]:
    """Convert aggregated 7D language-action vectors to noisy verbose text.

    First converts to clean text, then applies text-level noise to each numeric value.
    """
    texts = []
    for t in range(len(lang_actions)):
        clean_text = summarize_numeric_actions(
            lang_actions[t], sum_decimal="1f", include_rotation=include_rotation
        )
        clean_text = clean_text or "close gripper"
        noisy_text = add_text_level_noise(
            clean_text, rng,
            noise_frac=noise_frac,
            keep_prob=keep_prob,
            trans_floor_cm=trans_floor_cm,
            rot_floor_deg=rot_floor_deg,
        )
        texts.append(noisy_text)
    return texts


class CombinedNoisyBuilder(tfds.core.GeneratorBasedBuilder):
    """TFDS builder for combined noisy verbose language-action dataset."""

    VERSION = tfds.core.Version("1.0.0")
    RELEASE_NOTES = {"1.0.0": "Combined 40-task dataset with noisy verbose language-actions."}

    def __init__(self, *, episodes_data, **kwargs):
        self._episodes_data = episodes_data
        super().__init__(**kwargs)

    def _info(self):
        return tfds.core.DatasetInfo(
            builder=self,
            features=tfds.features.FeaturesDict({
                "steps": tfds.features.Dataset({
                    "observation": tfds.features.FeaturesDict({
                        "image": tfds.features.Image(shape=(256, 256, 3), dtype=np.uint8, encoding_format="jpeg"),
                        "wrist_image": tfds.features.Image(shape=(256, 256, 3), dtype=np.uint8, encoding_format="jpeg"),
                        "state": tfds.features.Tensor(shape=(8,), dtype=np.float32),
                        "joint_state": tfds.features.Tensor(shape=(7,), dtype=np.float32),
                    }),
                    "action": tfds.features.Tensor(shape=(7,), dtype=np.float32),
                    "discount": tfds.features.Scalar(dtype=np.float32),
                    "reward": tfds.features.Scalar(dtype=np.float32),
                    "is_first": tfds.features.Scalar(dtype=np.bool_),
                    "is_last": tfds.features.Scalar(dtype=np.bool_),
                    "is_terminal": tfds.features.Scalar(dtype=np.bool_),
                    "language_instruction": tfds.features.Text(),
                }),
                "episode_metadata": tfds.features.FeaturesDict({
                    "file_path": tfds.features.Text(),
                }),
            })
        )

    def _split_generators(self, dl_manager):
        return {"train": self._generate_examples()}

    def _generate_examples(self):
        for key, sample in self._episodes_data:
            yield key, sample


def main():
    parser = argparse.ArgumentParser(description="Generate combined noisy verbose language-action dataset.")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--output_name", type=str, default="libero_combined_langact_verbose_noisy")
    parser.add_argument("--format", type=str, default="verbose", choices=["verbose", "nonum"],
                        help="Language-action format. 'verbose' has numbers, 'nonum' has directions only.")
    parser.add_argument("--trajs_per_task", type=int, default=5)
    parser.add_argument("--noise_frac", type=float, default=0.1,
                        help="Proportional noise fraction. Default: 0.1 (10%%).")
    parser.add_argument("--keep_prob", type=float, default=0.3,
                        help="Probability of keeping each value unchanged. Default: 0.3 (30%%).")
    parser.add_argument("--trans_floor_cm", type=float, default=0.2,
                        help="Minimum noise std for translation in cm. Default: 0.2.")
    parser.add_argument("--rot_floor_deg", type=float, default=3.0,
                        help="Minimum noise std for rotation in degrees. Default: 3.0.")
    parser.add_argument("--window_size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no_noise", action="store_true", help="Disable noise (for comparison).")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    if args.format == "nonum":
        noise_desc = "N/A (no numbers to perturb)"
    elif args.no_noise:
        noise_desc = "DISABLED"
    else:
        noise_desc = (
            f"{args.noise_frac*100:.0f}% proportional, {args.keep_prob*100:.0f}% unchanged, "
            f"floor: {args.trans_floor_cm}cm / {args.rot_floor_deg}deg"
        )
    print(f"Generating: {args.output_name}")
    print(f"  Source datasets: {SOURCE_DATASETS}")
    print(f"  Format: {args.format}")
    print(f"  Trajs per task: {args.trajs_per_task}")
    print(f"  Noise: {noise_desc}")
    print(f"  Window: {args.window_size} steps")
    print(f"  Seed: {args.seed}")

    # Phase 1: Load all source datasets, group by task, sample trajectories
    all_episodes = []
    total_tasks = 0

    for ds_name in SOURCE_DATASETS:
        print(f"\nLoading {ds_name}...")
        builder = tfds.builder(ds_name, data_dir=args.data_dir)
        ds = builder.as_dataset(split="train")

        task_episodes = defaultdict(list)
        for episode in ds:
            steps = list(episode["steps"])
            if not steps:
                continue
            task_desc = steps[0]["language_instruction"].numpy().decode()
            task_episodes[task_desc].append((episode, steps))

        print(f"  {len(task_episodes)} tasks, {sum(len(v) for v in task_episodes.values())} total episodes")

        for task_desc, episodes in sorted(task_episodes.items()):
            n_sample = min(args.trajs_per_task, len(episodes))
            indices = rng.choice(len(episodes), size=n_sample, replace=False)
            sampled = [episodes[i] for i in sorted(indices)]
            total_tasks += 1

            for ep_episode, ep_steps in sampled:
                all_episodes.append({
                    "source_ds": ds_name,
                    "task_desc": task_desc,
                    "episode": ep_episode,
                    "steps": ep_steps,
                })

            print(f"    {task_desc}: {len(episodes)} avail, sampled {n_sample}")

    print(f"\nTotal: {total_tasks} tasks, {len(all_episodes)} episodes")

    # Phase 2: Compute language-actions with noise and build TFDS data
    episodes_data = []
    for ep_idx, ep_data in enumerate(all_episodes):
        steps = ep_data["steps"]
        states = np.stack([s["observation"]["state"].numpy() for s in steps], axis=0)
        actions = np.stack([s["action"].numpy() for s in steps], axis=0)

        lang_actions = compute_lang_action_vectors(states, actions)
        aggregated = aggregate_language_actions(lang_actions, window_size=args.window_size)

        if args.format == "nonum":
            texts = lang_actions_to_nonum_text(aggregated, include_rotation=True)
        elif args.no_noise:
            texts = lang_actions_to_verbose_text(aggregated, include_rotation=True)
        else:
            texts = lang_actions_to_noisy_verbose_text(
                aggregated, rng, include_rotation=True,
                noise_frac=args.noise_frac,
                keep_prob=args.keep_prob,
                trans_floor_cm=args.trans_floor_cm,
                rot_floor_deg=args.rot_floor_deg,
            )

        new_steps = []
        for t, step in enumerate(steps):
            new_steps.append({
                "observation": {
                    "image": step["observation"]["image"].numpy(),
                    "wrist_image": step["observation"]["wrist_image"].numpy(),
                    "state": step["observation"]["state"].numpy(),
                    "joint_state": step["observation"]["joint_state"].numpy(),
                },
                "action": step["action"].numpy(),
                "discount": float(step["discount"].numpy()),
                "reward": float(step["reward"].numpy()),
                "is_first": bool(step["is_first"].numpy()),
                "is_last": bool(step["is_last"].numpy()),
                "is_terminal": bool(step["is_terminal"].numpy()),
                "language_instruction": texts[t],
            })

        file_path = ""
        if "episode_metadata" in ep_data["episode"] and "file_path" in ep_data["episode"]["episode_metadata"]:
            file_path = ep_data["episode"]["episode_metadata"]["file_path"].numpy().decode()

        sample = {
            "steps": new_steps,
            "episode_metadata": {"file_path": f"{ep_data['source_ds']}::{file_path}"},
        }

        key = f"{ep_data['source_ds']}::{ep_data['task_desc'][:40]}::ep{ep_idx:05d}"
        episodes_data.append((key, sample))

        if (ep_idx + 1) % 20 == 0:
            print(f"  Processed {ep_idx + 1}/{len(all_episodes)} episodes")

    print(f"\nWriting TFDS dataset...")

    # Phase 3: Write TFDS dataset
    config = tfds.core.BuilderConfig(
        name="default", version=tfds.core.Version("1.0.0"),
        description=f"Combined 40-task LIBERO with text-level noisy verbose language-actions "
                    f"({args.noise_frac*100:.0f}% noise, {args.keep_prob*100:.0f}% kept, "
                    f"floor: {args.trans_floor_cm}cm/{args.rot_floor_deg}deg, "
                    f"{args.trajs_per_task} trajs/task)",
    )

    out_path = os.path.join(args.output_dir, args.output_name)
    builder = CombinedNoisyBuilder(
        episodes_data=episodes_data,
        config=config,
        data_dir=out_path,
    )
    builder.download_and_prepare()

    # Restructure from TFDS nested layout to flat layout
    tfds_dir = os.path.join(out_path, "combined_noisy_builder", "default", "1.0.0")
    target_dir = os.path.join(out_path, "1.0.0")
    if os.path.exists(tfds_dir) and not os.path.exists(target_dir):
        os.makedirs(target_dir, exist_ok=True)
        for f in os.listdir(tfds_dir):
            shutil.move(os.path.join(tfds_dir, f), os.path.join(target_dir, f))
        shutil.rmtree(os.path.join(out_path, "combined_noisy_builder"))
        shutil.rmtree(os.path.join(out_path, "downloads"), ignore_errors=True)

        builder_name = "combined_noisy_builder"
        for f in os.listdir(target_dir):
            if f.startswith(builder_name):
                new_name = f.replace(builder_name, args.output_name, 1)
                os.rename(os.path.join(target_dir, f), os.path.join(target_dir, new_name))

        info_path = os.path.join(target_dir, "dataset_info.json")
        if os.path.exists(info_path):
            with open(info_path, "r") as f:
                info = json.load(f)
            info["name"] = args.output_name
            with open(info_path, "w") as f:
                json.dump(info, f, indent=2)

    print(f"\nDone! Dataset written to: {out_path}")
    print(f"  {len(episodes_data)} episodes from {total_tasks} tasks")
    print(f"  Use --dataset_name {args.output_name} for training")


if __name__ == "__main__":
    main()
