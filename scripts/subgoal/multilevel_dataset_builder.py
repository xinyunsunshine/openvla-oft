"""
multilevel_dataset_builder.py

TFDS dataset builder that creates a LIBERO RLDS dataset where language_instruction
is a JSON dict containing all 6 instruction levels for each frame's subgoal:

    language_instruction = '{
        "task": "put the mushroom in the pot",
        "subgoal": "move the mushroom above the pot",
        "spatial_command": "position the mushroom directly above the pot opening",
        "low_level_motion": "move up and forward",
        "low_level_motion_with_numbers": "move up 8m, move forward 5m",
        "outcome": "the mushroom is directly over the pot opening",
        "object_affordance": "grasp mushroom cap with pinch grip"
    }'

Reads:
  - An existing LIBERO RLDS source dataset
  - A directory of results.json files (one per episode, from generate_subgoal_variations.py)
  - A directory of steering_commands.json files (from generate_steering_commands.py)

The results.json and steering_commands.json must be co-located in the same directory
(one dir per episode, e.g. outputs/ep00000/results.json + steering_commands.json).
Episodes are matched by sorted order to the source dataset.

Dataset name: libero_subgoal_multilevel
Use MultiLevelBatchTransform when loading, sampling one key at batch time.

Build:
    cd openvla-oft/scripts/subgoal
    tfds build --datasets=multilevel_dataset_builder \\
        --data_dir=/proj/.../rlds_datasets
"""

import json
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds


MULTILEVEL_KEYS = [
    "task",
    "subgoal",
    "spatial_command",
    "low_level_motion",
    "low_level_motion_with_numbers",
    "outcome",
    "object_affordance",
]


class MultilevelDatasetConfig(tfds.core.BuilderConfig):
    def __init__(
        self,
        *,
        source_dataset: str,
        source_data_dir: str,
        results_dir: str,
        max_trajs_per_task: int = 0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.source_dataset = source_dataset
        self.source_data_dir = source_data_dir
        # Directory containing per-episode subdirs, each with results.json and
        # steering_commands.json. Episodes are matched by sorted dir order.
        self.results_dir = results_dir
        # If > 0, keep at most this many episodes per task (first N seen)
        self.max_trajs_per_task = max_trajs_per_task


class MultilevelDataset(tfds.core.GeneratorBasedBuilder):
    """LIBERO dataset with per-step multi-level instruction dicts.

    language_instruction per step is a JSON-encoded dict with keys:
        task, subgoal, spatial_command, low_level_motion,
        low_level_motion_with_numbers, outcome, object_affordance

    Use MultiLevelBatchTransform (prismatic.vla.datasets) when loading.
    """

    VERSION = tfds.core.Version("1.0.0")
    RELEASE_NOTES = {"1.0.0": "Initial release with multi-level instruction labels."}

    BUILDER_CONFIGS = [
        MultilevelDatasetConfig(
            name="libero_goal_multilevel",
            source_dataset="libero_langact",
            source_data_dir="/proj/inf-scaling/SWE_sim/modified_libero_rlds_new",
            results_dir="/proj/inf-scaling/zwhong/idan/vla-rl/outputs/subgoal_results",
        ),
    ]

    def _info(self) -> tfds.core.DatasetInfo:
        return self.dataset_info_from_configs(
            features=tfds.features.FeaturesDict(
                {
                    "steps": tfds.features.Dataset(
                        {
                            "observation": tfds.features.FeaturesDict(
                                {
                                    "image": tfds.features.Image(
                                        shape=(256, 256, 3),
                                        dtype=np.uint8,
                                        encoding_format="jpeg",
                                        doc="Main camera RGB observation.",
                                    ),
                                    "wrist_image": tfds.features.Image(
                                        shape=(256, 256, 3),
                                        dtype=np.uint8,
                                        encoding_format="jpeg",
                                        doc="Wrist camera RGB observation.",
                                    ),
                                    "state": tfds.features.Tensor(
                                        shape=(8,),
                                        dtype=np.float32,
                                        doc="Robot EEF state (6D pose, 2D gripper).",
                                    ),
                                    "joint_state": tfds.features.Tensor(
                                        shape=(7,),
                                        dtype=np.float32,
                                        doc="Robot joint angles.",
                                    ),
                                }
                            ),
                            "action": tfds.features.Tensor(
                                shape=(7,),
                                dtype=np.float32,
                                doc="Robot EEF action.",
                            ),
                            "discount": tfds.features.Scalar(dtype=np.float32),
                            "reward": tfds.features.Scalar(dtype=np.float32),
                            "is_first": tfds.features.Scalar(dtype=np.bool_),
                            "is_last": tfds.features.Scalar(dtype=np.bool_),
                            "is_terminal": tfds.features.Scalar(dtype=np.bool_),
                            "language_instruction": tfds.features.Text(
                                doc=(
                                    "JSON-encoded dict with all instruction levels for this step. "
                                    "Keys: task, subgoal, spatial_command, low_level_motion, "
                                    "low_level_motion_with_numbers, outcome, object_affordance. "
                                    "Use MultiLevelBatchTransform to sample one key at batch time."
                                ),
                            ),
                        }
                    ),
                    "episode_metadata": tfds.features.FeaturesDict(
                        {
                            "file_path": tfds.features.Text(),
                            "subgoal_index": tfds.features.Scalar(
                                dtype=np.int32,
                                doc="Subgoal index for this episode's first step.",
                            ),
                            "n_subgoals": tfds.features.Scalar(
                                dtype=np.int32,
                                doc="Total number of subgoals in this episode.",
                            ),
                        }
                    ),
                }
            )
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        return {"train": self._generate_examples()}

    def _load_episode_data(self, results_dir: str) -> List[Dict]:
        """
        Load all (results.json, steering_commands.json) pairs from results_dir.
        Each episode dir should contain both files.
        Returns list of merged dicts indexed by sorted episode order.
        """
        results_path = Path(results_dir)

        # Find all results.json files
        results_files = sorted(results_path.rglob("results.json"))
        if not results_files:
            raise FileNotFoundError(f"No results.json files found in {results_dir}")

        episode_data = []
        for results_file in results_files:
            ep_dir = results_file.parent
            steering_file = ep_dir / "steering_commands.json"

            with open(results_file) as f:
                results = json.load(f)

            steering = None
            if steering_file.exists():
                with open(steering_file) as f:
                    steering = json.load(f)
            else:
                print(f"  Warning: no steering_commands.json in {ep_dir}, "
                      f"will use subgoal-only for affected fields")

            episode_data.append({"results": results, "steering": steering})

        print(f"  Loaded {len(episode_data)} episodes from {results_dir} "
              f"({sum(1 for e in episode_data if e['steering'] is not None)} with steering)")
        return episode_data

    def _build_steering_index(self, steering: Optional[Dict]) -> Dict[int, Dict]:
        """
        Returns dict mapping subgoal_index → steering command fields.
        """
        if steering is None:
            return {}
        return {
            cmd["subgoal_index"]: cmd
            for cmd in steering.get("steering_commands", [])
        }

    def _get_per_frame_multilevel(
        self,
        results: Dict,
        steering_index: Dict[int, Dict],
        t: int,
    ) -> str:
        """
        Returns JSON-encoded dict of all instruction levels for frame t.
        Falls back gracefully when steering fields are missing.
        """
        per_frame = results["per_frame_labels"]
        sg_idx = per_frame[t] if t < len(per_frame) else per_frame[-1]
        subgoal = results["subgoals"][sg_idx]
        task = results.get("task", "")

        cmd = steering_index.get(sg_idx, {})

        instruction_dict = {
            "task": task,
            "subgoal": subgoal,
            "spatial_command": cmd.get("spatial_command", subgoal),
            "low_level_motion": cmd.get("low_level_motion", subgoal),
            "low_level_motion_with_numbers": cmd.get("low_level_motion_with_numbers", subgoal),
            "outcome": cmd.get("outcome", subgoal),
            "object_affordance": cmd.get("object_affordance", subgoal),
        }

        return json.dumps(instruction_dict)

    def _generate_examples(self) -> Iterator[Tuple[str, Any]]:
        config = self.builder_config
        assert isinstance(config, MultilevelDatasetConfig)

        tf.config.set_visible_devices([], "GPU")

        # Load episode data (results + steering pairs)
        episode_data = self._load_episode_data(config.results_dir)

        # Load source RLDS dataset
        source_builder = tfds.builder(
            config.source_dataset,
            data_dir=config.source_data_dir,
        )
        source_ds = source_builder.as_dataset(split="train")

        ep_idx = 0
        ds_iter = iter(source_ds)

        while True:
            try:
                episode = next(ds_iter)
            except StopIteration:
                break
            except tf.errors.DataLossError:
                print(f"  Skipping corrupted shard")
                continue

            try:
                steps = list(episode["steps"])
            except tf.errors.DataLossError:
                print(f"  Skipping episode {ep_idx}: corrupted step data")
                ep_idx += 1
                continue

            if len(steps) == 0:
                ep_idx += 1
                continue

            if ep_idx >= len(episode_data):
                print(f"  Warning: more episodes in source ({ep_idx}) than episode dirs "
                      f"({len(episode_data)}). Stopping.")
                break

            ep = episode_data[ep_idx]
            results = ep["results"]

            # Per-task limit
            if config.max_trajs_per_task > 0:
                task_key = results.get("task", "")
                task_counts = getattr(self, "_task_counts", {})
                if task_counts.get(task_key, 0) >= config.max_trajs_per_task:
                    ep_idx += 1
                    continue
                task_counts[task_key] = task_counts.get(task_key, 0) + 1
                self._task_counts = task_counts

            steering_index = self._build_steering_index(ep["steering"])

            new_steps = []
            for t, step in enumerate(steps):
                lang = self._get_per_frame_multilevel(results, steering_index, t)
                new_steps.append(
                    {
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
                        "language_instruction": lang,
                    }
                )

            file_path = ""
            if "episode_metadata" in episode and "file_path" in episode["episode_metadata"]:
                file_path = episode["episode_metadata"]["file_path"].numpy().decode("utf-8")

            first_sg = results["per_frame_labels"][0] if results["per_frame_labels"] else 0
            n_sgs = len(results["subgoals"])

            sample = {
                "steps": new_steps,
                "episode_metadata": {
                    "file_path": file_path,
                    "subgoal_index": np.int32(first_sg),
                    "n_subgoals": np.int32(n_sgs),
                },
            }

            key = f"{config.source_dataset}::ep{ep_idx:05d}"
            yield key, sample
            ep_idx += 1

        print(f"  Processed {ep_idx} episodes")
