"""
subgoal_dataset_builder.py

TFDS dataset builder that creates a LIBERO RLDS dataset where language_instruction
is replaced with per-frame subgoal labels (with all variation phrasings encoded as JSON).

Reads:
  - An existing LIBERO RLDS source dataset
  - A directory of results.json files (one per episode, from the subgoal pipeline)
  - Optionally, steering_commands.json files (one per episode, next to each results.json)

Writes a new RLDS dataset where each step's language_instruction is a JSON-encoded
list of all phrasings for that frame's subgoal:

    language_instruction = '["canonical phrasing", "alt 1", "alt 2", ...]'

If steering_commands.json is present, the list is extended with multilevel commands
(spatial_command, low_level_motion, low_level_motion_with_numbers) for the active
subgoal. The high-level task description is intentionally excluded.

This single field supports both training modes:
  - Canonical model: always decodes [0] (the canonical phrasing)
  - Variation model: decodes and samples randomly at batch time (uniform over all levels)

The results.json files are matched to episodes by episode index (ep_idx order in
the source dataset). Run generate_subgoal_variations.py first to populate
subgoal_variations in each results.json.

Build:
    cd openvla-oft/scripts/subgoal
    tfds build --datasets=subgoal_dataset_builder \\
        --config=libero_goal_subgoal \\
        --data_dir=/proj/.../rlds_datasets

Or use build_subgoal_dataset.sh.
"""

import json
from pathlib import Path
from typing import Any, Dict, Iterator, List, Tuple

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds


class SubgoalDatasetConfig(tfds.core.BuilderConfig):
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
        # Directory containing results.json files (one per episode, named ep00000.json etc.
        # or discovered by sorted order to match source dataset episode order)
        self.results_dir = results_dir
        # If > 0, keep at most this many episodes per task (first N seen)
        self.max_trajs_per_task = max_trajs_per_task


class SubgoalDataset(tfds.core.GeneratorBasedBuilder):
    """LIBERO dataset with per-step subgoal instructions (multilevel, task excluded).

    language_instruction per step is a JSON-encoded list:
        ["canonical subgoal", "variation 1", ...,
         "spatial_command", "low_level_motion", "low_level_motion_with_numbers"]

    The high-level task description is intentionally excluded.
    Steering command levels are appended only when steering_commands.json exists
    next to the episode's results.json.

    Use CanonicalBatchTransform or VariationBatchTransform when loading.
    """

    VERSION = tfds.core.Version("1.0.0")
    RELEASE_NOTES = {"1.0.0": "Initial release with subgoal-labeled instructions."}

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
                                    "JSON-encoded list of subgoal phrasings for this step: "
                                    "[canonical, variation1, variation2, ...]. "
                                    "Decode with json.loads() and pick index 0 (canonical) "
                                    "or sample randomly (variation training)."
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

    def _load_results_index(self, results_dir: str) -> List[Dict]:
        """
        Load all results.json files from results_dir, sorted by filename.
        Also loads steering_commands.json from the same directory if present.
        Returns list of dicts indexed by episode order.
        """
        results_path = Path(results_dir)
        json_files = sorted(results_path.rglob("results.json"))
        if not json_files:
            raise FileNotFoundError(f"No results.json files found in {results_dir}")

        n_with_steering = 0
        results_list = []
        for p in json_files:
            with open(p) as f:
                r = json.load(f)
            steering_path = p.parent / "steering_commands.json"
            if steering_path.exists():
                with open(steering_path) as f:
                    r["steering_commands"] = json.load(f)
                n_with_steering += 1
            results_list.append(r)
        print(f"  Loaded {len(results_list)} results.json files from {results_dir} "
              f"({n_with_steering} with steering_commands.json)")
        return results_list

    # Multilevel steering command keys to include, in order.
    # "task" is intentionally excluded — only subgoal-level and below are sampled.
    _STEERING_KEYS = ("spatial_command", "low_level_motion", "low_level_motion_with_numbers")

    def _get_per_frame_variations(self, results: Dict, t: int) -> str:
        """
        Returns JSON-encoded list of all phrasings for frame t.

        List order:
          [canonical_subgoal, variation_1, ..., spatial_command, low_level_motion,
           low_level_motion_with_numbers]

        Task-level description is intentionally excluded.
        Falls back to canonical-only if neither subgoal_variations nor
        steering_commands are present.
        """
        per_frame = results["per_frame_labels"]
        sg_idx = per_frame[t] if t < len(per_frame) else per_frame[-1]

        if "subgoal_variations" in results:
            variations = list(results["subgoal_variations"][sg_idx])
        else:
            # No variations generated yet — use canonical only
            variations = [results["subgoals"][sg_idx]]

        # Append multilevel steering commands (task-level excluded)
        if "steering_commands" in results:
            sc = results["steering_commands"]
            if sg_idx < len(sc):
                entry = sc[sg_idx]
                for key in self._STEERING_KEYS:
                    value = entry.get(key, "")
                    if value:
                        variations.append(value)

        return json.dumps(variations)

    def _generate_examples(self) -> Iterator[Tuple[str, Any]]:
        config = self.builder_config
        assert isinstance(config, SubgoalDatasetConfig)

        tf.config.set_visible_devices([], "GPU")

        # Load results index (episode order must match source dataset order)
        results_list = self._load_results_index(config.results_dir)

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

            if ep_idx >= len(results_list):
                print(f"  Warning: more episodes in source ({ep_idx}) than results.json files "
                      f"({len(results_list)}). Stopping.")
                break

            results = results_list[ep_idx]

            # Per-task limit
            if config.max_trajs_per_task > 0:
                task_key = results.get("task", "")
                task_counts = getattr(self, "_task_counts", {})
                if task_counts.get(task_key, 0) >= config.max_trajs_per_task:
                    ep_idx += 1
                    continue
                task_counts[task_key] = task_counts.get(task_key, 0) + 1
                self._task_counts = task_counts

            new_steps = []
            for t, step in enumerate(steps):
                lang = self._get_per_frame_variations(results, t)
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

            # Metadata: subgoal index of step 0, total subgoals
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
