"""
TFDS dataset builder that creates LIBERO RLDS datasets with language-action prompts.

Reads from an existing LIBERO RLDS dataset, computes per-step language-action text
(describing the robot's motion), and writes a new RLDS dataset where language_instruction
is replaced with the language-action text. All other fields (images, actions, state, etc.)
are copied unchanged.
"""

from typing import Any, Dict, Iterator, Tuple

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

try:
    # Prefer LAP's native implementation (TF-based, authoritative)
    from lap.datasets.utils.transforms import libero_dataset_transform as _lap_transform
    from lap.policies.transforms.action_text import summarize_numeric_actions

    _USE_LAP = True
except ImportError:
    # Fallback to local numpy port
    _USE_LAP = False

from langact_utils import (
    compute_language_actions,
    language_actions_to_text,
)


class LiberoLangactConfig(tfds.core.BuilderConfig):
    """Config for specifying the source dataset and language-action format."""

    def __init__(
        self,
        *,
        source_dataset: str,
        source_data_dir: str,
        langact_format: str = "verbose",  # "verbose" or "nonum"
        window_size: int = 8,
        include_rotation: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.source_dataset = source_dataset
        self.source_data_dir = source_data_dir
        self.langact_format = langact_format
        self.window_size = window_size
        self.include_rotation = include_rotation


class LiberoLangact(tfds.core.GeneratorBasedBuilder):
    """LIBERO dataset with per-step language-action instructions.

    Reads from an existing LIBERO RLDS dataset, computes motion descriptions
    for each timestep, and replaces language_instruction with the generated text.

    The language-action text describes the robot's cumulative motion over the next
    N steps (default 8, matching OpenVLA-OFT's action chunk size).

    Example verbose:  "move forward 3.0 cm, move up 1.0 cm, rotate clockwise 15 degrees, close gripper"
    Example no-number: "move forward, move up, rotate clockwise, close gripper"
    """

    VERSION = tfds.core.Version("1.0.0")
    RELEASE_NOTES = {"1.0.0": "Initial release with language-action prompts."}

    # Configs will be set dynamically via manual_builder_configs
    MANUAL_DOWNLOAD_INSTRUCTIONS = "Set source_dataset and source_data_dir in the config."

    def _info(self) -> tfds.core.DatasetInfo:
        """Match the original LIBERO RLDS feature spec exactly."""
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
                            "discount": tfds.features.Scalar(
                                dtype=np.float32,
                                doc="Discount if provided, default to 1.",
                            ),
                            "reward": tfds.features.Scalar(
                                dtype=np.float32,
                                doc="Reward if provided, 1 on final step for demos.",
                            ),
                            "is_first": tfds.features.Scalar(
                                dtype=np.bool_,
                                doc="True on first step of the episode.",
                            ),
                            "is_last": tfds.features.Scalar(
                                dtype=np.bool_,
                                doc="True on last step of the episode.",
                            ),
                            "is_terminal": tfds.features.Scalar(
                                dtype=np.bool_,
                                doc="True on last step of the episode.",
                            ),
                            "language_instruction": tfds.features.Text(
                                doc="Language-action instruction describing robot motion for this step.",
                            ),
                        }
                    ),
                    "episode_metadata": tfds.features.FeaturesDict(
                        {
                            "file_path": tfds.features.Text(
                                doc="Path to the original data file.",
                            ),
                        }
                    ),
                }
            )
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        return {"train": self._generate_examples()}

    def _generate_examples(self) -> Iterator[Tuple[str, Any]]:
        """Load source dataset, compute language-actions, yield modified episodes."""
        config = self.builder_config
        assert isinstance(config, LiberoLangactConfig)

        # Load the source LIBERO dataset
        # Disable GPU for TF to avoid conflicts
        tf.config.set_visible_devices([], "GPU")

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
                print(f"  Skipping corrupted TFRecord shard (likely unpulled LFS pointer)")
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

            # Extract states and actions for the full trajectory
            states = np.stack([s["observation"]["state"].numpy() for s in steps], axis=0)  # (T, 8)
            actions = np.stack([s["action"].numpy() for s in steps], axis=0)  # (T, 7)

            # Compute per-step language-action vectors (7D)
            if _USE_LAP:
                # Use LAP's authoritative TF implementation
                traj_dict = {
                    "observation": {"state": tf.constant(states, dtype=tf.float32)},
                    "action": tf.constant(actions, dtype=tf.float32),
                    "language_instruction": tf.repeat("", len(steps)),
                }
                result = _lap_transform(traj_dict)
                lang_actions = result["language_action"].numpy()
            else:
                lang_actions = compute_language_actions(states, actions)

            # Convert to text (both formats computed, pick the one we need)
            verbose_texts, nonum_texts = language_actions_to_text(
                lang_actions,
                window_size=config.window_size,
                include_rotation=config.include_rotation,
            )

            if config.langact_format == "verbose":
                texts = verbose_texts
            else:
                texts = nonum_texts

            # Build the episode with modified language_instruction
            new_steps = []
            for t, step in enumerate(steps):
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
                        "language_instruction": texts[t],
                    }
                )

            # Get episode metadata
            file_path = ""
            if "episode_metadata" in episode and "file_path" in episode["episode_metadata"]:
                file_path = episode["episode_metadata"]["file_path"].numpy().decode("utf-8")

            sample = {
                "steps": new_steps,
                "episode_metadata": {"file_path": file_path},
            }

            key = f"{config.source_dataset}::ep{ep_idx:05d}"
            yield key, sample
            ep_idx += 1
