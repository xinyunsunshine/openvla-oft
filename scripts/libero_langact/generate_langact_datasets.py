"""
Generate LIBERO RLDS datasets with language-action prompts.

Reads each original LIBERO RLDS dataset, computes per-step language-action text
describing the robot's motion, and writes new RLDS datasets. Generates two variants:
  - verbose:  "move forward 3.0 cm, move up 1.0 cm, rotate clockwise 15 degrees, close gripper"
  - nonum:    "move forward, move up, rotate clockwise, close gripper"

Usage:
    python scripts/libero_langact/generate_langact_datasets.py \
        --data_dir /proj/inf-scaling/SWE_sim/modified_libero_rlds \
        --output_dir /proj/inf-scaling/SWE_sim/modified_libero_rlds \
        --datasets libero_spatial_no_noops libero_object_no_noops \
                   libero_goal_no_noops libero_10_no_noops

    # Generate only verbose format for one dataset:
    python scripts/libero_langact/generate_langact_datasets.py \
        --data_dir /proj/inf-scaling/SWE_sim/modified_libero_rlds \
        --output_dir /proj/inf-scaling/SWE_sim/modified_libero_rlds \
        --datasets libero_spatial_no_noops \
        --formats verbose

    # Custom window size:
    python scripts/libero_langact/generate_langact_datasets.py \
        --data_dir /proj/inf-scaling/SWE_sim/modified_libero_rlds \
        --output_dir /proj/inf-scaling/SWE_sim/modified_libero_rlds \
        --datasets libero_spatial_no_noops \
        --window_size 16
"""

import argparse
import json
import os
import shutil
import sys

import tensorflow as tf

# Disable GPU for TF to avoid conflicts with PyTorch
tf.config.set_visible_devices([], "GPU")

import tensorflow_datasets as tfds

# Add this script's directory to path so we can import the builder
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Add LAP source to path so the builder can use LAP's native TF implementation
_lap_src = os.path.join(os.path.dirname(__file__), "..", "..", "..", "lap", "src")
if os.path.isdir(_lap_src):
    sys.path.insert(0, os.path.abspath(_lap_src))

from libero_langact_dataset_builder import LiberoLangact, LiberoLangactConfig


DEFAULT_DATASETS = [
    "libero_spatial_no_noops",
    "libero_object_no_noops",
    "libero_goal_no_noops",
    "libero_10_no_noops",
]

FORMAT_SUFFIXES = {
    "verbose": "langact_verbose",
    "nonum": "langact_nonum",
}


def generate_dataset(
    source_dataset: str,
    source_data_dir: str,
    output_dir: str,
    langact_format: str,
    window_size: int,
    include_rotation: bool,
):
    """Generate a single language-action RLDS dataset."""
    suffix = FORMAT_SUFFIXES[langact_format]
    output_name = f"{source_dataset}_{suffix}"

    print(f"\n{'='*70}")
    print(f"Generating: {output_name}")
    print(f"  Source: {source_dataset} from {source_data_dir}")
    print(f"  Format: {langact_format}")
    print(f"  Window: {window_size} steps")
    print(f"  Rotation: {'yes' if include_rotation else 'no'}")
    print(f"  Output: {output_dir}/{output_name}")
    print(f"{'='*70}")

    config = LiberoLangactConfig(
        name="default",
        version=tfds.core.Version("1.0.0"),
        description=f"LIBERO with {langact_format} language-action prompts (window={window_size})",
        source_dataset=source_dataset,
        source_data_dir=source_data_dir,
        langact_format=langact_format,
        window_size=window_size,
        include_rotation=include_rotation,
    )

    builder = LiberoLangact(
        config=config,
        data_dir=os.path.join(output_dir, output_name),
    )

    # Generate the dataset
    builder.download_and_prepare()

    # TFDS creates nested dirs: {data_dir}/libero_langact/default/1.0.0/
    # OpenVLA-OFT expects:       {output_dir}/{output_name}/1.0.0/
    # Restructure to match the expected flat layout.
    tfds_dir = os.path.join(output_dir, output_name, "libero_langact", "default", "1.0.0")
    target_dir = os.path.join(output_dir, output_name, "1.0.0")
    if os.path.exists(tfds_dir) and not os.path.exists(target_dir):
        os.makedirs(target_dir, exist_ok=True)
        for f in os.listdir(tfds_dir):
            shutil.move(os.path.join(tfds_dir, f), os.path.join(target_dir, f))
        # Clean up the TFDS nested dirs
        shutil.rmtree(os.path.join(output_dir, output_name, "libero_langact"))
        shutil.rmtree(os.path.join(output_dir, output_name, "downloads"), ignore_errors=True)

        # Rename TFRecord files from builder name to dataset name
        builder_name = "libero_langact"
        for f in os.listdir(target_dir):
            if f.startswith(builder_name):
                new_name = f.replace(builder_name, output_name, 1)
                os.rename(os.path.join(target_dir, f), os.path.join(target_dir, new_name))

        # Fix dataset_info.json to use the correct dataset name
        info_path = os.path.join(target_dir, "dataset_info.json")
        if os.path.exists(info_path):
            with open(info_path, "r") as f:
                info = json.load(f)
            info["name"] = output_name
            info["moduleName"] = f"libero_langact.libero_langact_dataset_builder"
            with open(info_path, "w") as f:
                json.dump(info, f, indent=2)

    print(f"  Done! Dataset written to: {os.path.join(output_dir, output_name)}")
    return output_name


def verify_dataset(dataset_name: str, data_dir: str, num_episodes: int = 2):
    """Load and print a few examples from a generated dataset to verify correctness."""
    print(f"\n--- Verifying {dataset_name} ---")

    builder = tfds.builder(dataset_name, data_dir=data_dir)
    ds = builder.as_dataset(split="train")

    for ep_idx, episode in enumerate(ds):
        if ep_idx >= num_episodes:
            break

        steps = list(episode["steps"])
        print(f"\n  Episode {ep_idx}: {len(steps)} steps")

        # Show first 3 and last step
        show_indices = list(range(min(3, len(steps)))) + [len(steps) - 1]
        for t in show_indices:
            step = steps[t]
            lang = step["language_instruction"].numpy().decode("utf-8")
            action = step["action"].numpy()
            state = step["observation"]["state"].numpy()
            print(f"    Step {t:3d}: lang_instr=\"{lang}\"")
            print(f"             action={action}")
            print(f"             state_pos={state[:3]}, state_axang={state[3:6]}, grip={state[6:]}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate LIBERO RLDS datasets with language-action prompts."
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Directory containing source LIBERO RLDS datasets.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to write the generated datasets.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=DEFAULT_DATASETS,
        help=f"Source dataset names. Default: {DEFAULT_DATASETS}",
    )
    parser.add_argument(
        "--formats",
        nargs="+",
        default=["verbose", "nonum"],
        choices=["verbose", "nonum"],
        help="Language-action formats to generate. Default: both.",
    )
    parser.add_argument(
        "--window_size",
        type=int,
        default=8,
        help="Temporal aggregation window (number of future steps). Default: 8.",
    )
    parser.add_argument(
        "--include_rotation",
        action="store_true",
        default=True,
        help="Include rotation in language-action text. Default: True.",
    )
    parser.add_argument(
        "--no_rotation",
        action="store_true",
        help="Exclude rotation from language-action text.",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="After generation, load and print a few examples for verification.",
    )
    args = parser.parse_args()

    include_rotation = not args.no_rotation

    generated = []
    for source_ds in args.datasets:
        for fmt in args.formats:
            output_name = generate_dataset(
                source_dataset=source_ds,
                source_data_dir=args.data_dir,
                output_dir=args.output_dir,
                langact_format=fmt,
                window_size=args.window_size,
                include_rotation=include_rotation,
            )
            generated.append(output_name)

    print(f"\n{'='*70}")
    print("All datasets generated successfully!")
    print(f"Generated {len(generated)} datasets:")
    for name in generated:
        print(f"  - {name}")
    print(f"{'='*70}")

    if args.verify:
        for name in generated:
            verify_dataset(name, args.output_dir)


if __name__ == "__main__":
    main()
