#!/bin/bash
# build_multilevel_dataset.sh
#
# Builds all multilevel RLDS dataset variants:
#   libero_subgoal_multilevel        — all episodes
#   libero_subgoal_multilevel_5traj  — 5 episodes per task
#
# Requires per-episode dirs each containing:
#   results.json           (from test_subgoal_pipeline.py)
#   steering_commands.json (from generate_steering_commands.py)
#
# Usage:
#   bash build_multilevel_dataset.sh \
#       <source_dataset>   e.g. libero_goal_no_noops
#       <source_data_dir>  e.g. /proj/.../modified_libero_rlds_new
#       <results_dir>      e.g. /proj/.../outputs/subgoal_results_rlds
#       <output_data_dir>  e.g. /proj/.../modified_libero_rlds_new

set -e

SOURCE_DATASET="${1:?Usage: $0 <source_dataset> <source_data_dir> <results_dir> <output_data_dir>}"
SOURCE_DATA_DIR="${2:?}"
RESULTS_DIR="${3:?}"
OUTPUT_DATA_DIR="${4:?}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "================================================"
echo " Build Multi-Level RLDS Datasets"
echo "================================================"
echo " source_dataset : $SOURCE_DATASET"
echo " source_data_dir: $SOURCE_DATA_DIR"
echo " results_dir    : $RESULTS_DIR"
echo " output_data_dir: $OUTPUT_DATA_DIR"
echo "================================================"

build_variant() {
    local name="$1"
    local max_trajs="$2"
    echo ""
    echo "── Building $name (max_trajs_per_task=$max_trajs) ──"
    python - <<PYEOF
import sys
sys.path.insert(0, "$SCRIPT_DIR")
import tensorflow_datasets as tfds
from multilevel_dataset_builder import MultilevelDataset, MultilevelDatasetConfig

config = MultilevelDatasetConfig(
    name="$name",
    source_dataset="$SOURCE_DATASET",
    source_data_dir="$SOURCE_DATA_DIR",
    results_dir="$RESULTS_DIR",
    max_trajs_per_task=$max_trajs,
)

builder = MultilevelDataset(config=config, data_dir="$OUTPUT_DATA_DIR")
builder.download_and_prepare()
print(f"\nDataset written to: $OUTPUT_DATA_DIR/multilevel_dataset/$name/1.0.0/")
PYEOF
}

build_variant "libero_subgoal_multilevel"       0
build_variant "libero_subgoal_multilevel_5traj" 5
build_variant "libero_subgoal_multilevel_1traj" 1

# TFDS writes to data_dir/multilevel_dataset/<name>/ because the class is MultilevelDataset.
# The OpenVLA-OFT loader expects data_root_dir/<name>/ directly.
# Create symlinks so both paths work.
echo ""
echo "── Creating symlinks for OpenVLA-OFT loader ──"
for name in libero_subgoal_multilevel libero_subgoal_multilevel_5traj libero_subgoal_multilevel_1traj; do
    src="$OUTPUT_DATA_DIR/multilevel_dataset/$name"
    dst="$OUTPUT_DATA_DIR/$name"
    if [ -d "$src" ] && [ ! -e "$dst" ]; then
        ln -sfn "$src" "$dst"
        echo "  $dst -> $src"
    fi
done

echo ""
echo "================================================"
echo " Done. All variants built in $OUTPUT_DATA_DIR"
echo " Train with --data_root_dir $OUTPUT_DATA_DIR and:"
echo "   --dataset_name libero_subgoal_multilevel         (all data)"
echo "   --dataset_name libero_subgoal_multilevel_5traj   (5 traj/task)"
echo "   --variation_sampling True"
echo "   --variation_levels task,subgoal,low_level_motion,low_level_motion_with_numbers,spatial_command,object_affordance"
echo "================================================"
