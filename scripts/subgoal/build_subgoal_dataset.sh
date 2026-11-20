#!/bin/bash
# build_subgoal_dataset.sh
#
# Builds all subgoal RLDS dataset variants:
#   libero_subgoal        — all episodes
#   libero_subgoal_5traj  — 5 episodes per task
#   libero_subgoal_1traj  — 1 episode per task
#
# Usage:
#   bash build_subgoal_dataset.sh \
#       <source_dataset>   e.g. libero_goal_no_noops
#       <source_data_dir>  e.g. /proj/.../rlds_datasets
#       <results_dir>      e.g. /proj/.../outputs/subgoal_results_rlds
#       <output_data_dir>  e.g. /proj/.../rlds_datasets

set -e

SOURCE_DATASET="${1:?Usage: $0 <source_dataset> <source_data_dir> <results_dir> <output_data_dir>}"
SOURCE_DATA_DIR="${2:?}"
RESULTS_DIR="${3:?}"
OUTPUT_DATA_DIR="${4:?}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "================================================"
echo " Build Subgoal RLDS Datasets"
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
from subgoal_dataset_builder import SubgoalDataset, SubgoalDatasetConfig

config = SubgoalDatasetConfig(
    name="$name",
    source_dataset="$SOURCE_DATASET",
    source_data_dir="$SOURCE_DATA_DIR",
    results_dir="$RESULTS_DIR",
    max_trajs_per_task=$max_trajs,
)

builder = SubgoalDataset(config=config, data_dir="$OUTPUT_DATA_DIR")
builder.download_and_prepare()
print(f"\\nDataset written to: $OUTPUT_DATA_DIR/subgoal_dataset/$name/1.0.0/")
PYEOF
}

build_variant "libero_subgoal"       0
build_variant "libero_subgoal_5traj" 5
build_variant "libero_subgoal_1traj" 1

# TFDS writes to data_dir/subgoal_dataset/<name>/ because the class is SubgoalDataset.
# The OpenVLA-OFT loader expects data_root_dir/<name>/ directly.
# Create symlinks so both paths work.
echo ""
echo "── Creating symlinks for OpenVLA-OFT loader ──"
for name in libero_subgoal libero_subgoal_5traj libero_subgoal_1traj; do
    src="$OUTPUT_DATA_DIR/subgoal_dataset/$name"
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
echo "   --dataset_name libero_subgoal         (all data)"
echo "   --dataset_name libero_subgoal_5traj   (5 traj/task)"
echo "   --dataset_name libero_subgoal_1traj   (1 traj/task)"
echo "================================================"
