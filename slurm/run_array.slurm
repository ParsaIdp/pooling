#!/bin/bash
#SBATCH --job-name=tmaxavg_array
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --time=12:00:00
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --array=0-2

# ============================================================================
# T-Max-Avg Pooling Experiments - SLURM Array Job
# ============================================================================
#
# This script runs experiments in parallel across datasets using SLURM arrays.
# Each array task handles one dataset.
#
# Usage:
#   sbatch run_array.slurm                    # Run table14 on all 3 datasets
#   sbatch --export=EXPERIMENT=grid_search run_array.slurm
#
# ============================================================================

# Array of datasets to process
DATASETS=("cifar10" "cifar100" "mnist")

# Get the dataset for this array task
DATASET=${DATASETS[$SLURM_ARRAY_TASK_ID]}

# Print job info
echo "=============================================="
echo "Array Job ID: $SLURM_ARRAY_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Dataset: $DATASET"
echo "Node: $SLURMD_NODENAME"
echo "Start Time: $(date)"
echo "=============================================="

# Create directories
mkdir -p logs
mkdir -p results

cd $SLURM_SUBMIT_DIR

# Load modules (adjust as needed)
# module purge
# module load cuda/11.8
# module load python/3.10

# Activate environment
# source ~/.venv/bin/activate

echo ""
echo "GPU Information:"
nvidia-smi
echo ""

# Configuration
EXPERIMENT=${EXPERIMENT:-table14}
EPOCHS=${EPOCHS:-50}
RUNS=${RUNS:-6}
BATCH_SIZE=${BATCH_SIZE:-128}
LR=${LR:-0.001}
OUTPUT_DIR="./results/${DATASET}"
SEED=${SEED:-42}

mkdir -p "$OUTPUT_DIR"

echo "=============================================="
echo "Configuration:"
echo "  Dataset: $DATASET"
echo "  Experiment: $EXPERIMENT"
echo "  Epochs: $EPOCHS"
echo "  Runs: $RUNS"
echo "  Output: $OUTPUT_DIR"
echo "=============================================="
echo ""

# Run experiment for this dataset
python reproduce.py \
    --experiment "$EXPERIMENT" \
    --datasets "$DATASET" \
    --epochs "$EPOCHS" \
    --runs "$RUNS" \
    --batch_size "$BATCH_SIZE" \
    --lr "$LR" \
    --output_dir "$OUTPUT_DIR" \
    --seed "$SEED"

EXIT_STATUS=$?
echo ""
echo "=============================================="
if [ $EXIT_STATUS -eq 0 ]; then
    echo "Task $SLURM_ARRAY_TASK_ID ($DATASET) completed successfully!"
else
    echo "Task $SLURM_ARRAY_TASK_ID ($DATASET) failed with status: $EXIT_STATUS"
fi
echo "End Time: $(date)"
echo "=============================================="

exit $EXIT_STATUS


