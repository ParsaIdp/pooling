#!/bin/bash
#SBATCH --job-name=large_model_comparison
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --time=48:00:00
#SBATCH --partition=preemptible
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --array=0-5

# ============================================================================
# Large Model Comparison - SLURM Array Job
# ============================================================================
#
# This script compares different models with and without T-Max-Avg pooling.
# Each array task runs a different model configuration.
#
# Usage:
#   sbatch run_large_models.slurm                    # Run all 6 configurations
#   sbatch --array=0-2 run_large_models.slurm        # Run first 3 only
#
# ============================================================================

# Define configurations: MODEL USE_TMAXAVG
CONFIGS=(
    "resnet18 false"
    "resnet18 true"
    "resnet50 false"
    "resnet50 true"
    "vgg16 false"
    "vgg16 true"
)

# Extended configurations (uncomment to use)
# CONFIGS=(
#     "resnet18 false"
#     "resnet18 true"
#     "resnet34 false"
#     "resnet34 true"
#     "resnet50 false"
#     "resnet50 true"
#     "vgg16 false"
#     "vgg16 true"
#     "wrn28_10 false"
#     "wrn28_10 true"
# )

# Get configuration for this array task
CONFIG=${CONFIGS[$SLURM_ARRAY_TASK_ID]}
read MODEL USE_TMAXAVG <<< "$CONFIG"

# Print job info
echo "=============================================="
echo "Array Job ID: $SLURM_ARRAY_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Model: $MODEL"
echo "T-Max-Avg: $USE_TMAXAVG"
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

# Activate flashenv
source ~/miniconda/etc/profile.d/conda.sh
conda activate flashenv

echo ""
echo "GPU Information:"
nvidia-smi
echo ""

# Configuration
DATASET=${DATASET:-cifar10}
EPOCHS=${EPOCHS:-200}
RUNS=${RUNS:-3}
BATCH_SIZE=${BATCH_SIZE:-128}
LR=${LR:-0.1}
K=${K:-4}
T=${T:-0.7}
OUTPUT_DIR="./results/${MODEL}_${DATASET}"

mkdir -p "$OUTPUT_DIR"
mkdir -p "./data"

# Pre-download dataset (only first task does this, others wait)
LOCKFILE="./data/.download_lock"
if [ "$SLURM_ARRAY_TASK_ID" = "0" ]; then
    echo "Task 0: Pre-downloading dataset..."
    python -c "
from torchvision import datasets, transforms
print('Downloading CIFAR-10...')
datasets.CIFAR10(root='./data', train=True, download=True, transform=transforms.ToTensor())
datasets.CIFAR10(root='./data', train=False, download=True, transform=transforms.ToTensor())
print('CIFAR-10 downloaded successfully!')
print('Downloading CIFAR-100...')
datasets.CIFAR100(root='./data', train=True, download=True, transform=transforms.ToTensor())
datasets.CIFAR100(root='./data', train=False, download=True, transform=transforms.ToTensor())
print('CIFAR-100 downloaded successfully!')
"
    touch "$LOCKFILE"
    echo "Dataset download complete."
else
    # Wait for task 0 to finish downloading
    echo "Waiting for dataset download (task 0)..."
    while [ ! -f "$LOCKFILE" ]; do
        sleep 5
    done
    echo "Dataset ready."
fi

# Build flags
TMAXAVG_FLAG=""
if [ "$USE_TMAXAVG" = "true" ]; then
    TMAXAVG_FLAG="--use_tmaxavg"
    OUTPUT_DIR="${OUTPUT_DIR}_tmaxavg"
fi

mkdir -p "$OUTPUT_DIR"

echo "=============================================="
echo "Configuration:"
echo "  Model: $MODEL"
echo "  Dataset: $DATASET"
echo "  T-Max-Avg: $USE_TMAXAVG"
echo "  K: $K, T: $T"
echo "  Epochs: $EPOCHS"
echo "  Runs: $RUNS"
echo "  Output: $OUTPUT_DIR"
echo "=============================================="
echo ""

# Set a unique temp directory per job + task to avoid collisions between arrays
# This prevents multiprocessing from conflicting across parallel jobs
export TMPDIR="/tmp/parsaidp_large_models_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
export TEMP="$TMPDIR"
export TMP="$TMPDIR"
export TEMPDIR="$TMPDIR"
mkdir -p "$TMPDIR"
chmod 700 "$TMPDIR"

# Force Python's tempfile and multiprocessing to use our TMPDIR
export PYTHONUNBUFFERED=1

# Run experiment with H100 optimizations
# NOTE: --compile disabled due to temp space issues with parallel jobs
PYTHONPATH=. python reproduce.py \
    --experiment large_model \
    --model "$MODEL" \
    --datasets "$DATASET" \
    --epochs "$EPOCHS" \
    --runs "$RUNS" \
    --batch_size "$BATCH_SIZE" \
    --lr "$LR" \
    --weight_decay 5e-4 \
    --scheduler cosine \
    --K "$K" \
    --T "$T" \
    --output_dir "$OUTPUT_DIR" \
    --augment \
    --save_model \
    --amp \
    --bf16 \
    --auto_batch \
    --num_workers 12 \
    $TMAXAVG_FLAG

# Cleanup temp directory
EXIT_STATUS=$?
rm -rf "$TMPDIR" || echo "Warning: failed to remove TMPDIR $TMPDIR"

echo ""
echo "=============================================="
if [ $EXIT_STATUS -eq 0 ]; then
    echo "Task $SLURM_ARRAY_TASK_ID ($MODEL, tmaxavg=$USE_TMAXAVG) completed!"
else
    echo "Task $SLURM_ARRAY_TASK_ID failed with status: $EXIT_STATUS"
fi
echo "End Time: $(date)"
echo "=============================================="

exit $EXIT_STATUS

