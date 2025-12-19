#!/bin/bash
#SBATCH --job-name=cifar100_experiments
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --time=48:00:00
#SBATCH --partition=preemptible
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --array=0-11

# ============================================================================
# CIFAR-100 Experiments with Novel Pooling Methods
# ============================================================================
#
# Comprehensive experiments comparing:
# - Baseline (MaxPool)
# - T-Max-Avg pooling
# - Soft T-Max-Avg pooling (temperature-scaled)
# - Channel Adaptive pooling
#
# Models: ResNet18, ResNet50, VGG16
# Dataset: CIFAR-100
#
# Usage:
#   sbatch run_cifar100_experiments.slurm
#
# ============================================================================

# Configurations: MODEL POOL_TYPE EXTRA_ARGS
CONFIGS=(
    # ResNet18 variants
    "resnet18 baseline"
    "resnet18 tmaxavg"
    "resnet18 soft_tmaxavg"
    "resnet18 channel_adaptive"
    # ResNet50 variants
    "resnet50 baseline"
    "resnet50 tmaxavg"
    "resnet50 soft_tmaxavg"
    "resnet50 channel_adaptive"
    # VGG16 variants
    "vgg16 baseline"
    "vgg16 tmaxavg"
    "vgg16 soft_tmaxavg"
    "vgg16 channel_adaptive"
)

# Get configuration for this array task
CONFIG=${CONFIGS[$SLURM_ARRAY_TASK_ID]}
read MODEL POOL_TYPE <<< "$CONFIG"

echo "=============================================="
echo "CIFAR-100 Experiment"
echo "Array Job ID: $SLURM_ARRAY_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Model: $MODEL"
echo "Pool Type: $POOL_TYPE"
echo "Node: $SLURMD_NODENAME"
echo "Start Time: $(date)"
echo "=============================================="

# Create directories
mkdir -p logs results

cd $SLURM_SUBMIT_DIR

# Activate flashenv
source ~/miniconda/etc/profile.d/conda.sh
conda activate flashenv

echo ""
echo "GPU Information:"
nvidia-smi
echo ""

# Configuration
DATASET="cifar100"
EPOCHS=200
RUNS=3
BATCH_SIZE=128
LR=0.1
K=4
T=0.7
TEMPERATURE=0.1

OUTPUT_DIR="./results/${MODEL}_${DATASET}_${POOL_TYPE}"
mkdir -p "$OUTPUT_DIR"
mkdir -p "./data"

# Pre-download dataset (only first task does this)
LOCKFILE="./data/.download_lock_cifar100"
if [ "$SLURM_ARRAY_TASK_ID" = "0" ]; then
    echo "Task 0: Pre-downloading CIFAR-100..."
    python -c "
from torchvision import datasets, transforms
print('Downloading CIFAR-100...')
datasets.CIFAR100(root='./data', train=True, download=True, transform=transforms.ToTensor())
datasets.CIFAR100(root='./data', train=False, download=True, transform=transforms.ToTensor())
print('CIFAR-100 downloaded successfully!')
"
    touch "$LOCKFILE"
else
    echo "Waiting for dataset download (task 0)..."
    while [ ! -f "$LOCKFILE" ]; do
        sleep 5
    done
    echo "Dataset ready."
fi

# Build command based on pool type
case $POOL_TYPE in
    "baseline")
        POOL_FLAGS=""
        ;;
    "tmaxavg")
        POOL_FLAGS="--use_tmaxavg --K $K --T $T"
        ;;
    "soft_tmaxavg")
        POOL_FLAGS="--use_soft_tmaxavg --K $K --T $T --temperature $TEMPERATURE"
        ;;
    "channel_adaptive")
        POOL_FLAGS="--use_channel_adaptive"
        ;;
esac

echo "=============================================="
echo "Configuration:"
echo "  Model: $MODEL"
echo "  Dataset: $DATASET"
echo "  Pool Type: $POOL_TYPE"
echo "  Pool Flags: $POOL_FLAGS"
echo "  Epochs: $EPOCHS"
echo "  Runs: $RUNS"
echo "  Output: $OUTPUT_DIR"
echo "=============================================="

# Set a unique temp directory per job + task to avoid collisions between arrays
# This prevents multiprocessing from conflicting across parallel jobs
export TMPDIR="/tmp/parsaidp_cifar100_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
export TEMP="$TMPDIR"
export TMP="$TMPDIR"
export TEMPDIR="$TMPDIR"
mkdir -p "$TMPDIR"
chmod 700 "$TMPDIR"

# Force Python's tempfile and multiprocessing to use our TMPDIR
export PYTHONUNBUFFERED=1

# Run experiment
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
    --output_dir "$OUTPUT_DIR" \
    --augment \
    --save_model \
    --amp \
    --bf16 \
    --auto_batch \
    --num_workers 12 \
    $POOL_FLAGS

# Cleanup
EXIT_STATUS=$?
rm -rf "$TMPDIR" || echo "Warning: failed to remove TMPDIR $TMPDIR"

echo ""
echo "=============================================="
if [ $EXIT_STATUS -eq 0 ]; then
    echo "Task $SLURM_ARRAY_TASK_ID ($MODEL, $POOL_TYPE) completed!"
else
    echo "Task $SLURM_ARRAY_TASK_ID failed with status: $EXIT_STATUS"
fi
echo "End Time: $(date)"
echo "=============================================="

exit $EXIT_STATUS
