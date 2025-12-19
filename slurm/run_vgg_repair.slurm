#!/bin/bash
#SBATCH --job-name=cifar100_vgg_repair
#SBATCH --output=logs/cifar100_experiments_%A_%a.out
#SBATCH --error=logs/cifar100_experiments_%A_%a.err
#SBATCH --time=48:00:00
#SBATCH --partition=preemptible
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --array=8-11

# ============================================================================
# REPAIR JOB: VGG16 Experiments Only
# ============================================================================

# Configurations: MODEL POOL_TYPE EXTRA_ARGS
CONFIGS=(
    # ResNet18 variants (Skipping 0-3) (Placeholder)
    "skip skip"
    "skip skip"
    "skip skip"
    "skip skip"
    # ResNet50 variants (Skipping 4-7) (Placeholder)
    "skip skip"
    "skip skip"
    "skip skip"
    "skip skip"
    # VGG16 variants (Targeting 8-11)
    "vgg16 baseline"
    "vgg16 tmaxavg"
    "vgg16 soft_tmaxavg"
    "vgg16 channel_adaptive"
)

# Get configuration for this array task
CONFIG=${CONFIGS[$SLURM_ARRAY_TASK_ID]}
read MODEL POOL_TYPE <<< "$CONFIG"

echo "=============================================="
echo "CIFAR-100 Experiment (VGG REPAIR)"
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

# Set temp directory
export TMPDIR="$SLURM_SUBMIT_DIR/tmp_vgg_${SLURM_ARRAY_TASK_ID}"
mkdir -p "$TMPDIR"

# Run experiment
python reproduce.py \
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
rm -rf "$TMPDIR"

EXIT_STATUS=$?
echo "End Time: $(date)"
exit $EXIT_STATUS
