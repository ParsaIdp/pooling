#!/bin/bash
#SBATCH --job-name=cifar100_resnet_repair
#SBATCH --output=logs/cifar100_experiments_%A_%a.out
#SBATCH --error=logs/cifar100_experiments_%A_%a.err
#SBATCH --time=48:00:00
#SBATCH --partition=preemptible
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --array=1,2,4,5,6,7

# ============================================================================
# REPAIR JOB: ResNet Experiments Only
# ============================================================================

# Configurations: MODEL POOL_TYPE EXTRA_ARGS
CONFIGS=(
    # ResNet18 variants
    "resnet18 baseline"          # 0 (Done)
    "resnet18 tmaxavg"           # 1 (REDO)
    "resnet18 soft_tmaxavg"      # 2 (REDO)
    "resnet18 channel_adaptive"  # 3 (Done)
    # ResNet50 variants
    "resnet50 baseline"          # 4 (REDO)
    "resnet50 tmaxavg"           # 5 (REDO)
    "resnet50 soft_tmaxavg"      # 6 (REDO)
    "resnet50 channel_adaptive"  # 7 (REDO)
)

# Get configuration for this array task
CONFIG=${CONFIGS[$SLURM_ARRAY_TASK_ID]}
read MODEL POOL_TYPE <<< "$CONFIG"

echo "=============================================="
echo "CIFAR-100 Experiment (RESNET REPAIR)"
echo "Array Job ID: $SLURM_ARRAY_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Model: $MODEL"
echo "Pool Type: $POOL_TYPE"
echo "Start Time: $(date)"
echo "=============================================="

# Create directories
mkdir -p logs results

cd $SLURM_SUBMIT_DIR

# Activate flashenv
source ~/miniconda/etc/profile.d/conda.sh
conda activate flashenv

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

# CRITICAL FIX: Use local /tmp ensuring unique and cleanup
# Also reduce num_workers to avoid shared memory exhaustion
export TMPDIR="/tmp/parsaidp_resnet_repair_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
mkdir -p "$TMPDIR"
chmod 700 "$TMPDIR"

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
    --num_workers 4 \
    $POOL_FLAGS

# Cleanup
rm -rf "$TMPDIR"

EXIT_STATUS=$?
echo "End Time: $(date)"
exit $EXIT_STATUS
