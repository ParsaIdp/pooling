#!/bin/bash
#SBATCH --job-name=full_pooling_comparison
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --time=48:00:00
#SBATCH --partition=preemptible
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --array=0-15

# ============================================================================
# Full Pooling Innovation Comparison
# ============================================================================
#
# Tests ALL 8 innovations on VGG16 (CIFAR-100 + Tiny ImageNet):
#   0. Baseline (MaxPool)
#   1. T-Max-Avg
#   2. Soft T-Max-Avg
#   3. Channel Adaptive
#   4. Learnable T-Max-Avg
#   5. Gated T-Max-Avg
#   6. Attention-Weighted     <-- NEW INNOVATION
#   7. Stochastic Mix         <-- NEW INNOVATION
#
# ============================================================================

# Configurations: MODEL DATASET POOL_METHOD
CONFIGS=(
    # VGG16 on CIFAR-100 (8 methods)
    "vgg16 cifar100 baseline"
    "vgg16 cifar100 tmaxavg"
    "vgg16 cifar100 soft_tmaxavg"
    "vgg16 cifar100 channel_adaptive"
    "vgg16 cifar100 learnable_t"
    "vgg16 cifar100 gated"
    "vgg16 cifar100 attention_weighted"
    "vgg16 cifar100 stochastic_mix"
    # VGG16 on Tiny ImageNet (8 methods)
    "vgg16 tinyimagenet baseline"
    "vgg16 tinyimagenet tmaxavg"
    "vgg16 tinyimagenet soft_tmaxavg"
    "vgg16 tinyimagenet channel_adaptive"
    "vgg16 tinyimagenet learnable_t"
    "vgg16 tinyimagenet gated"
    "vgg16 tinyimagenet attention_weighted"
    "vgg16 tinyimagenet stochastic_mix"
)

CONFIG=${CONFIGS[$SLURM_ARRAY_TASK_ID]}
read MODEL DATASET POOL_METHOD <<< "$CONFIG"

echo "=============================================="
echo "Full Pooling Comparison"
echo "Array Job ID: $SLURM_ARRAY_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Model: $MODEL"
echo "Dataset: $DATASET"
echo "Pool Method: $POOL_METHOD"
echo "Node: $SLURMD_NODENAME"
echo "Start Time: $(date)"
echo "=============================================="

mkdir -p logs results
cd $SLURM_SUBMIT_DIR

# Activate flashenv
source ~/miniconda/etc/profile.d/conda.sh
conda activate flashenv

echo ""
echo "GPU Information:"
nvidia-smi
echo ""

# Set temp directory
export TMPDIR="/tmp/parsaidp_full_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
export TEMP="$TMPDIR"
export TMP="$TMPDIR"
export TEMPDIR="$TMPDIR"
mkdir -p "$TMPDIR"
chmod 700 "$TMPDIR"
export PYTHONUNBUFFERED=1

# Configuration
EPOCHS=200
RUNS=3
K=4
T=0.7
TEMPERATURE=0.1
OUTPUT_DIR="./results/${MODEL}_${DATASET}_${POOL_METHOD}_full"

mkdir -p "$OUTPUT_DIR"

# Build pool flags based on method
case $POOL_METHOD in
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
    "learnable_t")
        POOL_FLAGS="--use_learnable_t --K $K --T $T"
        ;;
    "gated")
        POOL_FLAGS="--use_gated"
        ;;
    "attention_weighted")
        POOL_FLAGS="--use_attention_weighted"
        ;;
    "stochastic_mix")
        POOL_FLAGS="--use_stochastic_mix"
        ;;
esac

echo "=============================================="
echo "Configuration:"
echo "  Model: $MODEL"
echo "  Dataset: $DATASET"
echo "  Pool Method: $POOL_METHOD"
echo "  Pool Flags: $POOL_FLAGS"
echo "  Epochs: $EPOCHS"
echo "  Runs: $RUNS"
echo "  Output: $OUTPUT_DIR"
echo "=============================================="

# Run experiment
PYTHONPATH=. python reproduce.py \
    --experiment large_model \
    --model "$MODEL" \
    --datasets "$DATASET" \
    --epochs "$EPOCHS" \
    --runs "$RUNS" \
    --batch_size 128 \
    --lr 0.1 \
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

EXIT_STATUS=$?
rm -rf "$TMPDIR" || echo "Warning: failed to remove TMPDIR"

echo ""
echo "=============================================="
if [ $EXIT_STATUS -eq 0 ]; then
    echo "Task $SLURM_ARRAY_TASK_ID ($MODEL, $DATASET, $POOL_METHOD) completed!"
else
    echo "Task $SLURM_ARRAY_TASK_ID failed with status: $EXIT_STATUS"
fi
echo "End Time: $(date)"
echo "=============================================="

exit $EXIT_STATUS

