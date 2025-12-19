#!/bin/bash
#SBATCH --job-name=tinyimagenet_exp
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --time=48:00:00
#SBATCH --partition=preemptible
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --array=0-7

# ============================================================================
# Tiny ImageNet Experiments
# ============================================================================
#
# Running baseline vs innovations on Tiny ImageNet-200.
#
# Usage:
#   sbatch run_tinyimagenet_experiments.slurm
# ============================================================================

# Model | Pooling Method | Extra Args
CONFIGS=(
    # ResNet18
    "resnet18 baseline"
    "resnet18 tmaxavg --use_tmaxavg --K 2 --T 0.7"
    "resnet18 soft_tmaxavg --use_soft_tmaxavg --K 2 --T 0.7 --temperature 0.1"
    "resnet18 channel_adaptive --use_channel_adaptive"
    
    # VGG16
    "vgg16 baseline"
    "vgg16 tmaxavg --use_tmaxavg --K 2 --T 0.7"
    "vgg16 soft_tmaxavg --use_soft_tmaxavg --K 2 --T 0.7 --temperature 0.1"
    "vgg16 channel_adaptive --use_channel_adaptive"
)

# Get configuration for this array task
CONFIG_STR="${CONFIGS[$SLURM_ARRAY_TASK_ID]}"
MODEL=$(echo "$CONFIG_STR" | cut -d' ' -f1)
METHOD=$(echo "$CONFIG_STR" | cut -d' ' -f2)
EXTRA_ARGS=$(echo "$CONFIG_STR" | cut -d' ' -f3-)

echo "=============================================="
echo "Tiny ImageNet Experiment"
echo "Job ID: $SLURM_ARRAY_JOB_ID"
echo "Task ID: $SLURM_ARRAY_TASK_ID"
echo "Model: $MODEL"
echo "Method: $METHOD"
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
DATASET="tinyimagenet"
EPOCHS=100
RUNS=3
BATCH_SIZE=128
LR=0.1
OUTPUT_DIR="./results/${MODEL}_tiny_${METHOD}"

mkdir -p "$OUTPUT_DIR"

if [ "$METHOD" == "baseline" ]; then
    EXTRA_ARGS=""
fi

# Set temp directory (needed for compilation)
export TMPDIR="$SLURM_SUBMIT_DIR/tmp_tiny_${SLURM_ARRAY_TASK_ID}"
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
    $EXTRA_ARGS

# Cleanup
rm -rf "$TMPDIR"

EXIT_STATUS=$?
echo ""
echo "=============================================="
if [ $EXIT_STATUS -eq 0 ]; then
    echo "Task $SLURM_ARRAY_TASK_ID ($MODEL - $METHOD) completed!"
else
    echo "Task $SLURM_ARRAY_TASK_ID failed with status: $EXIT_STATUS"
fi
echo "End Time: $(date)"
echo "=============================================="

exit $EXIT_STATUS
