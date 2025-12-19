#!/bin/bash
#SBATCH --job-name=ablation_studies
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
# Ablation Studies: K and T values
# ============================================================================
#
# Studying the impact of K and T hyperparameters on ResNet18 performance.
#
# Usage:
#   sbatch run_ablation_studies.slurm
# ============================================================================

# Configurations: K T
CONFIGS=(
    # Varying K (fixed T=0.7)
    "2 0.7"  # Baseline K
    "3 0.7"
    "4 0.7"
    "6 0.7"
    # Varying T (fixed K=4, based on previous best guesses)
    "4 0.3"
    "4 0.5"
    "4 0.9"
    # Combined check
    "6 0.5"
)

# Get configuration for this array task
CONFIG=${CONFIGS[$SLURM_ARRAY_TASK_ID]}
read K_VAL T_VAL <<< "$CONFIG"

echo "=============================================="
echo "Ablation Study: ResNet18 on CIFAR-100"
echo "Array Job ID: $SLURM_ARRAY_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "K: $K_VAL"
echo "T: $T_VAL"
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
MODEL="resnet18"
DATASET="cifar100"
EPOCHS=200
RUNS=3
BATCH_SIZE=128
LR=0.1
OUTPUT_DIR="./results/ablation_${MODEL}_${DATASET}"

mkdir -p "$OUTPUT_DIR"

# Set a unique temp directory per job + task to avoid collisions between arrays
# This prevents multiprocessing from conflicting across parallel jobs
export TMPDIR="/tmp/parsaidp_ablation_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
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
    --use_tmaxavg \
    --K "$K_VAL" \
    --T "$T_VAL"

# Cleanup
EXIT_STATUS=$?
rm -rf "$TMPDIR" || echo "Warning: failed to remove TMPDIR $TMPDIR"

echo ""
echo "=============================================="
if [ $EXIT_STATUS -eq 0 ]; then
    echo "Task $SLURM_ARRAY_TASK_ID (K=$K_VAL, T=$T_VAL) completed!"
else
    echo "Task $SLURM_ARRAY_TASK_ID failed with status: $EXIT_STATUS"
fi
echo "End Time: $(date)"
echo "=============================================="

exit $EXIT_STATUS
