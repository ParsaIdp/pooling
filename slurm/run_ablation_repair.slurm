#!/bin/bash
#SBATCH --job-name=ablation_repair
#SBATCH --output=logs/ablation_studies_%A_%a.out
#SBATCH --error=logs/ablation_studies_%A_%a.err
#SBATCH --time=48:00:00
#SBATCH --partition=preemptible
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --array=0-6

# ============================================================================
# REPAIR JOB: Ablation Studies (Tasks 0-6)
# ============================================================================

# Configurations: K T
CONFIGS=(
    # Varying K (fixed T=0.7)
    "2 0.7"  # Task 0 (REDO)
    "3 0.7"  # Task 1 (REDO)
    "4 0.7"  # Task 2 (REDO)
    "6 0.7"  # Task 3 (REDO)
    # Varying T (fixed K=4)
    "4 0.3"  # Task 4 (REDO)
    "4 0.5"  # Task 5 (REDO)
    "4 0.9"  # Task 6 (REDO)
    # Task 7 (6 0.5) is DONE on Job 1495236
)

# Get configuration for this array task
CONFIG=${CONFIGS[$SLURM_ARRAY_TASK_ID]}
read K_VAL T_VAL <<< "$CONFIG"

echo "=============================================="
echo "Ablation Study REPAIR: ResNet18 on CIFAR-100"
echo "Array Job ID: $SLURM_ARRAY_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "K: $K_VAL"
echo "T: $T_VAL"
echo "Start Time: $(date)"
echo "=============================================="

# Create directories (reuse existing)
mkdir -p logs results
cd $SLURM_SUBMIT_DIR

# Activate flashenv
source ~/miniconda/etc/profile.d/conda.sh
conda activate flashenv

# CRITICAL FIX: Local TMPDIR and reduced workers
export TMPDIR="/tmp/parsaidp_ablation_repair_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
mkdir -p "$TMPDIR"
chmod 700 "$TMPDIR"

# DATASET=cifar100 (from original script logic)

python reproduce.py \
    --experiment large_model \
    --model resnet18 \
    --datasets cifar100 \
    --epochs 200 \
    --runs 3 \
    --batch_size 128 \
    --lr 0.1 \
    --weight_decay 5e-4 \
    --scheduler cosine \
    --output_dir "./results/ablation_resnet18_cifar100" \
    --augment \
    --save_model \
    --amp \
    --bf16 \
    --auto_batch \
    --num_workers 4 \
    --use_tmaxavg \
    --K "$K_VAL" \
    --T "$T_VAL"

# Cleanup
rm -rf "$TMPDIR"

EXIT_STATUS=$?
echo "End Time: $(date)"
exit $EXIT_STATUS
