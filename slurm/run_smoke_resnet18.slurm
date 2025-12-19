#!/bin/bash
#SBATCH --job-name=smoke_resnet18_cifar100
#SBATCH --output=logs/smoke_resnet18_%j.out
#SBATCH --error=logs/smoke_resnet18_%j.err
#SBATCH --time=01:00:00
#SBATCH --partition=preemptible
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1

set -e

echo "=============================================="
echo "Smoke test: ResNet18 on CIFAR-100 (1 epoch, 1 run)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Start Time: $(date)"
echo "=============================================="

mkdir -p logs results data

cd "$SLURM_SUBMIT_DIR"

# Activate flashenv
source ~/miniconda/etc/profile.d/conda.sh
conda activate flashenv

echo ""
echo "GPU Information:"
nvidia-smi || echo "nvidia-smi not available"
echo ""

DATASET="cifar100"
EPOCHS=1
RUNS=1
BATCH_SIZE=128
LR=0.1
K=4
T=0.7
TEMPERATURE=0.1

OUTPUT_DIR="./results/smoke_resnet18_${DATASET}"
mkdir -p "$OUTPUT_DIR"

export TMPDIR="$SLURM_SUBMIT_DIR/tmp_smoke_resnet18_$SLURM_JOB_ID"
mkdir -p "$TMPDIR"

PYTHONPATH=. python reproduce.py \
    --experiment large_model \
    --model resnet18 \
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
    --num_workers 4

STATUS=$?
rm -rf "$TMPDIR"

echo "End Time: $(date)"
echo "Exit status: $STATUS"
exit $STATUS


