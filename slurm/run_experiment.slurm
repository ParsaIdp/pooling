#!/bin/bash
#SBATCH --job-name=tmaxavg_exp
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=48:00:00
#SBATCH --partition=preemptible
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gres=gpu:1

# ============================================================================
# T-Max-Avg Pooling Experiments - SLURM Submission Script
# ============================================================================
#
# BASIC USAGE:
#   sbatch run_experiment.slurm                          # Run default (table14)
#   sbatch run_experiment.slurm table14                  # Run table14 experiments
#   sbatch run_experiment.slurm chestx                   # Run ChestX experiments
#   sbatch run_experiment.slurm grid_search              # Run grid search
#
# LARGE MODEL EXAMPLES:
#   # Train ResNet-18 with T-Max-Avg pooling
#   sbatch --export=MODEL=resnet18,USE_TMAXAVG=true run_experiment.slurm large_model
#
#   # Train VGG-16 on CIFAR-100 with augmentation
#   sbatch --export=MODEL=vgg16,DATASETS=cifar100,AUGMENT=true,EPOCHS=200 run_experiment.slurm large_model
#
#   # Train Wide ResNet with specific K and T values
#   sbatch --export=MODEL=wrn28_10,USE_TMAXAVG=true,K=4,T=0.8 run_experiment.slurm large_model
#
#   # Compare ResNet-50 with and without T-Max-Avg
#   sbatch --export=MODEL=resnet50,EPOCHS=200,AUGMENT=true run_experiment.slurm large_model
#
# AVAILABLE MODELS:
#   lenet, vgg11, vgg13, vgg16, vgg19, resnet18, resnet34, resnet50,
#   resnet101, wideresnet, wrn28_10, wrn40_2
#
# ============================================================================

# Print job info
echo "=============================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURMD_NODENAME"
echo "Start Time: $(date)"
echo "=============================================="

# Create logs directory if it doesn't exist
mkdir -p logs
mkdir -p results

# Navigate to the project directory
cd $SLURM_SUBMIT_DIR

# Load required modules (adjust based on your cluster)
# module purge
# module load cuda/11.8
# module load python/3.10

# Activate virtual environment (adjust path as needed)
# source ~/.venv/bin/activate
# OR
# conda activate myenv

# Check GPU availability
echo ""
echo "GPU Information:"
nvidia-smi
echo ""

# Set environment variables
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Parse experiment type from command line argument
EXPERIMENT=${1:-table14}

# Get optional environment variables with defaults
MODEL=${MODEL:-resnet18}
EPOCHS=${EPOCHS:-50}
RUNS=${RUNS:-6}
BATCH_SIZE=${BATCH_SIZE:-128}
LR=${LR:-0.1}
WEIGHT_DECAY=${WEIGHT_DECAY:-5e-4}
SCHEDULER=${SCHEDULER:-cosine}
DATASETS=${DATASETS:-"cifar10"}
OUTPUT_DIR=${OUTPUT_DIR:-"./results"}
SEED=${SEED:-42}
AUGMENT=${AUGMENT:-false}
USE_TMAXAVG=${USE_TMAXAVG:-false}
POOL_TYPE=${POOL_TYPE:-max}
K=${K:-2}
T=${T:-0.7}
SAVE_MODEL=${SAVE_MODEL:-false}

# Build augment flag
AUGMENT_FLAG=""
if [ "$AUGMENT" = "true" ]; then
    AUGMENT_FLAG="--augment"
fi

# Build T-Max-Avg flag
TMAXAVG_FLAG=""
if [ "$USE_TMAXAVG" = "true" ]; then
    TMAXAVG_FLAG="--use_tmaxavg"
fi

# Build save model flag
SAVE_FLAG=""
if [ "$SAVE_MODEL" = "true" ]; then
    SAVE_FLAG="--save_model"
fi

echo "=============================================="
echo "Experiment Configuration:"
echo "  Experiment: $EXPERIMENT"
echo "  Model: $MODEL"
echo "  Epochs: $EPOCHS"
echo "  Runs: $RUNS"
echo "  Batch Size: $BATCH_SIZE"
echo "  Learning Rate: $LR"
echo "  Weight Decay: $WEIGHT_DECAY"
echo "  Scheduler: $SCHEDULER"
echo "  Datasets: $DATASETS"
echo "  Output Dir: $OUTPUT_DIR"
echo "  Seed: $SEED"
echo "  Augmentation: $AUGMENT"
echo "  T-Max-Avg: $USE_TMAXAVG"
echo "  Pool Type: $POOL_TYPE"
echo "  K: $K"
echo "  T: $T"
echo "=============================================="
echo ""

# Run the experiment
if [ "$EXPERIMENT" = "large_model" ]; then
    # H100 optimization flags
    AMP_FLAG=""
    if [ "${AMP:-true}" = "true" ]; then
        AMP_FLAG="--amp --bf16"
    fi

    COMPILE_FLAG=""
    if [ "${COMPILE:-true}" = "true" ]; then
        COMPILE_FLAG="--compile"
    fi

    AUTO_BATCH_FLAG=""
    if [ "${AUTO_BATCH:-true}" = "true" ]; then
        AUTO_BATCH_FLAG="--auto_batch"
    fi

    python reproduce.py \
        --experiment "$EXPERIMENT" \
        --model "$MODEL" \
        --datasets $DATASETS \
        --epochs "$EPOCHS" \
        --runs "$RUNS" \
        --batch_size "$BATCH_SIZE" \
        --lr "$LR" \
        --weight_decay "$WEIGHT_DECAY" \
        --scheduler "$SCHEDULER" \
        --pool_type "$POOL_TYPE" \
        --K "$K" \
        --T "$T" \
        --output_dir "$OUTPUT_DIR" \
        --seed "$SEED" \
        --num_workers 12 \
        $AUGMENT_FLAG \
        $TMAXAVG_FLAG \
        $SAVE_FLAG \
        $AMP_FLAG \
        $COMPILE_FLAG \
        $AUTO_BATCH_FLAG
else
    python reproduce.py \
        --experiment "$EXPERIMENT" \
        --datasets $DATASETS \
        --epochs "$EPOCHS" \
        --runs "$RUNS" \
        --batch_size "$BATCH_SIZE" \
        --lr "$LR" \
        --output_dir "$OUTPUT_DIR" \
        --seed "$SEED"
fi

# Check exit status
EXIT_STATUS=$?
if [ $EXIT_STATUS -eq 0 ]; then
    echo ""
    echo "=============================================="
    echo "Experiment completed successfully!"
    echo "Results saved to: $OUTPUT_DIR"
    echo "End Time: $(date)"
    echo "=============================================="
else
    echo ""
    echo "=============================================="
    echo "Experiment failed with exit status: $EXIT_STATUS"
    echo "End Time: $(date)"
    echo "=============================================="
fi

exit $EXIT_STATUS
