#!/bin/bash
# ============================================================================
# ESE-5390 Final Project - One-Click Reproducibility Script
# ============================================================================
# 
# This script reproduces all results from our paper:
# "Adaptive Pooling Innovations for Efficient Deep Learning"
#
# Usage:
#   ./reproduce_results.sh [--quick]
#
# Options:
#   --quick    Run quick smoke tests (5 epochs) instead of full training
#
# Requirements:
#   - SLURM cluster with GPU nodes
#   - Conda with flashenv environment OR requirements.txt dependencies
#
# ============================================================================

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Parse arguments
QUICK_MODE=false
if [ "$1" == "--quick" ]; then
    QUICK_MODE=true
    echo "Running in QUICK mode (5 epochs for smoke testing)"
fi

echo "=============================================="
echo "ESE-5390 Final Project Reproducibility Script"
echo "=============================================="
echo "Working directory: $SCRIPT_DIR"
echo "Quick mode: $QUICK_MODE"
echo ""

# ============================================================================
# Step 1: Environment Setup
# ============================================================================
echo "[1/5] Setting up environment..."

# Check for conda
if command -v conda &> /dev/null; then
    echo "  Found conda installation"
    source ~/miniconda/etc/profile.d/conda.sh 2>/dev/null || source ~/anaconda3/etc/profile.d/conda.sh 2>/dev/null || true
    
    # Create environment if it doesn't exist
    if ! conda env list | grep -q "flashenv"; then
        echo "  Creating flashenv conda environment..."
        conda create -n flashenv python=3.10 -y
        conda activate flashenv
        pip install -r requirements.txt
    else
        echo "  Activating existing flashenv environment..."
        conda activate flashenv
    fi
else
    echo "  Conda not found, using pip with requirements.txt"
    pip install -r requirements.txt
fi

# Verify key dependencies
python -c "import torch; print(f'  PyTorch version: {torch.__version__}')"
python -c "import torch; print(f'  CUDA available: {torch.cuda.is_available()}')"

# ============================================================================
# Step 2: Dataset Download
# ============================================================================
echo ""
echo "[2/5] Downloading datasets..."

mkdir -p data

python -c "
from torchvision import datasets, transforms
import os

print('  Downloading CIFAR-100...')
datasets.CIFAR100(root='./data', train=True, download=True)
datasets.CIFAR100(root='./data', train=False, download=True)
print('  CIFAR-100 ready!')

print('  Downloading CIFAR-10...')
datasets.CIFAR10(root='./data', train=True, download=True)
datasets.CIFAR10(root='./data', train=False, download=True)
print('  CIFAR-10 ready!')

# Tiny ImageNet requires manual setup
if not os.path.exists('./data/tiny-imagenet-200'):
    print('  Note: Tiny ImageNet must be downloaded manually from:')
    print('        http://cs231n.stanford.edu/tiny-imagenet-200.zip')
    print('        Extract to ./data/tiny-imagenet-200/')
else:
    print('  Tiny ImageNet found!')
"

# ============================================================================
# Step 3: Create Output Directories
# ============================================================================
echo ""
echo "[3/5] Creating output directories..."

mkdir -p logs
mkdir -p results
mkdir -p paper/figures

echo "  Created: logs/, results/, paper/figures/"

# ============================================================================
# Step 4: Submit Training Jobs
# ============================================================================
echo ""
echo "[4/5] Submitting training jobs..."

if command -v sbatch &> /dev/null; then
    echo "  SLURM detected - submitting batch jobs"
    
    if [ "$QUICK_MODE" = true ]; then
        # Quick smoke test - 1 run, 5 epochs
        echo "  Running quick smoke test..."
        EPOCHS=5 RUNS=1 sbatch --array=0-3 run_full_comparison.slurm
    else
        # Full experiments
        echo "  Submitting full experiment suite..."
        
        # VGG16 on CIFAR-100 (all 8 methods)
        echo "    - VGG16 CIFAR-100 experiments..."
        sbatch --array=0-7 run_full_comparison.slurm
        
        # VGG16 on Tiny ImageNet (all 8 methods)
        echo "    - VGG16 Tiny ImageNet experiments..."
        sbatch --array=8-15 run_full_comparison.slurm
        
        # Ablation studies
        echo "    - Ablation studies..."
        sbatch run_ablation_studies.slurm
        
        # Large model comparison
        echo "    - Large model comparison..."
        sbatch run_large_models.slurm
    fi
    
    echo ""
    echo "  Jobs submitted! Monitor with: squeue -u \$USER"
    echo "  Logs will appear in: logs/"
    
else
    echo "  SLURM not detected - running locally"
    
    if [ "$QUICK_MODE" = true ]; then
        EPOCHS=5
        RUNS=1
    else
        EPOCHS=200
        RUNS=3
    fi
    
    # Run key experiments locally
    echo "  Running baseline experiment..."
    python reproduce.py \
        --experiment large_model \
        --model vgg16 \
        --datasets cifar100 \
        --epochs $EPOCHS \
        --runs $RUNS \
        --output_dir ./results/vgg16_cifar100_baseline \
        --augment --amp
    
    echo "  Running Stochastic Mix experiment..."
    python reproduce.py \
        --experiment large_model \
        --model vgg16 \
        --datasets cifar100 \
        --epochs $EPOCHS \
        --runs $RUNS \
        --use_stochastic_mix \
        --output_dir ./results/vgg16_cifar100_stochastic_mix \
        --augment --amp
fi

# ============================================================================
# Step 5: Generate Results Summary
# ============================================================================
echo ""
echo "[5/5] Generating results summary..."

python << 'EOF'
import os
import glob
import csv

print("\n" + "="*60)
print("RESULTS SUMMARY")
print("="*60)

results_dir = "./results"
if os.path.exists(results_dir):
    results = []
    for exp_dir in sorted(os.listdir(results_dir)):
        csv_files = glob.glob(os.path.join(results_dir, exp_dir, "*.csv"))
        if csv_files:
            with open(csv_files[0], 'r') as f:
                reader = csv.DictReader(f)
                rows = list(reader)
                if rows and 'best_acc' in rows[-1]:
                    acc = float(rows[-1]['best_acc'])
                    results.append((exp_dir, acc))
    
    if results:
        print(f"\n{'Experiment':<50} {'Accuracy':>10}")
        print("-"*62)
        for name, acc in sorted(results, key=lambda x: -x[1]):
            print(f"{name:<50} {acc:>10.2f}%")
    else:
        print("\nNo completed experiments found yet.")
        print("Run 'squeue -u $USER' to check job status.")
else:
    print("\nResults directory not found. Experiments may still be running.")

print("\n" + "="*60)
EOF

echo ""
echo "=============================================="
echo "Reproducibility script completed!"
echo "=============================================="
echo ""
echo "Next steps:"
echo "  1. Monitor jobs: squeue -u \$USER"
echo "  2. View logs: tail -f logs/*.out"
echo "  3. Check results: ls results/"
echo "  4. Compile paper: cd paper && pdflatex main.tex"
echo ""
