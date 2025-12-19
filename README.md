# Adaptive Pooling Innovations for Efficient Deep Learning

**ESE-5390 Machine Learning Systems — Fall 2025**  
**University of Pennsylvania**

---

## 📋 Project Overview

This project presents a comprehensive investigation of **8 adaptive pooling mechanisms** for Convolutional Neural Networks (CNNs). We demonstrate that learnable pooling can significantly outperform static baselines, achieving up to **+2.03% accuracy improvement** on benchmark datasets.

### Key Results

| Method | CIFAR-100 (VGG16) | Tiny ImageNet (VGG16) |
|--------|-------------------|----------------------|
| Baseline (Max Pool) | 72.70% | 59.51% |
| **Stochastic Mix** (ours) | **74.11%** (+1.41%) | **61.54%** (+2.03%) |
| **Attention-Weighted** (ours) | 73.74% (+1.04%) | 61.38% (+1.87%) |

---

## 🚀 Quick Start

### One-Click Reproduction

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/adaptive-pooling-cnn.git
cd adaptive-pooling-cnn

# Run full experiment suite
./reproduce_results.sh

# Or run quick smoke test (5 epochs)
./reproduce_results.sh --quick
```

### Manual Setup

```bash
# 1. Create conda environment
conda create -n pooling python=3.10 -y
conda activate pooling

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run experiments
python reproduce.py --experiment large_model \
    --model vgg16 --datasets cifar100 \
    --epochs 200 --runs 3 \
    --use_stochastic_mix \
    --output_dir ./results/vgg16_stochastic_mix
```

---

## 📁 Project Structure

```
adaptive-pooling-cnn/
├── reproduce.py              # Main experiment script
├── pooling_innovations.py    # All 8 pooling implementations
├── generate_plots.py         # Publication-quality figure generation
├── reproduce_results.sh      # One-click reproducibility script
├── requirements.txt          # Python dependencies
│
├── slurm/                    # SLURM cluster scripts
│   ├── run_full_comparison.slurm
│   ├── run_ablation_studies.slurm
│   └── run_large_models.slurm
│
├── paper/
│   ├── main.tex              # LaTeX paper source
│   ├── main.pdf              # Compiled paper
│   ├── references.bib        # Bibliography
│   └── figures/              # Generated figures
│
└── results/                  # Experiment outputs (gitignored)
```

---

## 🔬 Pooling Methods Implemented

| # | Method | Description | Novel? |
|---|--------|-------------|--------|
| 1 | **Baseline** | Standard Max Pooling | - |
| 2 | **T-Max-Avg** | Threshold-based hybrid | - |
| 3 | **Soft T-Max-Avg** | Differentiable threshold | - |
| 4 | **Channel Adaptive** | Per-channel learned blend | - |
| 5 | **Learnable T** | Learned threshold parameters | - |
| 6 | **Gated** | Spatial attention gates | - |
| 7 | **Attention-Weighted** | Self-attention pooling | ✅ |
| 8 | **Stochastic Mix** | Regularized random blending | ✅ |

---

## 📊 Running Experiments

### Using SLURM (Cluster)

```bash
# Full comparison (8 methods × 2 datasets)
sbatch slurm/run_full_comparison.slurm

# Ablation study (K, T hyperparameters)
sbatch slurm/run_ablation_studies.slurm

# Monitor jobs
squeue -u $USER
```

### Local Execution

```bash
# Single experiment
python reproduce.py \
    --experiment large_model \
    --model vgg16 \
    --datasets cifar100 \
    --epochs 200 \
    --runs 3 \
    --use_attention_weighted \
    --output_dir ./results/vgg16_attention \
    --augment --amp --bf16

# Available pooling flags:
#   --use_tmaxavg
#   --use_soft_tmaxavg
#   --use_channel_adaptive
#   --use_learnable_t
#   --use_gated
#   --use_attention_weighted  (NEW)
#   --use_stochastic_mix      (NEW)
```

---

## 📈 Reproducing Paper Results

### Table 1: VGG16 on CIFAR-100

```bash
# Baseline
python reproduce.py --model vgg16 --datasets cifar100 --epochs 200 --runs 3

# Stochastic Mix (best)
python reproduce.py --model vgg16 --datasets cifar100 --epochs 200 --runs 3 --use_stochastic_mix

# Attention-Weighted
python reproduce.py --model vgg16 --datasets cifar100 --epochs 200 --runs 3 --use_attention_weighted
```

### Ablation Study

```bash
for K in 2 3 4 6; do
  for T in 0.3 0.5 0.7 0.9; do
    python reproduce.py --model resnet18 --datasets cifar100 \
        --use_tmaxavg --K $K --T $T --epochs 200 --runs 3 \
        --output_dir results/ablation_K${K}_T${T}
  done
done
```

---

## 🛠️ Requirements

- Python 3.10+
- PyTorch 2.0+ with CUDA
- NVIDIA GPU (tested on H100, A100)
- 16GB+ GPU memory recommended

---

## 📝 Citation

```bibtex
@techreport{idehpour2025pooling,
  title={Adaptive Pooling Innovations for Efficient Deep Learning},
  author={Idehpour, Parsa},
  institution={University of Pennsylvania},
  year={2025},
  note={ESE-5390 Machine Learning Systems, Fall 2025}
}
```

---

## 📄 Paper

The full paper is available in `paper/main.pdf`. To recompile:

```bash
cd paper
pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex
```

---

## 🏆 Main Findings

1. **Stochastic Mix Pooling** achieves the best results across all configurations
2. **Architecture matters**: VGG16 benefits significantly; ResNet18 is robust to pooling changes
3. **Higher resolution amplifies benefits**: Tiny ImageNet shows larger improvements than CIFAR-100
4. **Negligible overhead**: All methods add <8% training time

---

## 📜 License

MIT License
