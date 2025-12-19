#!/usr/bin/env python3
"""
Reproduction script for T-Max-Avg pooling experiments.
Supports LeNet-5 style models and ChestX architecture.
"""

import os
import random
import argparse
import tempfile
import numpy as np
import pandas as pd

# Configure tempfile to use TMPDIR from environment before any multiprocessing imports
# This prevents multiprocessing conflicts when running parallel SLURM array jobs
if 'TMPDIR' in os.environ:
    tempfile.tempdir = os.environ['TMPDIR']
    # Ensure the directory exists
    os.makedirs(os.environ['TMPDIR'], exist_ok=True)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from torchvision import datasets, transforms
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# Import novel pooling methods
try:
    from pooling_innovations import (
        LearnableTMaxAvgPool2d, SoftTMaxAvgPool2d,
        ChannelAdaptivePool2d, GatedTMaxAvgPool2d
    )
    INNOVATIONS_AVAILABLE = True
except ImportError:
    INNOVATIONS_AVAILABLE = False


# ============================================================================
# Utility Functions
# ============================================================================

def set_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device():
    """Get the best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def setup_for_h100():
    """
    Configure PyTorch for optimal H100 performance.
    """
    if not torch.cuda.is_available():
        return

    # Enable TF32 for faster matrix operations on H100/A100
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # Enable cudnn benchmark for consistent input sizes
    torch.backends.cudnn.benchmark = True

    # Print GPU info
    gpu_name = torch.cuda.get_device_name(0)
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"GPU: {gpu_name}")
    print(f"GPU Memory: {gpu_memory:.1f} GB")

    # Check for H100/A100 specific features
    if "H100" in gpu_name or "A100" in gpu_name:
        print("Detected high-end GPU - enabling optimizations:")
        print("  - TF32 enabled for matmul")
        print("  - cuDNN benchmark enabled")
        print("  - Recommend using --amp for mixed precision")


def get_optimal_batch_size(model_name: str, gpu_memory_gb: float = 80.0) -> int:
    """
    Get recommended batch size based on model and GPU memory.
    Assumes H100 80GB by default.
    """
    # Approximate batch sizes for H100 80GB with mixed precision
    batch_sizes = {
        'lenet': 2048,
        'vgg11': 512,
        'vgg13': 512,
        'vgg16': 384,
        'vgg19': 256,
        'resnet18': 1024,
        'resnet34': 512,
        'resnet50': 384,
        'resnet101': 256,
        'resnet152': 128,
        'wideresnet': 256,
        'wrn28_10': 256,
        'wrn40_2': 512,
    }

    base_batch = batch_sizes.get(model_name, 256)

    # Scale based on available memory (assuming 80GB baseline)
    scale = gpu_memory_gb / 80.0
    return int(base_batch * scale)


# ============================================================================
# Pooling Layers
# ============================================================================

class AvgTopKPool2d(nn.Module):
    """
    Avg-TopK pooling.
    Select top-K values inside each pooling window and average them.
    """
    def __init__(self, kernel_size=2, stride=None, K=2):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride or kernel_size
        self.K = K

    def forward(self, x):
        N, C, H, W = x.shape

        patches = F.unfold(
            x,
            kernel_size=self.kernel_size,
            stride=self.stride
        )

        k2 = self.kernel_size * self.kernel_size
        patches = patches.view(N, C, k2, -1)
        patches = patches.permute(0, 1, 3, 2)

        K_eff = min(self.K, patches.size(-1))
        topk, _ = torch.topk(patches, K_eff, dim=-1)

        avg_vals = topk.mean(dim=-1)

        H_out = (H - self.kernel_size) // self.stride + 1
        W_out = (W - self.kernel_size) // self.stride + 1
        out = avg_vals.view(N, C, H_out, W_out)
        return out


class TMaxAvgPool2d(nn.Module):
    """
    T-Max-Avg pooling.
    - Select top-K values in each window.
    - If max(top-K) >= T -> output max.
    - Else -> output average of top-K.
    """
    def __init__(self, kernel_size=2, stride=None, K=2, T=0.7):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride or kernel_size
        self.K = K
        self.T = T

    def forward(self, x):
        N, C, H, W = x.shape

        patches = F.unfold(
            x,
            kernel_size=self.kernel_size,
            stride=self.stride
        )

        k2 = self.kernel_size * self.kernel_size
        patches = patches.view(N, C, k2, -1)
        patches = patches.permute(0, 1, 3, 2)

        K_eff = min(self.K, patches.size(-1))
        topk, _ = torch.topk(patches, K_eff, dim=-1)

        max_vals, _ = topk.max(dim=-1, keepdim=True)
        avg_vals = topk.mean(dim=-1, keepdim=True)

        T_tensor = torch.tensor(self.T, device=x.device, dtype=x.dtype)
        cond = (max_vals >= T_tensor)

        out_vals = torch.where(cond, max_vals, avg_vals)
        out_vals = out_vals.squeeze(-1)

        H_out = (H - self.kernel_size) // self.stride + 1
        W_out = (W - self.kernel_size) // self.stride + 1
        out = out_vals.view(N, C, H_out, W_out)
        return out


# ============================================================================
# Model Architectures
# ============================================================================

class LeNet5Paper(nn.Module):
    """
    LeNet-5 style CNN close to the paper:
    - 3 conv layers
    - 2 pooling layers (configurable: Avg / Max / Avg-TopK / T-Max-Avg)
    - tanh activations
    - fully connected head: 120 -> 84 -> num_classes
    """
    def __init__(self,
                 pool_layer_factory,
                 pool_kernel_size=2,
                 num_classes=10,
                 in_channels=3):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, 6, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, padding=2)
        self.conv3 = nn.Conv2d(16, 120, kernel_size=5, padding=2)

        self.pool1 = pool_layer_factory(kernel_size=pool_kernel_size)
        self.pool2 = pool_layer_factory(kernel_size=pool_kernel_size)

        self.fc1 = None
        self.fc2 = None

        self._build_fc(num_classes, in_channels)

    def _build_fc(self, num_classes, in_channels):
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, 32, 32)
            feat = self._forward_features(dummy)
            flat_dim = feat.view(1, -1).size(1)

        self.fc1 = nn.Linear(flat_dim, 84)
        self.fc2 = nn.Linear(84, num_classes)

    def _forward_features(self, x):
        x = torch.tanh(self.conv1(x))
        x = self.pool1(x)
        x = torch.tanh(self.conv2(x))
        x = self.pool2(x)
        x = torch.tanh(self.conv3(x))
        return x

    def forward(self, x):
        x = self._forward_features(x)
        x = torch.flatten(x, 1)
        x = torch.tanh(self.fc1(x))
        x = self.fc2(x)
        return x


class ChestX(nn.Module):
    """
    ChestX-style CNN (simplified from the paper's Table 15).
    - Three conv blocks.
    - Last pooling can be MaxPool or T-Max-Avg.
    - CIFAR-10 input: 32x32x3.
    """
    def __init__(self,
                 num_classes: int = 10,
                 use_tmaxavg: bool = False,
                 T: float = 0.7,
                 K: int = 2):
        super().__init__()

        # Block 1: 32x32 -> 16x16
        self.conv1_1 = nn.Conv2d(3, 32, kernel_size=5, padding=2)
        self.conv1_2 = nn.Conv2d(32, 32, kernel_size=5, padding=2)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Block 2: 16x16 -> 8x8
        self.conv2_1 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv2_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Block 3: 8x8 -> 4x4
        self.conv3_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)

        if use_tmaxavg:
            self.pool3 = TMaxAvgPool2d(kernel_size=2, stride=2, K=K, T=T)
        else:
            self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # FC branch
        self.dropout1 = nn.Dropout(p=0.5)
        self.fc1 = nn.Linear(4 * 4 * 128, 1024)
        self.dropout2 = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(1024, 512)
        self.dropout3 = nn.Dropout(p=0.5)
        self.fc3 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1_1(x))
        x = F.relu(self.conv1_2(x))
        x = self.pool1(x)

        x = F.relu(self.conv2_1(x))
        x = F.relu(self.conv2_2(x))
        x = self.pool2(x)

        x = F.relu(self.conv3_1(x))
        x = F.relu(self.conv3_2(x))
        x = self.pool3(x)

        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        x = F.relu(self.fc2(x))
        x = self.dropout3(x)
        x = self.fc3(x)
        return x


# ============================================================================
# Larger Model Architectures
# ============================================================================

class VGGBlock(nn.Module):
    """A VGG-style block with multiple conv layers."""
    def __init__(self, in_channels, out_channels, num_convs, use_bn=True):
        super().__init__()
        layers = []
        for i in range(num_convs):
            layers.append(nn.Conv2d(
                in_channels if i == 0 else out_channels,
                out_channels,
                kernel_size=3,
                padding=1
            ))
            if use_bn:
                layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU(inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class VGGStyle(nn.Module):
    """
    VGG-style CNN with configurable pooling.
    Supports VGG11, VGG13, VGG16, VGG19 configurations.
    """
    # VGG configurations: number of conv layers per block
    CONFIGS = {
        'vgg11': [1, 1, 2, 2, 2],
        'vgg13': [2, 2, 2, 2, 2],
        'vgg16': [2, 2, 3, 3, 3],
        'vgg19': [2, 2, 4, 4, 4],
    }

    def __init__(self,
                 config: str = 'vgg16',
                 num_classes: int = 10,
                 in_channels: int = 3,
                 use_tmaxavg: bool = False,
                 pool_type: str = 'max',  # 'max', 'avg', 'avgtopk', 'tmaxavg'
                 K: int = 2,
                 T: float = 0.7,
                 use_bn: bool = True,
                 dropout: float = 0.5,
                 **kwargs):
        super().__init__()
        self.kwargs = kwargs

        if config not in self.CONFIGS:
            raise ValueError(f"Unknown VGG config: {config}. Choose from {list(self.CONFIGS.keys())}")

        num_convs_per_block = self.CONFIGS[config]
        channels = [64, 128, 256, 512, 512]

        # Build feature extractor
        self.features = nn.ModuleList()
        self.pools = nn.ModuleList()

        in_ch = in_channels
        for i, (num_convs, out_ch) in enumerate(zip(num_convs_per_block, channels)):
            self.features.append(VGGBlock(in_ch, out_ch, num_convs, use_bn))
            # Use T-Max-Avg for the last 2 pooling layers (where it matters most)
            # Use T-Max-Avg for the last 2 pooling layers (where it matters most)
            if use_tmaxavg and i >= 3:
                self.pools.append(TMaxAvgPool2d(kernel_size=2, stride=2, K=K, T=T))
            elif kwargs.get('use_soft_tmaxavg') and i >= 3 and INNOVATIONS_AVAILABLE:
                self.pools.append(SoftTMaxAvgPool2d(
                    kernel_size=2, stride=2, K=K, T=T,
                    temperature=kwargs.get('temperature', 0.1)
                ))
            elif kwargs.get('use_channel_adaptive') and i >= 3 and INNOVATIONS_AVAILABLE:
                self.pools.append(ChannelAdaptivePool2d(
                    kernel_size=2, stride=2, num_channels=out_ch
                ))
            elif kwargs.get('use_learnable_t') and i >= 3 and INNOVATIONS_AVAILABLE:
                self.pools.append(LearnableTMaxAvgPool2d(
                    kernel_size=2, stride=2, K=K, init_T=T, num_channels=out_ch
                ))
            elif kwargs.get('use_gated') and i >= 3 and INNOVATIONS_AVAILABLE:
                self.pools.append(GatedTMaxAvgPool2d(
                    kernel_size=2, stride=2, num_channels=out_ch
                ))
            elif kwargs.get('use_attention_weighted') and i >= 3 and INNOVATIONS_AVAILABLE:
                from pooling_innovations import AttentionWeightedPool2d
                self.pools.append(AttentionWeightedPool2d(
                    kernel_size=2, stride=2, num_channels=out_ch
                ))
            elif kwargs.get('use_stochastic_mix') and i >= 3 and INNOVATIONS_AVAILABLE:
                from pooling_innovations import StochasticMixPool2d
                self.pools.append(StochasticMixPool2d(
                    kernel_size=2, stride=2
                ))
            elif pool_type == 'max':
                self.pools.append(nn.MaxPool2d(kernel_size=2, stride=2))
            elif pool_type == 'avg':
                self.pools.append(nn.AvgPool2d(kernel_size=2, stride=2))
            in_ch = out_ch

        # Adaptive pooling to handle different input sizes
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        for feat, pool in zip(self.features, self.pools):
            x = feat(x)
            x = pool(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


class BasicBlock(nn.Module):
    """Basic residual block for ResNet-18/34."""
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


class Bottleneck(nn.Module):
    """Bottleneck residual block for ResNet-50/101/152."""
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion,
                               kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


class ResNetStyle(nn.Module):
    """
    ResNet-style CNN with configurable pooling.
    Supports ResNet-18, 34, 50, 101, 152.
    """
    CONFIGS = {
        'resnet18': (BasicBlock, [2, 2, 2, 2]),
        'resnet34': (BasicBlock, [3, 4, 6, 3]),
        'resnet50': (Bottleneck, [3, 4, 6, 3]),
        'resnet101': (Bottleneck, [3, 4, 23, 3]),
        'resnet152': (Bottleneck, [3, 8, 36, 3]),
    }

    def __init__(self,
                 config: str = 'resnet18',
                 num_classes: int = 10,
                 in_channels: int = 3,
                 use_tmaxavg: bool = False,
                 pool_type: str = 'max',
                 K: int = 2,
                 T: float = 0.7,
                 **kwargs):
        super().__init__()

        if config not in self.CONFIGS:
            raise ValueError(f"Unknown ResNet config: {config}")

        block, layers = self.CONFIGS[config]
        self.in_channels = 64

        # Initial conv layer (adapted for CIFAR - smaller kernel)
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        # Initial pooling (optional for small images like CIFAR)
        if use_tmaxavg:
            self.initial_pool = TMaxAvgPool2d(kernel_size=2, stride=2, K=K, T=T)
        elif kwargs.get('use_soft_tmaxavg') and INNOVATIONS_AVAILABLE:
            self.initial_pool = SoftTMaxAvgPool2d(
                kernel_size=2, stride=2, K=K, T=T,
                temperature=kwargs.get('temperature', 0.1)
            )
        elif kwargs.get('use_channel_adaptive') and INNOVATIONS_AVAILABLE:
            self.initial_pool = ChannelAdaptivePool2d(
                kernel_size=2, stride=2, num_channels=64
            )
        elif kwargs.get('use_learnable_t') and INNOVATIONS_AVAILABLE:
            self.initial_pool = LearnableTMaxAvgPool2d(
                kernel_size=2, stride=2, K=K, init_T=T, num_channels=64
            )
        elif kwargs.get('use_gated') and INNOVATIONS_AVAILABLE:
            self.initial_pool = GatedTMaxAvgPool2d(
                kernel_size=2, stride=2, num_channels=64
            )
        elif kwargs.get('use_attention_weighted') and INNOVATIONS_AVAILABLE:
            from pooling_innovations import AttentionWeightedPool2d
            self.initial_pool = AttentionWeightedPool2d(
                kernel_size=2, stride=2, num_channels=64
            )
        elif kwargs.get('use_stochastic_mix') and INNOVATIONS_AVAILABLE:
            from pooling_innovations import StochasticMixPool2d
            self.initial_pool = StochasticMixPool2d(
                kernel_size=2, stride=2
            )
        elif pool_type == 'max':
            self.initial_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        elif pool_type == 'avg':
            self.initial_pool = nn.AvgPool2d(kernel_size=2, stride=2)
        elif pool_type == 'avgtopk':
            self.initial_pool = AvgTopKPool2d(kernel_size=2, stride=2, K=K)
        else:
            self.initial_pool = TMaxAvgPool2d(kernel_size=2, stride=2, K=K, T=T)

        # IMPORTANT: Enable initial pooling so that pooling variants (baseline vs
        # T-Max-Avg vs soft vs channel-adaptive, etc.) actually affect the
        # forward pass. Previously this was disabled for CIFAR (32x32), which
        # meant all ResNet variants behaved identically and produced identical
        # results. Using a 2x2 stride-2 pool here matches the intent of the
        # architecture search and ensures pooling choices change the network.
        self.use_initial_pool = True

        # Residual layers
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        # Global average pooling and classifier
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        # Initialize weights
        self._initialize_weights()

    def _make_layer(self, block, out_channels, num_blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * block.expansion),
            )

        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * block.expansion
        for _ in range(1, num_blocks):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        if self.use_initial_pool:
            x = self.initial_pool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


class WideResNet(nn.Module):
    """
    Wide ResNet for CIFAR.
    WRN-depth-width, e.g., WRN-28-10.
    """
    def __init__(self,
                 depth: int = 28,
                 widen_factor: int = 10,
                 num_classes: int = 10,
                 in_channels: int = 3,
                 dropout: float = 0.3,
                 use_tmaxavg: bool = False,
                 pool_type: str = 'avg',
                 K: int = 2,
                 T: float = 0.7,
                 **kwargs):
        super().__init__()
        self.use_tmaxavg = use_tmaxavg
        self.pool_type = pool_type
        self.K = K
        self.T = T
        self.kwargs = kwargs

        n_channels = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]
        assert (depth - 4) % 6 == 0, "Depth should be 6n+4"
        n = (depth - 4) // 6

        self.conv1 = nn.Conv2d(in_channels, n_channels[0], kernel_size=3,
                               stride=1, padding=1, bias=False)

        self.block1 = self._make_layer(n_channels[0], n_channels[1], n, 1, dropout)
        self.block2 = self._make_layer(n_channels[1], n_channels[2], n, 2, dropout)
        self.block3 = self._make_layer(n_channels[2], n_channels[3], n, 2, dropout)

        self.bn = nn.BatchNorm2d(n_channels[3])
        self.relu = nn.ReLU(inplace=True)

        # Final pooling
        if self.use_tmaxavg:
            # Check for innovative pooling types in kwargs
            if self.kwargs.get('use_soft_tmaxavg'):
                from pooling_innovations import create_soft_tmaxavg
                self.pool = create_soft_tmaxavg(kernel_size=8, stride=1, K=self.K, T=self.T,
                                         temperature=self.kwargs.get('temperature', 0.1),
                                         learnable_T=self.kwargs.get('learnable_T', False))
            elif self.kwargs.get('use_channel_adaptive'):
                from pooling_innovations import create_channel_adaptive
                self.pool = create_channel_adaptive(kernel_size=8, stride=1, num_channels=n_channels[3])
            elif self.kwargs.get('use_gated'):
                from pooling_innovations import create_gated_tmaxavg
                self.pool = create_gated_tmaxavg(kernel_size=8, stride=1, num_channels=n_channels[3])
            elif self.kwargs.get('use_learnable_t'):
                from pooling_innovations import create_learnable_tmaxavg
                self.pool = create_learnable_tmaxavg(kernel_size=8, stride=1, K=self.K,
                                              num_channels=n_channels[3])
            else:
                self.pool = TMaxAvgPool2d(kernel_size=8, stride=1, K=self.K, T=self.T)
        elif self.pool_type == 'max':
            self.pool = nn.AdaptiveMaxPool2d((1, 1))
        else:
            self.pool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc = nn.Linear(n_channels[3], num_classes)

        self._initialize_weights()

    def _make_layer(self, in_channels, out_channels, num_blocks, stride, dropout):
        layers = []
        for i in range(num_blocks):
            layers.append(WideBasicBlock(
                in_channels if i == 0 else out_channels,
                out_channels,
                stride if i == 0 else 1,
                dropout
            ))
        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


class WideBasicBlock(nn.Module):
    """Wide ResNet basic block."""
    def __init__(self, in_channels, out_channels, stride=1, dropout=0.3):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.dropout = nn.Dropout(p=dropout)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1,
                                      stride=stride, bias=False)

    def forward(self, x):
        out = self.bn1(x)
        out = self.relu1(out)
        out = self.conv1(out)
        out = self.dropout(out)
        out = self.bn2(out)
        out = self.relu2(out)
        out = self.conv2(out)
        out += self.shortcut(x)
        return out


# ============================================================================
# Model Factories
# ============================================================================

def make_avg_pool_model(pool_kernel_size=2, num_classes=10, in_channels=3):
    def factory(kernel_size):
        return nn.AvgPool2d(kernel_size=kernel_size, stride=kernel_size)
    return LeNet5Paper(factory, pool_kernel_size, num_classes, in_channels)


def make_max_pool_model(pool_kernel_size=2, num_classes=10, in_channels=3):
    def factory(kernel_size):
        return nn.MaxPool2d(kernel_size=kernel_size, stride=kernel_size)
    return LeNet5Paper(factory, pool_kernel_size, num_classes, in_channels)


def make_avgtopk_model(pool_kernel_size=2, K=2, num_classes=10, in_channels=3):
    def factory(kernel_size):
        return AvgTopKPool2d(kernel_size=kernel_size, stride=kernel_size, K=K)
    return LeNet5Paper(factory, pool_kernel_size, num_classes, in_channels)


def make_tmaxavg_model(pool_kernel_size=2, K=2, T=0.7, num_classes=10, in_channels=3):
    def factory(kernel_size):
        return TMaxAvgPool2d(kernel_size=kernel_size, stride=kernel_size, K=K, T=T)
    return LeNet5Paper(factory, pool_kernel_size, num_classes, in_channels)


# Factories for larger models

def make_vgg_model(config='vgg16', num_classes=10, in_channels=3,
                   pool_type='max', use_tmaxavg=False, K=2, T=0.7, **kwargs):
    """Create a VGG-style model."""
    return VGGStyle(
        config=config,
        num_classes=num_classes,
        in_channels=in_channels,
        use_tmaxavg=use_tmaxavg,
        pool_type=pool_type,
        K=K,
        T=T,
        **kwargs
    )


def make_resnet_model(config='resnet18', num_classes=10, in_channels=3,
                      pool_type='max', use_tmaxavg=False, K=2, T=0.7, **kwargs):
    """Create a ResNet-style model."""
    return ResNetStyle(
        config=config,
        num_classes=num_classes,
        in_channels=in_channels,
        use_tmaxavg=use_tmaxavg,
        pool_type=pool_type,
        K=K,
        T=T,
        **kwargs
    )


def make_wideresnet_model(depth=28, widen_factor=10, num_classes=10, in_channels=3,
                          pool_type='avg', use_tmaxavg=False, K=2, T=0.7, **kwargs):
    """Create a Wide ResNet model."""
    return WideResNet(
        depth=depth,
        widen_factor=widen_factor,
        num_classes=num_classes,
        in_channels=in_channels,
        use_tmaxavg=use_tmaxavg,
        pool_type=pool_type,
        K=K,
        T=T,
        **kwargs
    )


# Model registry for easy access
MODEL_REGISTRY = {
    'lenet': make_avg_pool_model,  # LeNet with avg pooling
    'lenet_max': make_max_pool_model,
    'lenet_avgtopk': make_avgtopk_model,
    'lenet_tmaxavg': make_tmaxavg_model,
    'vgg11': lambda **kw: make_vgg_model(config='vgg11', **kw),
    'vgg13': lambda **kw: make_vgg_model(config='vgg13', **kw),
    'vgg16': lambda **kw: make_vgg_model(config='vgg16', **kw),
    'vgg19': lambda **kw: make_vgg_model(config='vgg19', **kw),
    'resnet18': lambda **kw: make_resnet_model(config='resnet18', **kw),
    'resnet34': lambda **kw: make_resnet_model(config='resnet34', **kw),
    'resnet50': lambda **kw: make_resnet_model(config='resnet50', **kw),
    'resnet101': lambda **kw: make_resnet_model(config='resnet101', **kw),
    'wideresnet': make_wideresnet_model,
    'wrn28_10': lambda **kw: make_wideresnet_model(depth=28, widen_factor=10, **kw),
    'wrn40_2': lambda **kw: make_wideresnet_model(depth=40, widen_factor=2, **kw),
}


# ============================================================================
# Data Loading
# ============================================================================

# Basic transforms
transform_cifar = transforms.Compose([
    transforms.ToTensor(),
])

# Augmented transforms for larger models
transform_cifar_augmented = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_cifar_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_mnist = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
])

# Tiny ImageNet (64x64)
transform_tiny_augmented = transforms.Compose([
    transforms.RandomCrop(64, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4802, 0.4481, 0.3975), (0.2302, 0.2265, 0.2262)),
])

transform_tiny_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4802, 0.4481, 0.3975), (0.2302, 0.2265, 0.2262)),
])


def get_dataloaders(dataset_name: str,
                    batch_size: int = 128,
                    data_root: str = "./data",
                    num_workers: int = 4,
                    augment: bool = False):
    """
    Get data loaders for the specified dataset.

    Args:
        dataset_name: Name of the dataset (cifar10, cifar100, mnist)
        batch_size: Batch size for training and testing
        data_root: Root directory for data
        num_workers: Number of data loading workers
        augment: Whether to use data augmentation (recommended for larger models)
    """
    dataset_name = dataset_name.lower()

    if dataset_name == "cifar10":
        train_transform = transform_cifar_augmented if augment else transform_cifar
        test_transform = transform_cifar_test if augment else transform_cifar
        train_set = datasets.CIFAR10(
            root=data_root, train=True, download=True, transform=train_transform
        )
        test_set = datasets.CIFAR10(
            root=data_root, train=False, download=True, transform=test_transform
        )
        num_classes = 10
        in_channels = 3

    elif dataset_name == "cifar100":
        train_transform = transform_cifar_augmented if augment else transform_cifar
        test_transform = transform_cifar_test if augment else transform_cifar
        train_set = datasets.CIFAR100(
            root=data_root, train=True, download=True, transform=train_transform
        )
        test_set = datasets.CIFAR100(
            root=data_root, train=False, download=True, transform=test_transform
        )
        num_classes = 100
        in_channels = 3

    elif dataset_name == "mnist":
        train_set = datasets.MNIST(
            root=data_root, train=True, download=True, transform=transform_mnist
        )
        test_set = datasets.MNIST(
            root=data_root, train=False, download=True, transform=transform_mnist
        )
        num_classes = 10
        in_channels = 3

    elif dataset_name == "tinyimagenet":
        # Assumes data is prepared by prepare_tiny_imagenet.py
        tiny_root = os.path.join(data_root, "tiny-imagenet-200")
        if not os.path.exists(tiny_root):
            raise FileNotFoundError(f"Tiny ImageNet not found at {tiny_root}. Run prepare_tiny_imagenet.py first.")
            
        train_transform = transform_tiny_augmented if augment else transform_tiny_test
        test_transform = transform_tiny_test
        
        train_set = datasets.ImageFolder(root=os.path.join(tiny_root, "train"), transform=train_transform)
        test_set = datasets.ImageFolder(root=os.path.join(tiny_root, "val"), transform=test_transform)
        
        num_classes = 200
        in_channels = 3

    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers,
        pin_memory=True
    )
    test_loader = DataLoader(
        test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, test_loader, num_classes, in_channels


# ============================================================================
# Training & Evaluation
# ============================================================================

def evaluate(model, loader, criterion, device):
    model.eval()
    correct = 0
    total = 0
    loss_sum = 0.0

    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            labels = labels.to(device)

            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss_sum += loss.item() * imgs.size(0)

            _, preds = outputs.max(1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()

    acc = 100.0 * correct / total
    avg_loss = loss_sum / total
    return acc, avg_loss


def evaluate_amp(model, loader, criterion, device, autocast_dtype=torch.float16):
    """Evaluate model with automatic mixed precision."""
    model.eval()
    correct = 0
    total = 0
    loss_sum = 0.0

    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            with torch.amp.autocast('cuda', dtype=autocast_dtype):
                outputs = model(imgs)
                loss = criterion(outputs, labels)

            loss_sum += loss.item() * imgs.size(0)
            _, preds = outputs.max(1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()

    acc = 100.0 * correct / total
    avg_loss = loss_sum / total
    return acc, avg_loss


def train_single_run(model, train_loader, test_loader, device,
                     epochs: int = 50, lr: float = 1e-3, verbose: bool = True):
    """Train one model instance and return final test accuracy."""
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_acc = 0.0

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0

        for imgs, labels in train_loader:
            imgs = imgs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * imgs.size(0)

        train_loss = running_loss / len(train_loader.dataset)
        test_acc, test_loss = evaluate(model, test_loader, criterion, device)

        if verbose:
            print(f"Epoch [{epoch:02d}/{epochs:02d}] "
                  f"TrainLoss={train_loss:.4f} "
                  f"TestLoss={test_loss:.4f} "
                  f"TestAcc={test_acc:.2f}%")

        best_acc = max(best_acc, test_acc)

    return best_acc


def run_experiment(build_model_fn, dataset_name: str, pool_kernel_size: int,
                   device, epochs: int = 50, runs: int = 6, lr: float = 1e-3,
                   batch_size: int = 128, **model_kwargs):
    """Run the same setup multiple times and report average accuracy."""
    acc_list = []

    train_loader, test_loader, num_classes, in_channels = get_dataloaders(
        dataset_name, batch_size=batch_size
    )

    for run in range(1, runs + 1):
        print(f"\n=== Run {run}/{runs} ===")
        set_seed(42 + run)

        model = build_model_fn(
            pool_kernel_size=pool_kernel_size,
            num_classes=num_classes,
            in_channels=in_channels,
            **model_kwargs
        )

        acc = train_single_run(
            model, train_loader, test_loader, device,
            epochs=epochs, lr=lr, verbose=True
        )
        acc_list.append(acc)
        print(f"[Run {run}] Best test accuracy: {acc:.2f}%")

    acc_array = np.array(acc_list)
    mean_acc = acc_array.mean()
    std_acc = acc_array.std()

    print("\n=== Summary ===")
    print("Accuracies:", acc_list)
    print(f"Mean accuracy: {mean_acc:.2f}%  Std: {std_acc:.2f}%")

    return acc_list, mean_acc, std_acc


# ============================================================================
# Table 14 Configuration
# ============================================================================

TABLE14_CONFIG = {
    "cifar100": {
        "Avg": {"builder": make_avg_pool_model, "pool_kernel_size": 3, "kwargs": {}},
        "Max": {"builder": make_max_pool_model, "pool_kernel_size": 3, "kwargs": {}},
        "AvgTopK": {"builder": make_avgtopk_model, "pool_kernel_size": 4, "kwargs": {"K": 6}},
        "TMaxAvg": {"builder": make_tmaxavg_model, "pool_kernel_size": 4, "kwargs": {"K": 6, "T": 0.7}},
    },
    "cifar10": {
        "Avg": {"builder": make_avg_pool_model, "pool_kernel_size": 3, "kwargs": {}},
        "Max": {"builder": make_max_pool_model, "pool_kernel_size": 3, "kwargs": {}},
        "AvgTopK": {"builder": make_avgtopk_model, "pool_kernel_size": 4, "kwargs": {"K": 6}},
        "TMaxAvg": {"builder": make_tmaxavg_model, "pool_kernel_size": 4, "kwargs": {"K": 6, "T": 0.7}},
    },
    "mnist": {
        "Avg": {"builder": make_avg_pool_model, "pool_kernel_size": 3, "kwargs": {}},
        "Max": {"builder": make_max_pool_model, "pool_kernel_size": 3, "kwargs": {}},
        "AvgTopK": {"builder": make_avgtopk_model, "pool_kernel_size": 3, "kwargs": {"K": 1}},
        "TMaxAvg": {"builder": make_tmaxavg_model, "pool_kernel_size": 3, "kwargs": {"K": 3, "T": 0.8}},
    },
}


def run_table14_experiments(device, epochs: int = 50, runs: int = 6,
                            batch_size: int = 128, lr: float = 1e-3,
                            datasets_to_run=None):
    """Run all Table-14-style best configs and collect stats."""
    records = []

    if datasets_to_run is None:
        datasets_to_run = list(TABLE14_CONFIG.keys())

    for dataset_name in datasets_to_run:
        if dataset_name not in TABLE14_CONFIG:
            print(f"Warning: {dataset_name} not in TABLE14_CONFIG, skipping.")
            continue

        cfg = TABLE14_CONFIG[dataset_name]
        print(f"\n######## Dataset = {dataset_name.upper()} ########")

        for method_name, mc in cfg.items():
            print(f"\n>>> Method: {method_name}")

            acc_list, mean_acc, std_acc = run_experiment(
                build_model_fn=mc["builder"],
                dataset_name=dataset_name,
                pool_kernel_size=mc["pool_kernel_size"],
                device=device,
                epochs=epochs,
                runs=runs,
                lr=lr,
                batch_size=batch_size,
                **mc["kwargs"]
            )

            rec = {
                "dataset": dataset_name,
                "method": method_name,
                "pool_size": mc["pool_kernel_size"],
                "extra": mc["kwargs"],
                "mean_acc": mean_acc,
                "std_acc": std_acc,
                "all_runs": acc_list,
            }
            records.append(rec)

    df = pd.DataFrame(records)
    return df


# ============================================================================
# ChestX 5-Fold Training
# ============================================================================

def get_cifar10_dataset(root="./data"):
    train_set = datasets.CIFAR10(
        root=root, train=True, download=True, transform=transform_cifar
    )
    test_set = datasets.CIFAR10(
        root=root, train=False, download=True, transform=transform_cifar
    )
    return train_set, test_set


def build_loader_from_subset(dataset, indices, batch_size=128, shuffle=True, num_workers=4):
    subset = torch.utils.data.Subset(dataset, indices)
    loader = DataLoader(
        subset, batch_size=batch_size, shuffle=shuffle,
        num_workers=num_workers, pin_memory=True
    )
    return loader


def train_chestx_on_split(train_loader, val_loader, device, use_tmaxavg: bool,
                          T: float = 0.7, K: int = 2, epochs: int = 50, lr: float = 1e-3):
    model = ChestX(
        num_classes=10, use_tmaxavg=use_tmaxavg, T=T, K=K
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_val_acc = 0.0

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0

        for imgs, labels in train_loader:
            imgs = imgs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * imgs.size(0)

        train_loss = running_loss / len(train_loader.dataset)
        val_acc, val_loss = evaluate(model, val_loader, criterion, device)
        print(f"[Epoch {epoch:02d}] TrainLoss={train_loss:.4f} "
              f"ValLoss={val_loss:.4f} ValAcc={val_acc:.2f}%")

        best_val_acc = max(best_val_acc, val_acc)

    print(f"Best val accuracy on this fold: {best_val_acc:.2f}%")
    return model


def train_5fold_chestx(device, use_tmaxavg: bool, T: float = 0.7, K: int = 2,
                       epochs: int = 50, lr: float = 1e-3, batch_size: int = 128,
                       n_splits: int = 5):
    full_train, test_set = get_cifar10_dataset()
    num_samples = len(full_train)
    indices = np.arange(num_samples)

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    models = []

    for fold_id, (train_idx, val_idx) in enumerate(kf.split(indices), start=1):
        print(f"\n========== Fold {fold_id}/{n_splits} ==========")
        set_seed(100 + fold_id)

        train_loader = build_loader_from_subset(
            full_train, train_idx, batch_size=batch_size, shuffle=True
        )
        val_loader = build_loader_from_subset(
            full_train, val_idx, batch_size=batch_size, shuffle=False
        )

        model = train_chestx_on_split(
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            use_tmaxavg=use_tmaxavg,
            T=T,
            K=K,
            epochs=epochs,
            lr=lr,
        )
        models.append(model)

    test_loader = DataLoader(
        test_set, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True
    )
    return models, test_loader


def ensemble_predict(models, loader, device, voting: str = "soft"):
    """
    voting = "soft": average probabilities.
    voting = "hard": majority vote on class labels.
    """
    for m in models:
        m.eval()

    all_targets = []
    all_preds = []

    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            labels = labels.to(device)
            all_targets.append(labels.cpu().numpy())

            logits_list = []
            preds_list = []

            for m in models:
                logits = m(imgs)
                logits_list.append(logits)
                _, p = logits.max(1)
                preds_list.append(p)

            if voting == "soft":
                probs = [F.softmax(l, dim=1) for l in logits_list]
                avg_probs = torch.stack(probs, dim=0).mean(dim=0)
                _, final_pred = avg_probs.max(1)
            elif voting == "hard":
                stacked = torch.stack(preds_list, dim=0)
                final_pred, _ = torch.mode(stacked, dim=0)
            else:
                raise ValueError("voting must be 'soft' or 'hard'")

            all_preds.append(final_pred.cpu().numpy())

    y_true = np.concatenate(all_targets, axis=0)
    y_pred = np.concatenate(all_preds, axis=0)
    return y_true, y_pred


def compute_metrics(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred) * 100.0
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    return {
        "accuracy": acc,
        "precision": prec * 100.0,
        "recall": rec * 100.0,
        "f1": f1 * 100.0,
    }


def evaluate_chestx_ensembles(models, test_loader, device):
    single_metrics = []
    for m in models:
        y_true, y_pred = ensemble_predict([m], test_loader, device, voting="hard")
        single_metrics.append(compute_metrics(y_true, y_pred))

    avg_single = {
        k: np.mean([m[k] for m in single_metrics])
        for k in single_metrics[0].keys()
    }

    y_true_h, y_pred_h = ensemble_predict(models, test_loader, device, voting="hard")
    metrics_h = compute_metrics(y_true_h, y_pred_h)

    y_true_s, y_pred_s = ensemble_predict(models, test_loader, device, voting="soft")
    metrics_s = compute_metrics(y_true_s, y_pred_s)

    return avg_single, metrics_h, metrics_s


# ============================================================================
# Grid Search Functions
# ============================================================================

def sweep_T_for_dataset(dataset_name: str, pool_kernel_size: int, K: int,
                        T_values, device, epochs: int = 20, runs: int = 6,
                        batch_size: int = 128, lr: float = 1e-3):
    """For a given dataset + (pool_size, K), try multiple T values."""
    records = []

    for T in T_values:
        print(f"\n######## dataset={dataset_name}, "
              f"pool={pool_kernel_size}, K={K}, T={T:.2f} ########")

        acc_list, mean_acc, std_acc = run_experiment(
            build_model_fn=make_tmaxavg_model,
            dataset_name=dataset_name,
            pool_kernel_size=pool_kernel_size,
            device=device,
            epochs=epochs,
            runs=runs,
            lr=lr,
            batch_size=batch_size,
            K=K,
            T=T,
        )

        records.append({
            "dataset": dataset_name,
            "pool_size": pool_kernel_size,
            "K": K,
            "T": T,
            "mean_acc": mean_acc,
            "std_acc": std_acc,
            "all_runs": acc_list,
        })

    df = pd.DataFrame(records)
    return df


def grid_search_tmaxavg(dataset_name: str, pool_sizes, K_values, T_values,
                        device, epochs: int = 20, runs: int = 6,
                        batch_size: int = 128, lr: float = 1e-3):
    """Grid search over (pool_size, K), selecting best T for each pair."""
    summary_rows = []

    for pool in pool_sizes:
        for K in K_values:
            print(f"\n==============================")
            print(f"Grid search for dataset={dataset_name}, "
                  f"pool_size={pool}, K={K}")
            print(f"==============================")

            df_T = sweep_T_for_dataset(
                dataset_name=dataset_name,
                pool_kernel_size=pool,
                K=K,
                T_values=T_values,
                device=device,
                epochs=epochs,
                runs=runs,
                batch_size=batch_size,
                lr=lr,
            )

            best_row = df_T.loc[df_T["mean_acc"].idxmax()]

            summary_rows.append({
                "dataset": dataset_name,
                "pool_size": pool,
                "K": K,
                "best_T": best_row["T"],
                "best_acc": best_row["mean_acc"],
                "best_std": best_row["std_acc"],
            })

    df_summary = pd.DataFrame(summary_rows)
    return df_summary


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="T-Max-Avg Pooling Experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run Table 14 experiments (LeNet-5 style)
  python reproduce.py --experiment table14

  # Train a larger model (ResNet-18 with T-Max-Avg)
  python reproduce.py --experiment large_model --model resnet18 --use_tmaxavg --augment

  # Train VGG-16 on CIFAR-100
  python reproduce.py --experiment large_model --model vgg16 --datasets cifar100 --epochs 200

  # Train Wide ResNet with T-Max-Avg
  python reproduce.py --experiment large_model --model wrn28_10 --use_tmaxavg --K 4 --T 0.7

Available models: lenet, vgg11, vgg13, vgg16, vgg19, resnet18, resnet34, resnet50,
                  resnet101, wideresnet, wrn28_10, wrn40_2
        """
    )

    # Experiment selection
    parser.add_argument("--experiment", type=str, default="table14",
                        choices=["table14", "chestx", "grid_search", "large_model"],
                        help="Which experiment to run")

    # Model selection (for large_model experiment)
    parser.add_argument("--model", type=str, default="resnet18",
                        choices=list(MODEL_REGISTRY.keys()),
                        help="Model architecture to use (for large_model experiment)")

    # Pooling configuration
    parser.add_argument("--pool_type", type=str, default="max",
                        choices=["max", "avg", "avgtopk", "tmaxavg"],
                        help="Pooling type to use")
    parser.add_argument("--use_tmaxavg", action="store_true",
                        help="Use T-Max-Avg pooling in key layers")
    parser.add_argument("--K", type=int, default=2,
                        help="K parameter for TopK/T-Max-Avg pooling")
    parser.add_argument("--T", type=float, default=0.7,
                        help="Threshold T for T-Max-Avg pooling")

    # Novel pooling methods
    parser.add_argument("--use_soft_tmaxavg", action="store_true",
                        help="Use Soft T-Max-Avg with temperature-scaled blending")
    parser.add_argument("--use_channel_adaptive", action="store_true",
                        help="Use Channel Adaptive pooling (learned per-channel blend)")
    parser.add_argument("--use_learnable_t", action="store_true",
                        help="Use learnable T in T-Max-Avg pooling")
    parser.add_argument("--use_gated", action="store_true",
                        help="Use Gated T-Max-Avg pooling")
    parser.add_argument("--use_attention_weighted", action="store_true",
                        help="Use Attention-Weighted pooling (NEW)")
    parser.add_argument("--use_stochastic_mix", action="store_true",
                        help="Use Stochastic Mix pooling (NEW)")
    parser.add_argument("--temperature", type=float, default=0.1,
                        help="Temperature for Soft T-Max-Avg (lower = sharper transition)")

    # Dataset configuration
    parser.add_argument("--datasets", type=str, nargs="+",
                        default=["cifar10", "cifar100", "mnist"],
                        help="Datasets to use")
    parser.add_argument("--augment", action="store_true",
                        help="Use data augmentation (recommended for larger models)")

    # Training configuration
    parser.add_argument("--epochs", type=int, default=50,
                        help="Number of training epochs")
    parser.add_argument("--runs", type=int, default=6,
                        help="Number of runs per configuration")
    parser.add_argument("--batch_size", type=int, default=128,
                        help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=5e-4,
                        help="Weight decay for optimizer")
    parser.add_argument("--scheduler", type=str, default="cosine",
                        choices=["none", "step", "cosine"],
                        help="Learning rate scheduler")

    # Output configuration
    parser.add_argument("--output_dir", type=str, default="./results",
                        help="Directory to save results")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--save_model", action="store_true",
                        help="Save trained model checkpoints")

    # H100 optimization arguments
    parser.add_argument("--amp", action="store_true",
                        help="Use automatic mixed precision (FP16) for faster training on H100")
    parser.add_argument("--bf16", action="store_true",
                        help="Use BFloat16 instead of FP16 for AMP (better for H100)")
    parser.add_argument("--compile", action="store_true",
                        help="Use torch.compile for optimized kernels (PyTorch 2.0+)")
    parser.add_argument("--auto_batch", action="store_true",
                        help="Automatically determine optimal batch size for GPU")
    parser.add_argument("--num_workers", type=int, default=8,
                        help="Number of data loading workers (increase for H100)")

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Set seed
    set_seed(args.seed)

    # Get device
    device = get_device()
    print(f"Using device: {device}")

    if args.experiment == "table14":
        print("\n" + "=" * 60)
        print("Running Table 14 Experiments")
        print("=" * 60)

        df_results = run_table14_experiments(
            device=device,
            epochs=args.epochs,
            runs=args.runs,
            batch_size=args.batch_size,
            lr=args.lr,
            datasets_to_run=args.datasets,
        )

        output_file = os.path.join(args.output_dir, "table14_results.csv")
        df_results.to_csv(output_file, index=False)
        print(f"\nResults saved to {output_file}")
        print(df_results)

    elif args.experiment == "chestx":
        print("\n" + "=" * 60)
        print("Running ChestX Experiments")
        print("=" * 60)

        # Train baseline ChestX
        print("\n--- Training ChestX (baseline) ---")
        models_base, test_loader = train_5fold_chestx(
            device=device,
            use_tmaxavg=False,
            epochs=args.epochs,
            lr=args.lr,
            batch_size=args.batch_size,
            n_splits=5,
        )

        avg_single_base, hard_base, soft_base = evaluate_chestx_ensembles(
            models_base, test_loader, device
        )

        print("\n=== ChestX (baseline) ===")
        print("Average single-model (over folds):", avg_single_base)
        print("5-fold Hard Voting Ensemble:", hard_base)
        print("5-fold Soft Voting Ensemble:", soft_base)

        # Train ChestX(T-Max-Avg)
        print("\n--- Training ChestX(T-Max-Avg) ---")
        models_tmax, test_loader = train_5fold_chestx(
            device=device,
            use_tmaxavg=True,
            T=0.7,
            K=2,
            epochs=args.epochs,
            lr=args.lr,
            batch_size=args.batch_size,
            n_splits=5,
        )

        avg_single_tmax, hard_tmax, soft_tmax = evaluate_chestx_ensembles(
            models_tmax, test_loader, device
        )

        print("\n=== ChestX(T-Max-Avg(2,2,0.7)) ===")
        print("Average single-model (over folds):", avg_single_tmax)
        print("5-fold Hard Voting Ensemble:", hard_tmax)
        print("5-fold Soft Voting Ensemble:", soft_tmax)

        # Save results
        results = {
            "baseline": {"single": avg_single_base, "hard": hard_base, "soft": soft_base},
            "tmaxavg": {"single": avg_single_tmax, "hard": hard_tmax, "soft": soft_tmax},
        }
        output_file = os.path.join(args.output_dir, "chestx_results.csv")
        pd.DataFrame([
            {"model": "ChestX", "type": "single", **avg_single_base},
            {"model": "ChestX", "type": "hard", **hard_base},
            {"model": "ChestX", "type": "soft", **soft_base},
            {"model": "ChestX(T-Max-Avg)", "type": "single", **avg_single_tmax},
            {"model": "ChestX(T-Max-Avg)", "type": "hard", **hard_tmax},
            {"model": "ChestX(T-Max-Avg)", "type": "soft", **soft_tmax},
        ]).to_csv(output_file, index=False)
        print(f"\nResults saved to {output_file}")

    elif args.experiment == "grid_search":
        print("\n" + "=" * 60)
        print("Running Grid Search Experiments")
        print("=" * 60)

        pool_sizes = [2, 3, 4]
        K_values = [1, 2, 3, 6]
        T_values = [0.3, 0.5, 0.7, 0.8, 0.9]

        all_results = []

        for dataset_name in args.datasets:
            print(f"\n>>> Grid search for {dataset_name}")
            df_grid = grid_search_tmaxavg(
                dataset_name=dataset_name,
                pool_sizes=pool_sizes,
                K_values=K_values,
                T_values=T_values,
                device=device,
                epochs=args.epochs,
                runs=args.runs,
                batch_size=args.batch_size,
                lr=args.lr,
            )
            all_results.append(df_grid)
            print(df_grid)

        df_all = pd.concat(all_results, ignore_index=True)
        output_file = os.path.join(args.output_dir, "grid_search_results.csv")
        df_all.to_csv(output_file, index=False)
        print(f"\nResults saved to {output_file}")

    elif args.experiment == "large_model":
        print("\n" + "=" * 60)
        print(f"Training Large Model: {args.model}")
        print("=" * 60)

        # Setup H100 optimizations
        setup_for_h100()

        # Auto-determine batch size if not specified or if using default
        if args.batch_size == 128 and torch.cuda.is_available():
            gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
            recommended_batch = get_optimal_batch_size(args.model, gpu_mem)
            if args.amp:
                recommended_batch = int(recommended_batch * 1.5)  # AMP allows larger batches
            print(f"Recommended batch size for {args.model}: {recommended_batch}")
            if args.auto_batch:
                args.batch_size = recommended_batch
                print(f"Using auto batch size: {args.batch_size}")

        for dataset_name in args.datasets:
            print(f"\n>>> Dataset: {dataset_name}")

            # Get data loaders with augmentation if requested
            # Use more workers for H100 to keep GPU fed
            num_workers = args.num_workers if hasattr(args, 'num_workers') else 8
            train_loader, test_loader, num_classes, in_channels = get_dataloaders(
                dataset_name,
                batch_size=args.batch_size,
                augment=args.augment,
                num_workers=num_workers
            )

            all_results = []

            for run in range(1, args.runs + 1):
                print(f"\n=== Run {run}/{args.runs} ===")
                set_seed(args.seed + run)

                # Create model
                model_fn = MODEL_REGISTRY[args.model]
                model = model_fn(
                    num_classes=num_classes,
                    in_channels=in_channels,
                    pool_type=args.pool_type,
                    use_tmaxavg=args.use_tmaxavg,
                    K=args.K,
                    T=args.T,
                    use_soft_tmaxavg=args.use_soft_tmaxavg,
                    use_channel_adaptive=args.use_channel_adaptive,
                    use_learnable_t=args.use_learnable_t,
                    use_gated=args.use_gated,
                    use_attention_weighted=args.use_attention_weighted,
                    use_stochastic_mix=args.use_stochastic_mix,
                    temperature=args.temperature,
                )
                model = model.to(device)

                # Compile model for H100 (PyTorch 2.0+)
                if args.compile and hasattr(torch, 'compile'):
                    print("Compiling model with torch.compile (inductor backend)...")
                    model = torch.compile(model, mode="reduce-overhead")

                # Count parameters
                total_params = sum(p.numel() for p in model.parameters())
                trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
                print(f"Model: {args.model}")
                print(f"Total parameters: {total_params:,}")
                print(f"Trainable parameters: {trainable_params:,}")
                print(f"T-Max-Avg: {args.use_tmaxavg}, K={args.K}, T={args.T}")
                print(f"Augmentation: {args.augment}")
                print(f"Mixed Precision (AMP): {args.amp}")
                print(f"Batch Size: {args.batch_size}")
                print(f"Compile: {args.compile}")

                # Training setup
                criterion = nn.CrossEntropyLoss()
                optimizer = torch.optim.SGD(
                    model.parameters(),
                    lr=args.lr,
                    momentum=0.9,
                    weight_decay=args.weight_decay,
                    nesterov=True
                )

                # Learning rate scheduler
                if args.scheduler == "cosine":
                    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                        optimizer, T_max=args.epochs
                    )
                elif args.scheduler == "step":
                    scheduler = torch.optim.lr_scheduler.MultiStepLR(
                        optimizer,
                        milestones=[args.epochs // 2, args.epochs * 3 // 4],
                        gamma=0.1
                    )
                else:
                    scheduler = None

                # Mixed precision training setup
                scaler = torch.amp.GradScaler('cuda') if args.amp else None
                autocast_dtype = torch.bfloat16 if args.bf16 else torch.float16

                best_acc = 0.0

                for epoch in range(1, args.epochs + 1):
                    model.train()
                    running_loss = 0.0
                    correct = 0
                    total = 0

                    for imgs, labels in train_loader:
                        imgs = imgs.to(device, non_blocking=True)
                        labels = labels.to(device, non_blocking=True)

                        optimizer.zero_grad(set_to_none=True)  # More efficient than zero_grad()

                        # Mixed precision forward pass
                        if args.amp:
                            with torch.amp.autocast('cuda', dtype=autocast_dtype):
                                outputs = model(imgs)
                                loss = criterion(outputs, labels)
                            scaler.scale(loss).backward()
                            scaler.step(optimizer)
                            scaler.update()
                        else:
                            outputs = model(imgs)
                            loss = criterion(outputs, labels)
                            loss.backward()
                            optimizer.step()

                        running_loss += loss.item() * imgs.size(0)
                        _, predicted = outputs.max(1)
                        total += labels.size(0)
                        correct += predicted.eq(labels).sum().item()

                    train_loss = running_loss / len(train_loader.dataset)
                    train_acc = 100.0 * correct / total

                    # Evaluate (with AMP for consistency)
                    if args.amp:
                        test_acc, test_loss = evaluate_amp(
                            model, test_loader, criterion, device, autocast_dtype
                        )
                    else:
                        test_acc, test_loss = evaluate(model, test_loader, criterion, device)

                    # Update scheduler
                    if scheduler is not None:
                        scheduler.step()
                        current_lr = scheduler.get_last_lr()[0]
                    else:
                        current_lr = args.lr

                    print(f"Epoch [{epoch:03d}/{args.epochs:03d}] "
                          f"LR={current_lr:.6f} "
                          f"TrainLoss={train_loss:.4f} TrainAcc={train_acc:.2f}% "
                          f"TestLoss={test_loss:.4f} TestAcc={test_acc:.2f}%")

                    if test_acc > best_acc:
                        best_acc = test_acc
                        if args.save_model:
                            model_path = os.path.join(
                                args.output_dir,
                                f"{args.model}_{dataset_name}_run{run}_best.pt"
                            )
                            torch.save(model.state_dict(), model_path)

                print(f"[Run {run}] Best test accuracy: {best_acc:.2f}%")
                all_results.append({
                    "run": run,
                    "best_acc": best_acc,
                    "model": args.model,
                    "dataset": dataset_name,
                    "use_tmaxavg": args.use_tmaxavg,
                    "K": args.K,
                    "T": args.T,
                })

            # Summary
            acc_list = [r["best_acc"] for r in all_results]
            mean_acc = np.mean(acc_list)
            std_acc = np.std(acc_list)

            print("\n=== Summary ===")
            print(f"Model: {args.model}")
            print(f"Dataset: {dataset_name}")
            print(f"T-Max-Avg: {args.use_tmaxavg} (K={args.K}, T={args.T})")
            print(f"Accuracies: {acc_list}")
            print(f"Mean accuracy: {mean_acc:.2f}% ± {std_acc:.2f}%")

            # Save results
            df_results = pd.DataFrame(all_results)
            output_file = os.path.join(
                args.output_dir,
                f"{args.model}_{dataset_name}_results.csv"
            )
            df_results.to_csv(output_file, index=False)
            print(f"Results saved to {output_file}")


if __name__ == "__main__":
    main()


