#!/usr/bin/env python3
"""
Innovative Pooling Methods for T-Max-Avg Research.

Novel extensions:
1. LearnableTMaxAvgPool2d - Per-channel learnable thresholds
2. SoftTMaxAvgPool2d - Temperature-scaled smooth blending
3. ChannelAdaptivePool2d - Learned max/avg blend per channel
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LearnableTMaxAvgPool2d(nn.Module):
    """
    T-Max-Avg pooling with learnable per-channel thresholds.
    
    Instead of a fixed T, learn optimal thresholds during training.
    Each channel can have its own threshold, allowing the network
    to adapt pooling behavior to feature importance.
    
    Args:
        kernel_size: Pooling window size
        stride: Pooling stride (default: kernel_size)
        K: Number of top values to consider
        init_T: Initial threshold value
        num_channels: Number of channels (for per-channel T)
        per_channel: If True, learn one T per channel; else one global T
    """
    def __init__(self, kernel_size=2, stride=None, K=2, init_T=0.5, 
                 num_channels=None, per_channel=True):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride or kernel_size
        self.K = K
        self.per_channel = per_channel
        
        # Learnable threshold(s) - use sigmoid to keep in [0, 1]
        if per_channel and num_channels is not None:
            # Raw values before sigmoid
            self.T_raw = nn.Parameter(torch.full((num_channels,), self._inv_sigmoid(init_T)))
        else:
            self.T_raw = nn.Parameter(torch.tensor(self._inv_sigmoid(init_T)))
    
    @staticmethod
    def _inv_sigmoid(x):
        """Inverse sigmoid for initialization."""
        x = max(min(x, 0.999), 0.001)  # Clamp to avoid inf
        return torch.log(torch.tensor(x / (1 - x))).item()
    
    @property
    def T(self):
        """Get threshold(s) in [0, 1] range."""
        return torch.sigmoid(self.T_raw)
    
    def forward(self, x):
        N, C, H, W = x.shape
        
        # Unfold to get patches
        patches = F.unfold(x, kernel_size=self.kernel_size, stride=self.stride)
        k2 = self.kernel_size * self.kernel_size
        patches = patches.view(N, C, k2, -1)
        patches = patches.permute(0, 1, 3, 2)  # (N, C, num_patches, k2)
        
        # Get top-K values
        K_eff = min(self.K, patches.size(-1))
        topk, _ = torch.topk(patches, K_eff, dim=-1)
        
        max_vals, _ = topk.max(dim=-1, keepdim=True)  # (N, C, num_patches, 1)
        avg_vals = topk.mean(dim=-1, keepdim=True)
        
        # Get T with proper shape for broadcasting
        T = self.T
        if self.per_channel and T.dim() == 1:
            T = T.view(1, -1, 1, 1)  # (1, C, 1, 1)
        
        # Threshold comparison
        cond = (max_vals >= T)
        out_vals = torch.where(cond, max_vals, avg_vals)
        out_vals = out_vals.squeeze(-1)
        
        # Reshape output
        H_out = (H - self.kernel_size) // self.stride + 1
        W_out = (W - self.kernel_size) // self.stride + 1
        out = out_vals.view(N, C, H_out, W_out)
        return out
    
    def extra_repr(self):
        T_val = self.T.mean().item()
        return f'kernel_size={self.kernel_size}, stride={self.stride}, K={self.K}, T_mean={T_val:.3f}'


class SoftTMaxAvgPool2d(nn.Module):
    """
    Soft T-Max-Avg pooling with temperature-scaled blending.
    
    Instead of hard if/else switching, use a smooth sigmoid blend:
        blend = sigmoid((max_val - T) / temperature)
        output = blend * max_val + (1 - blend) * avg_val
    
    This is fully differentiable and provides better gradient flow.
    Lower temperature = sharper transition (approaches hard switching).
    
    Args:
        kernel_size: Pooling window size
        stride: Pooling stride (default: kernel_size)
        K: Number of top values to consider
        T: Threshold value
        temperature: Controls blending sharpness (lower = sharper)
        learnable_T: If True, make T learnable
        learnable_temp: If True, make temperature learnable
        num_channels: Number of channels (for per-channel params)
    """
    def __init__(self, kernel_size=2, stride=None, K=2, T=0.5, temperature=0.1,
                 learnable_T=False, learnable_temp=False, num_channels=None):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride or kernel_size
        self.K = K
        self.num_channels = num_channels
        
        # T parameter
        if learnable_T:
            init_T_raw = self._inv_sigmoid(T)
            if num_channels:
                self.T_raw = nn.Parameter(torch.full((num_channels,), init_T_raw))
            else:
                self.T_raw = nn.Parameter(torch.tensor(init_T_raw))
            self._learnable_T = True
        else:
            self.register_buffer('_T', torch.tensor(T))
            self._learnable_T = False
        
        # Temperature parameter (use softplus to keep positive)
        if learnable_temp:
            # Initialize such that softplus gives desired temperature
            init_temp_raw = self._inv_softplus(temperature)
            self.temp_raw = nn.Parameter(torch.tensor(init_temp_raw))
            self._learnable_temp = True
        else:
            self.register_buffer('_temp', torch.tensor(temperature))
            self._learnable_temp = False
    
    @staticmethod
    def _inv_sigmoid(x):
        x = max(min(x, 0.999), 0.001)
        return torch.log(torch.tensor(x / (1 - x))).item()
    
    @staticmethod
    def _inv_softplus(x):
        """Inverse softplus for initialization."""
        x = max(x, 0.001)
        return torch.log(torch.exp(torch.tensor(x)) - 1).item()
    
    @property
    def T(self):
        if self._learnable_T:
            return torch.sigmoid(self.T_raw)
        return self._T
    
    @property
    def temperature(self):
        if self._learnable_temp:
            return F.softplus(self.temp_raw) + 0.01  # Min temperature
        return self._temp
    
    def forward(self, x):
        N, C, H, W = x.shape
        
        # Unfold to get patches
        patches = F.unfold(x, kernel_size=self.kernel_size, stride=self.stride)
        k2 = self.kernel_size * self.kernel_size
        patches = patches.view(N, C, k2, -1)
        patches = patches.permute(0, 1, 3, 2)
        
        # Get top-K values
        K_eff = min(self.K, patches.size(-1))
        topk, _ = torch.topk(patches, K_eff, dim=-1)
        
        max_vals = topk.max(dim=-1, keepdim=True)[0]
        avg_vals = topk.mean(dim=-1, keepdim=True)
        
        # Get T with proper broadcasting shape
        T = self.T
        if self._learnable_T and self.num_channels and T.dim() == 1:
            T = T.view(1, -1, 1, 1)
        
        # Soft blending with temperature
        temp = self.temperature
        blend = torch.sigmoid((max_vals - T) / temp)
        
        # Blend max and avg
        out_vals = blend * max_vals + (1 - blend) * avg_vals
        out_vals = out_vals.squeeze(-1)
        
        # Reshape output
        H_out = (H - self.kernel_size) // self.stride + 1
        W_out = (W - self.kernel_size) // self.stride + 1
        out = out_vals.view(N, C, H_out, W_out)
        return out
    
    def extra_repr(self):
        T_val = self.T.mean().item() if self.T.numel() > 1 else self.T.item()
        temp_val = self.temperature.item()
        return (f'kernel_size={self.kernel_size}, stride={self.stride}, K={self.K}, '
                f'T={T_val:.3f}, temp={temp_val:.3f}')


class ChannelAdaptivePool2d(nn.Module):
    """
    Channel-wise adaptive pooling with learned max/avg blend.
    
    Each channel learns its own blend weight α:
        output = α * max_pooled + (1 - α) * avg_pooled
    
    Uses sigmoid(weight) to keep α in [0, 1].
    α=1 means pure max pooling, α=0 means pure average pooling.
    
    Args:
        kernel_size: Pooling window size
        stride: Pooling stride (default: kernel_size)
        num_channels: Number of input channels
        init_alpha: Initial blend value (0.5 = equal mix)
    """
    def __init__(self, kernel_size=2, stride=None, num_channels=64, init_alpha=0.5):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride or kernel_size
        self.num_channels = num_channels
        
        # Raw weights (before sigmoid)
        init_raw = self._inv_sigmoid(init_alpha)
        self.alpha_raw = nn.Parameter(torch.full((num_channels,), init_raw))
    
    @staticmethod
    def _inv_sigmoid(x):
        x = max(min(x, 0.999), 0.001)
        return torch.log(torch.tensor(x / (1 - x))).item()
    
    @property
    def alpha(self):
        """Get blend weights in [0, 1] range."""
        return torch.sigmoid(self.alpha_raw)
    
    def forward(self, x):
        N, C, H, W = x.shape
        
        # Max pooling
        max_pooled = F.max_pool2d(x, kernel_size=self.kernel_size, stride=self.stride)
        
        # Average pooling
        avg_pooled = F.avg_pool2d(x, kernel_size=self.kernel_size, stride=self.stride)
        
        # Get alpha with proper shape for broadcasting (1, C, 1, 1)
        alpha = self.alpha.view(1, -1, 1, 1)
        
        # Blend
        out = alpha * max_pooled + (1 - alpha) * avg_pooled
        return out
    
    def extra_repr(self):
        alpha_mean = self.alpha.mean().item()
        alpha_std = self.alpha.std().item()
        return (f'kernel_size={self.kernel_size}, stride={self.stride}, '
                f'channels={self.num_channels}, alpha_mean={alpha_mean:.3f}±{alpha_std:.3f}')


class GatedTMaxAvgPool2d(nn.Module):
    """
    Gated T-Max-Avg pooling with learned gating per spatial location.
    
    Uses a small conv to predict per-location gate values:
        gate = sigmoid(conv(x))
        output = gate * max_pooled + (1 - gate) * avg_pooled
    
    This is the most expressive variant - learns spatially-varying
    pooling strategy based on local feature content.
    
    Args:
        kernel_size: Pooling window size
        stride: Pooling stride (default: kernel_size)
        num_channels: Number of input channels
    """
    def __init__(self, kernel_size=2, stride=None, num_channels=64):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride or kernel_size
        self.num_channels = num_channels
        
        # Gating network: 1x1 conv to predict gate values
        self.gate_conv = nn.Sequential(
            nn.Conv2d(num_channels, num_channels // 4, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_channels // 4, num_channels, kernel_size=1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # Compute gate values at input resolution
        gate = self.gate_conv(x)
        
        # Pool both gate and features
        gate_pooled = F.avg_pool2d(gate, kernel_size=self.kernel_size, stride=self.stride)
        max_pooled = F.max_pool2d(x, kernel_size=self.kernel_size, stride=self.stride)
        avg_pooled = F.avg_pool2d(x, kernel_size=self.kernel_size, stride=self.stride)
        
        # Blend using learned gates
        out = gate_pooled * max_pooled + (1 - gate_pooled) * avg_pooled
        return out
    
    def extra_repr(self):
        return f'kernel_size={self.kernel_size}, stride={self.stride}, channels={self.num_channels}'


# =============================================================================
# NEW INNOVATION: Attention-Weighted Pooling
# =============================================================================

class AttentionWeightedPool2d(nn.Module):
    """
    Attention-Weighted Pooling - Uses self-attention to weight spatial locations.
    
    Instead of fixed max/avg operations, learns which spatial positions 
    matter most via a lightweight attention mechanism.
    
    Key insight: Not all pixels in a pooling window are equally important.
    Attention learns importance weights adaptively per-input.
    """
    def __init__(self, kernel_size=2, stride=None, num_channels=64, reduction=4):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride or kernel_size
        self.num_channels = num_channels
        
        # Lightweight attention: squeeze spatial info, generate attention
        hidden = max(num_channels // reduction, 8)
        self.attention = nn.Sequential(
            nn.Conv2d(num_channels, hidden, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, num_channels, 1, bias=False),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        N, C, H, W = x.shape
        
        # Generate attention weights (per-pixel importance)
        attn = self.attention(x)  # (N, C, H, W)
        
        # Apply attention
        weighted = x * attn
        
        # Pool the weighted features
        out = F.avg_pool2d(weighted, kernel_size=self.kernel_size, stride=self.stride)
        
        # Also pool attention for normalization
        attn_pooled = F.avg_pool2d(attn, kernel_size=self.kernel_size, stride=self.stride)
        
        # Normalize by attention sum (like weighted average)
        out = out / (attn_pooled + 1e-6)
        
        return out
    
    def extra_repr(self):
        return f'kernel_size={self.kernel_size}, stride={self.stride}, channels={self.num_channels}'


class StochasticMixPool2d(nn.Module):
    """
    Stochastic Mix Pooling - Randomly blends max and avg during training.
    
    During training: randomly samples blend ratio from Beta distribution
    During inference: uses learned optimal blend ratio
    
    This adds regularization and helps find robust pooling strategies.
    """
    def __init__(self, kernel_size=2, stride=None, alpha=2.0, beta=2.0):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride or kernel_size
        self.alpha = alpha  # Beta distribution parameter
        self.beta = beta
        # Learned blend ratio for inference
        self.blend = nn.Parameter(torch.tensor(0.5))
        
    def forward(self, x):
        max_pool = F.max_pool2d(x, kernel_size=self.kernel_size, stride=self.stride)
        avg_pool = F.avg_pool2d(x, kernel_size=self.kernel_size, stride=self.stride)
        
        if self.training:
            # Sample random blend ratio from Beta distribution
            blend_ratio = torch.distributions.Beta(self.alpha, self.beta).sample()
            blend_ratio = blend_ratio.to(x.device)
        else:
            # Use learned blend ratio at inference
            blend_ratio = torch.sigmoid(self.blend)
        
        return blend_ratio * max_pool + (1 - blend_ratio) * avg_pool
    
    def extra_repr(self):
        return f'kernel_size={self.kernel_size}, stride={self.stride}, blend={torch.sigmoid(self.blend).item():.3f}'


# =============================================================================
# Factory Functions
# =============================================================================

def create_attention_weighted(kernel_size=2, stride=None, num_channels=64, reduction=4):
    """Factory for AttentionWeightedPool2d."""
    return AttentionWeightedPool2d(
        kernel_size=kernel_size, stride=stride,
        num_channels=num_channels, reduction=reduction
    )


def create_stochastic_mix(kernel_size=2, stride=None, alpha=2.0, beta=2.0):
    """Factory for StochasticMixPool2d."""
    return StochasticMixPool2d(
        kernel_size=kernel_size, stride=stride, alpha=alpha, beta=beta
    )


def create_learnable_tmaxavg(kernel_size=2, stride=None, K=2, init_T=0.5, num_channels=None):
    """Factory for LearnableTMaxAvgPool2d."""
    return LearnableTMaxAvgPool2d(
        kernel_size=kernel_size, stride=stride, K=K, 
        init_T=init_T, num_channels=num_channels
    )


def create_soft_tmaxavg(kernel_size=2, stride=None, K=2, T=0.5, temperature=0.1,
                        learnable_T=False, learnable_temp=False, num_channels=None):
    """Factory for SoftTMaxAvgPool2d."""
    return SoftTMaxAvgPool2d(
        kernel_size=kernel_size, stride=stride, K=K, T=T,
        temperature=temperature, learnable_T=learnable_T,
        learnable_temp=learnable_temp, num_channels=num_channels
    )


def create_channel_adaptive(kernel_size=2, stride=None, num_channels=64, init_alpha=0.5):
    """Factory for ChannelAdaptivePool2d."""
    return ChannelAdaptivePool2d(
        kernel_size=kernel_size, stride=stride,
        num_channels=num_channels, init_alpha=init_alpha
    )


def create_gated_tmaxavg(kernel_size=2, stride=None, num_channels=64):
    """Factory for GatedTMaxAvgPool2d."""
    return GatedTMaxAvgPool2d(
        kernel_size=kernel_size, stride=stride, num_channels=num_channels
    )


# =============================================================================
# Test Functions
# =============================================================================

def test_pooling_layers():
    """Test all pooling layer implementations."""
    print("Testing innovative pooling layers...")
    
    # Create test input
    batch_size, channels, height, width = 2, 16, 32, 32
    x = torch.randn(batch_size, channels, height, width)
    
    # Expected output size for kernel=2, stride=2
    expected_h = height // 2
    expected_w = width // 2
    
    # Test LearnableTMaxAvgPool2d
    print("\n1. LearnableTMaxAvgPool2d")
    pool1 = LearnableTMaxAvgPool2d(kernel_size=2, K=2, num_channels=channels)
    out1 = pool1(x)
    assert out1.shape == (batch_size, channels, expected_h, expected_w), \
        f"Shape mismatch: {out1.shape}"
    print(f"   Input: {x.shape} -> Output: {out1.shape}")
    print(f"   Learned T mean: {pool1.T.mean().item():.4f}")
    
    # Test SoftTMaxAvgPool2d
    print("\n2. SoftTMaxAvgPool2d")
    pool2 = SoftTMaxAvgPool2d(kernel_size=2, K=2, T=0.5, temperature=0.1)
    out2 = pool2(x)
    assert out2.shape == (batch_size, channels, expected_h, expected_w), \
        f"Shape mismatch: {out2.shape}"
    print(f"   Input: {x.shape} -> Output: {out2.shape}")
    
    # Test with learnable T
    pool2b = SoftTMaxAvgPool2d(kernel_size=2, K=2, learnable_T=True, num_channels=channels)
    out2b = pool2b(x)
    print(f"   Learnable T version: {out2b.shape}")
    
    # Test ChannelAdaptivePool2d
    print("\n3. ChannelAdaptivePool2d")
    pool3 = ChannelAdaptivePool2d(kernel_size=2, num_channels=channels)
    out3 = pool3(x)
    assert out3.shape == (batch_size, channels, expected_h, expected_w), \
        f"Shape mismatch: {out3.shape}"
    print(f"   Input: {x.shape} -> Output: {out3.shape}")
    print(f"   Alpha mean: {pool3.alpha.mean().item():.4f}")
    
    # Test GatedTMaxAvgPool2d
    print("\n4. GatedTMaxAvgPool2d")
    pool4 = GatedTMaxAvgPool2d(kernel_size=2, num_channels=channels)
    out4 = pool4(x)
    assert out4.shape == (batch_size, channels, expected_h, expected_w), \
        f"Shape mismatch: {out4.shape}"
    print(f"   Input: {x.shape} -> Output: {out4.shape}")
    
    # Test gradient flow
    print("\n5. Testing gradient flow...")
    x_grad = torch.randn(batch_size, channels, height, width, requires_grad=True)
    
    for name, pool in [("Learnable", pool1), ("Soft", pool2), 
                       ("ChannelAdaptive", pool3), ("Gated", pool4)]:
        out = pool(x_grad)
        loss = out.sum()
        loss.backward(retain_graph=True)
        assert x_grad.grad is not None, f"{name}: No gradient!"
        print(f"   {name}: Gradient OK (mean abs: {x_grad.grad.abs().mean():.6f})")
        x_grad.grad.zero_()
    
    print("\n✓ All tests passed!")
    return True


if __name__ == "__main__":
    test_pooling_layers()
