#!/usr/bin/env python3
"""
HyFreqMoE-SR (HyReSR × FreqMoE-SR) — corrected & merged architecture.

Design goals
- Stable early training (no "waterpaint"/mosaic phase).
- Strong global structure (window attention at low-res + global mixer).
- Strong textures / generativity (frequency head + *adaptive* noise injection).
- Conditional specialization with *real* sparse MoE (token MoE at bottleneck).
- Practical speed: attention only at low resolutions; conv-first everywhere else.
- Deploy path: re-parameterizable large-kernel texture conv.

This file intentionally fixes the main issues observed in the two source archs:
- FreqMoE-SR: fixed blur split -> bad start + early artifacts; MoE stacked outputs -> VRAM blowup; noise injection not conditioned.
- HyReSR: ChannelLayerNorm permutes (slow, batch=1 sensitive); MoE computes all experts; window attention needs robust padding.

Compatible with traiNNer-redux ARCH_REGISTRY.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch import Tensor, nn
from torch.nn import functional as F  # noqa: N812

from traiNNer.utils.registry import ARCH_REGISTRY


# -------------------------
# Helpers
# -------------------------

def _pad_to_window(x: Tensor, window_size: int) -> tuple[Tensor, tuple[int, int]]:
    """Pad H/W to be divisible by window_size."""
    _, _, h, w = x.shape
    pad_h = (window_size - h % window_size) % window_size
    pad_w = (window_size - w % window_size) % window_size
    if pad_h == 0 and pad_w == 0:
        return x, (0, 0)
    x = F.pad(x, (0, pad_w, 0, pad_h))
    return x, (pad_h, pad_w)


def _init_depthwise_identity(conv: nn.Conv2d) -> None:
    """Initialize depthwise conv to identity mapping (center=1)."""
    if conv.groups != conv.in_channels or conv.in_channels != conv.out_channels:
        raise ValueError("Expected depthwise conv with in_ch==out_ch==groups")
    with torch.no_grad():
        conv.weight.zero_()
        k = conv.kernel_size[0]
        c = k // 2
        conv.weight[:, 0, c, c] = 1.0
        if conv.bias is not None:
            conv.bias.zero_()


# -------------------------
# Norm + core blocks
# -------------------------

class GN(nn.Module):
    """GroupNorm wrapper with sane defaults for SR (batch often small)."""

    def __init__(self, channels: int, groups: int = 1, eps: float = 1e-6) -> None:
        super().__init__()
        g = min(groups, channels)
        # Ensure divisibility; fall back to 1.
        if channels % g != 0:
            g = 1
        self.norm = nn.GroupNorm(g, channels, eps=eps)

    def forward(self, x: Tensor) -> Tensor:
        return self.norm(x)


class ConvGatedBlock(nn.Module):
    """NAF-like gated conv block (fast, stable)."""

    def __init__(self, dim: int, expansion: float = 2.0, norm_groups: int = 1) -> None:
        super().__init__()
        hidden = int(dim * expansion)
        self.norm = GN(dim, groups=norm_groups)
        self.pw1 = nn.Conv2d(dim, hidden * 2, 1)
        self.dw = nn.Conv2d(hidden * 2, hidden * 2, 3, padding=1, groups=hidden * 2)
        self.pw2 = nn.Conv2d(hidden, dim, 1)

        # Residual scale (stable start)
        self.beta = nn.Parameter(torch.zeros(1))

    def forward(self, x: Tensor) -> Tensor:
        y = self.norm(x)
        y = self.pw1(y)
        y = self.dw(y)
        u, v = y.chunk(2, dim=1)
        y = u * F.gelu(v)
        y = self.pw2(y)
        return x + self.beta * y


class GatedFFN(nn.Module):
    """Conv FFN (1x1 -> gate -> 1x1) with residual scale."""

    def __init__(self, dim: int, expansion: float = 2.0, norm_groups: int = 1) -> None:
        super().__init__()
        hidden = int(dim * expansion)
        self.norm = GN(dim, groups=norm_groups)
        self.fc1 = nn.Conv2d(dim, hidden * 2, 1)
        self.fc2 = nn.Conv2d(hidden, dim, 1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x: Tensor) -> Tensor:
        y = self.norm(x)
        y = self.fc1(y)
        u, v = y.chunk(2, dim=1)
        y = u * F.gelu(v)
        y = self.fc2(y)
        return x + self.gamma * y


class WindowAttention(nn.Module):
    """Window attention for 2D features, with optional cyclic shift and robust padding."""

    def __init__(
        self,
        dim: int,
        window_size: int = 8,
        num_heads: int = 4,
        shift_size: int = 0,
    ) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError("dim must be divisible by num_heads")
        if shift_size >= window_size:
            raise ValueError("shift_size must be < window_size")

        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        self.shift_size = shift_size
        self.head_dim = dim // num_heads

        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: Tensor) -> Tensor:
        b, c, h, w = x.shape

        x, pads = _pad_to_window(x, self.window_size)
        pad_h, pad_w = pads
        _, _, hp, wp = x.shape

        # (B,C,H,W) -> (B,H,W,C)
        x = x.permute(0, 2, 3, 1).contiguous()

        if self.shift_size > 0:
            x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))

        # Partition windows
        x = x.view(
            b,
            hp // self.window_size,
            self.window_size,
            wp // self.window_size,
            self.window_size,
            c,
        )
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        x = x.view(-1, self.window_size * self.window_size, c)  # (BnW, T, C)

        qkv = self.qkv(x)  # (BnW, T, 3C)
        q, k, v = qkv.chunk(3, dim=-1)

        # (BnW, heads, T, head_dim)
        q = q.view(q.shape[0], q.shape[1], self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(k.shape[0], k.shape[1], self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(v.shape[0], v.shape[1], self.num_heads, self.head_dim).transpose(1, 2)

        # SDPA if available (fast + stable), fallback otherwise
        if hasattr(F, "scaled_dot_product_attention"):
            out = F.scaled_dot_product_attention(q, k, v)  # (BnW, heads, T, head_dim)
        else:
            attn = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
            attn = attn.softmax(dim=-1)
            out = attn @ v

        out = out.transpose(1, 2).reshape(x.shape[0], x.shape[1], c)  # (BnW,T,C)
        out = self.proj(out)

        # Unpartition windows
        out = out.view(
            b,
            hp // self.window_size,
            wp // self.window_size,
            self.window_size,
            self.window_size,
            c,
        )
        out = out.permute(0, 1, 3, 2, 4, 5).contiguous().view(b, hp, wp, c)

        if self.shift_size > 0:
            out = torch.roll(out, shifts=(self.shift_size, self.shift_size), dims=(1, 2))

        if pad_h > 0 or pad_w > 0:
            out = out[:, :h, :w, :]

        return out.permute(0, 3, 1, 2).contiguous()


class AttnFFNBlock(nn.Module):
    """Conv block + (optional) window attention + FFN."""

    def __init__(
        self,
        dim: int,
        use_attn: bool,
        window_size: int,
        num_heads: int,
        shift: bool,
        norm_groups: int = 1,
    ) -> None:
        super().__init__()
        self.conv = ConvGatedBlock(dim, expansion=2.0, norm_groups=norm_groups)
        self.use_attn = use_attn
        if use_attn:
            self.norm_attn = GN(dim, groups=norm_groups)
            self.attn = WindowAttention(
                dim,
                window_size=window_size,
                num_heads=num_heads,
                shift_size=window_size // 2 if shift else 0,
            )
            self.delta = nn.Parameter(torch.zeros(1))
        self.ffn = GatedFFN(dim, expansion=2.0, norm_groups=norm_groups)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        if self.use_attn:
            y = self.attn(self.norm_attn(x))
            x = x + self.delta * y
        x = self.ffn(x)
        return x


# -------------------------
# Frequency split + fusion
# -------------------------

class DepthwiseBlur(nn.Module):
    """Fixed average blur (deterministic LF reference)."""

    def __init__(self, channels: int, kernel_size: int = 3) -> None:
        super().__init__()
        padding = kernel_size // 2
        weight = torch.ones(1, 1, kernel_size, kernel_size)
        weight = weight / weight.numel()
        self.register_buffer("weight", weight)
        self.channels = channels
        self.padding = padding

    def forward(self, x: Tensor) -> Tensor:
        w = self.weight.repeat(self.channels, 1, 1, 1)
        return F.conv2d(x, w, padding=self.padding, groups=self.channels)


class HybridFreqSplit(nn.Module):
    """
    Fixes FreqMoE-SR 'bad start' while keeping frequency inductive bias.

    low = mix( fixed_blur(x), learned_depthwise(x) )
    learned_depthwise is initialized to identity => low≈x at step 0 => high≈0 (stable start).
    mix coefficient alpha is learnable per-channel in [0,1].
    """

    def __init__(self, channels: int, blur_ks: int = 3, norm_groups: int = 1) -> None:
        super().__init__()
        self.blur = DepthwiseBlur(channels, blur_ks)
        self.low_dw = nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=True)
        _init_depthwise_identity(self.low_dw)
        self.low_norm = GN(channels, groups=norm_groups)
        self.lift = nn.Conv2d(channels, channels, 1, bias=True)

        # alpha per-channel (start near 0 => rely on identity low_dw early)
        self.alpha = nn.Parameter(torch.full((1, channels, 1, 1), -4.0))

        # HF bypass (lets model keep a safe identity path for details)
        self.hf_bypass = nn.Parameter(torch.zeros(1))

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        low_fixed = self.blur(x)
        low_learn = self.low_norm(self.low_dw(x))
        a = torch.sigmoid(self.alpha)
        low = a * low_fixed + (1.0 - a) * low_learn

        high = x - low
        # Safe bypass keeps early training stable; model learns to use it
        high = high + self.hf_bypass * (x - low_fixed)

        high = self.lift(high)
        return low, high


class CrossFrequencyFusion(nn.Module):
    """Light cross-gating LF<->HF (from HyReSR, kept because it's strong and stable)."""

    def __init__(self, dim: int, reduction: int = 8) -> None:
        super().__init__()
        hidden = max(dim // reduction, 4)
        self.hf_proj = nn.Conv2d(dim, dim, 1)
        self.lf_proj = nn.Conv2d(dim, dim, 1)
        self.hf_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, hidden, 1),
            nn.GELU(),
            nn.Conv2d(hidden, dim, 1),
            nn.Sigmoid(),
        )
        self.lf_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, hidden, 1),
            nn.GELU(),
            nn.Conv2d(hidden, dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, lf: Tensor, hf: Tensor) -> tuple[Tensor, Tensor]:
        lf = lf + self.hf_proj(hf) * self.hf_gate(hf)
        hf = hf + self.lf_proj(lf) * self.lf_gate(lf)
        return lf, hf


# -------------------------
# Sparse Token MoE (real dispatch, at bottleneck only)
# -------------------------

class TokenMLPExpert(nn.Module):
    def __init__(self, dim: int, hidden_mult: float = 2.0) -> None:
        super().__init__()
        hidden = int(dim * hidden_mult)
        self.fc1 = nn.Linear(dim, hidden)
        self.fc2 = nn.Linear(hidden, dim)

    def forward(self, x: Tensor) -> Tensor:
        return self.fc2(F.gelu(self.fc1(x)))


class SparseTopKMoE(nn.Module):
    """
    Real sparse MoE on tokens: only evaluates selected experts per token.
    Used ONLY at the bottleneck (small HW) => strong specialization, low overhead.

    Note: no aux loss here because traiNNer-redux arch forward is expected to return only the image.
    """

    def __init__(self, dim: int, num_experts: int = 4, top_k: int = 2, hidden_mult: float = 2.0) -> None:
        super().__init__()
        if top_k > num_experts:
            raise ValueError("top_k must be <= num_experts")
        self.num_experts = num_experts
        self.top_k = top_k
        self.router = nn.Linear(dim, num_experts, bias=True)
        self.experts = nn.ModuleList([TokenMLPExpert(dim, hidden_mult=hidden_mult) for _ in range(num_experts)])

        # Residual scale (stable)
        self.scale = nn.Parameter(torch.zeros(1))

    def forward(self, x: Tensor) -> Tensor:
        """
        x: (B, C, H, W)
        """
        b, c, h, w = x.shape
        tokens = x.permute(0, 2, 3, 1).contiguous().view(b, h * w, c)  # (B,N,C)

        logits = self.router(tokens)  # (B,N,E)
        topv, topi = logits.topk(self.top_k, dim=-1)  # (B,N,K)
        wts = F.softmax(topv, dim=-1)  # (B,N,K)

        # Dispatch per expert (batch all tokens routed to that expert)
        out = torch.zeros_like(tokens)
        for e_idx, expert in enumerate(self.experts):
            # mask of tokens where this expert is selected in any slot
            for k in range(self.top_k):
                sel = topi[..., k] == e_idx  # (B,N)
                if not sel.any():
                    continue
                # gather selected tokens
                x_e = tokens[sel]  # (T,C)
                y_e = expert(x_e)  # (T,C)
                out[sel] += y_e * wts[..., k][sel].unsqueeze(-1)

        out = out.view(b, h, w, c).permute(0, 3, 1, 2).contiguous()
        return x + self.scale * out


# -------------------------
# Texture head: reparam large-kernel + adaptive noise
# -------------------------

class ReparamLargeKernelConv(nn.Module):
    """
    Re-parameterizable depthwise large kernel conv:
    y = a0 * x + a1 * DW(1x1) + a2 * DW(3x3) + a3 * DW(kxk)
    In eval(): fused into a single DW(kxk) for speed.
    """

    def __init__(self, channels: int, kernel_size: int = 7) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.conv1x1 = nn.Conv2d(channels, channels, 1, bias=False, groups=channels)
        self.conv3x3 = nn.Conv2d(channels, channels, 3, padding=1, bias=False, groups=channels)
        self.convkxk = nn.Conv2d(channels, channels, kernel_size, padding=padding, bias=False, groups=channels)
        self.alpha = nn.Parameter(torch.ones(4))
        self.reparam_conv = nn.Conv2d(channels, channels, kernel_size, padding=padding, bias=False, groups=channels)

    def forward_train(self, x: Tensor) -> Tensor:
        out1 = self.conv1x1(x)
        out3 = self.conv3x3(x)
        outk = self.convkxk(x)
        return self.alpha[0] * x + self.alpha[1] * out1 + self.alpha[2] * out3 + self.alpha[3] * outk

    def reparam(self) -> None:
        pad = self.convkxk.kernel_size[0] // 2
        w1 = F.pad(self.conv1x1.weight, (pad, pad, pad, pad))
        w3 = F.pad(self.conv3x3.weight, (pad - 1, pad - 1, pad - 1, pad - 1))
        identity = torch.zeros_like(self.convkxk.weight)
        c = pad
        identity[:, :, c, c] = 1.0
        combined = self.alpha[0] * identity + self.alpha[1] * w1 + self.alpha[2] * w3 + self.alpha[3] * self.convkxk.weight
        self.reparam_conv.weight = nn.Parameter(combined.to(self.reparam_conv.weight.device))

    def train(self, mode: bool = True) -> "ReparamLargeKernelConv":
        super().train(mode)
        if not mode:
            self.reparam()
        return self

    def forward(self, x: Tensor) -> Tensor:
        if self.training:
            return self.forward_train(x)
        return self.reparam_conv(x)


class AdaptiveNoiseInjection(nn.Module):
    """
    'Smart' noise injection:
    - noise is random (for generativity) but *modulated* by features (for control/stability)
    - produces spatially-varying amplitude + gate
    - has warmup so early training doesn't artifact

    Injected noise is applied ONLY inside the HF/texture branch.
    """

    def __init__(
        self,
        channels: int,
        cond_channels: int,
        warmup_iters: int = 4000,
        colored_noise: bool = True,
    ) -> None:
        super().__init__()
        self.warmup_iters = max(int(warmup_iters), 0)
        self.colored_noise = colored_noise

        self.noise_proj = nn.Conv2d(1, channels, 1)
        self.amp = nn.Sequential(
            nn.Conv2d(cond_channels, channels, 1),
            nn.Softplus(),
        )
        self.gate = nn.Sequential(
            nn.Conv2d(cond_channels, channels, 1),
            nn.Sigmoid(),
        )

        # global strength (start ~0)
        self.strength = nn.Parameter(torch.tensor(-4.0))  # sigmoid ~ 0.018
        self.register_buffer("_step", torch.zeros((), dtype=torch.long), persistent=False)

        if colored_noise:
            self.blur = DepthwiseBlur(1, kernel_size=5)

        # Initialize conv biases to discourage early noise
        for m in self.modules():
            if isinstance(m, nn.Conv2d) and m.bias is not None:
                nn.init.zeros_(m.bias)

    def _warmup_scale(self) -> Tensor:
        if not self.training or self.warmup_iters == 0:
            return torch.ones((), device=self._step.device, dtype=torch.float32)
        t = (self._step.float() / float(self.warmup_iters)).clamp(0.0, 1.0)
        return t

    def forward(self, x: Tensor, cond: Tensor, noise: Optional[Tensor] = None) -> Tensor:
        if self.training:
            self._step += 1

        if noise is None:
            noise = torch.randn(x.shape[0], 1, x.shape[2], x.shape[3], device=x.device, dtype=x.dtype)
        if self.colored_noise:
            # correlate noise spatially (less 'static-like')
            noise = self.blur(noise)

        proj = self.noise_proj(noise)
        amp = self.amp(cond)
        gate = self.gate(cond)

        s = torch.sigmoid(self.strength) * self._warmup_scale().to(x.device)
        return x + (gate * amp) * proj * s


class FrequencyTextureHead(nn.Module):
    """
    Final frequency head:
      - LF refine keeps structure crisp.
      - HF texture branch is expressive + generative (adaptive noise + reparam large-kernel).
      - Cross gating prevents HF from breaking geometry.
    """

    def __init__(self, dim: int, norm_groups: int = 1, noise_warmup: int = 4000) -> None:
        super().__init__()
        self.split = HybridFreqSplit(dim, blur_ks=3, norm_groups=norm_groups)
        self.lf_ref = nn.Sequential(
            ConvGatedBlock(dim, norm_groups=norm_groups),
            ConvGatedBlock(dim, norm_groups=norm_groups),
        )
        self.hf_pre = ConvGatedBlock(dim, norm_groups=norm_groups)
        self.hf_lk = ReparamLargeKernelConv(dim, kernel_size=7)
        self.hf_post = ConvGatedBlock(dim, norm_groups=norm_groups)

        self.noise = AdaptiveNoiseInjection(dim, cond_channels=dim, warmup_iters=noise_warmup, colored_noise=True)
        self.xfuse = CrossFrequencyFusion(dim)

        self.merge = nn.Conv2d(dim * 2, dim, 1)

    def forward(self, x: Tensor) -> Tensor:
        lf, hf = self.split(x)
        lf = self.lf_ref(lf)

        # HF: controlled stochasticity conditioned on LF (global structure)
        hf = self.hf_pre(hf)
        hf = self.noise(hf, cond=lf)
        hf = self.hf_lk(hf)
        hf = self.hf_post(hf)

        lf, hf = self.xfuse(lf, hf)
        return self.merge(torch.cat([lf, hf], dim=1))

    def switch_to_deploy(self) -> None:
        # Ensure LK conv fused for inference
        self.hf_lk.train(False)


# -------------------------
# Encoder / Decoder
# -------------------------

class Downsample(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 3, stride=2, padding=1)

    def forward(self, x: Tensor) -> Tensor:
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch * 4, 3, padding=1)
        self.ps = nn.PixelShuffle(2)

    def forward(self, x: Tensor) -> Tensor:
        return self.ps(self.conv(x))


class Stage(nn.Module):
    def __init__(
        self,
        dim: int,
        depth: int,
        use_attn: bool,
        window_size: int,
        num_heads: int,
        shift: bool,
        norm_groups: int,
    ) -> None:
        super().__init__()
        self.blocks = nn.Sequential(
            *[
                AttnFFNBlock(
                    dim,
                    use_attn=use_attn,
                    window_size=window_size,
                    num_heads=num_heads,
                    shift=shift,
                    norm_groups=norm_groups,
                )
                for _ in range(depth)
            ]
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.blocks(x)


class GlobalMixer(nn.Module):
    """Cheap global context: large-kernel depthwise conv + pointwise mixing."""

    def __init__(self, dim: int, kernel_size: int = 9) -> None:
        super().__init__()
        pad = kernel_size // 2
        self.dw = nn.Conv2d(dim, dim, kernel_size, padding=pad, groups=dim)
        self.pw = nn.Conv2d(dim, dim, 1)
        self.scale = nn.Parameter(torch.zeros(1))

    def forward(self, x: Tensor) -> Tensor:
        y = self.pw(self.dw(x))
        return x + self.scale * y


class PixelShuffleHead(nn.Module):
    def __init__(self, dim: int, out_ch: int, scale: int) -> None:
        super().__init__()
        if scale not in (1, 2, 4):
            raise ValueError("scale must be 1, 2, or 4")
        layers: list[nn.Module] = []
        if scale == 1:
            layers += [nn.Conv2d(dim, out_ch, 3, padding=1)]
        elif scale == 2:
            layers += [nn.Conv2d(dim, dim * 4, 3, padding=1), nn.PixelShuffle(2), nn.Conv2d(dim, out_ch, 3, padding=1)]
        else:  # 4
            layers += [
                nn.Conv2d(dim, dim * 4, 3, padding=1),
                nn.PixelShuffle(2),
                nn.Conv2d(dim, dim * 4, 3, padding=1),
                nn.PixelShuffle(2),
                nn.Conv2d(dim, out_ch, 3, padding=1),
            ]
        self.net = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


# -------------------------
# Main network
# -------------------------

@ARCH_REGISTRY.register()
class HyFreqMoE_SR(nn.Module):
    """
    Combined corrected architecture (recommended replacement for both):
    - Conv-first multi-scale encoder/decoder
    - Window attention only at 1/2 and 1/4 scales
    - Real sparse Token-MoE at bottleneck
    - Frequency + adaptive noise texture head for generativity
    """

    def __init__(
        self,
        scale: int = 4,
        in_ch: int = 3,
        out_ch: int = 3,
        dim: int = 64,
        depths: tuple[int, int, int] = (4, 4, 6),
        window_size: int = 8,
        num_heads: tuple[int, int] = (4, 8),  # stage2, stage3
        norm_groups: int = 1,
        moe_experts: int = 4,
        moe_top_k: int = 2,
        noise_warmup: int = 4000,
    ) -> None:
        super().__init__()
        if scale not in (1, 2, 4):
            raise ValueError("scale must be 1, 2, or 4")
        self.scale = scale

        # Stem
        self.stem = nn.Conv2d(in_ch, dim, 3, padding=1)

        # Encoder
        self.enc1 = Stage(dim, depths[0], use_attn=False, window_size=window_size, num_heads=1, shift=False, norm_groups=norm_groups)
        self.down1 = Downsample(dim, dim * 2)

        self.enc2 = Stage(dim * 2, depths[1], use_attn=True, window_size=window_size, num_heads=num_heads[0], shift=True, norm_groups=norm_groups)
        self.down2 = Downsample(dim * 2, dim * 4)

        # Bottleneck: attention + global + sparse token MoE
        self.enc3 = Stage(dim * 4, max(depths[2] - 2, 1), use_attn=True, window_size=window_size, num_heads=num_heads[1], shift=True, norm_groups=norm_groups)
        self.global_mixer = GlobalMixer(dim * 4, kernel_size=9)
        self.token_moe = SparseTopKMoE(dim * 4, num_experts=moe_experts, top_k=moe_top_k, hidden_mult=2.0)
        self.bneck = nn.Sequential(
            AttnFFNBlock(dim * 4, use_attn=True, window_size=window_size, num_heads=num_heads[1], shift=False, norm_groups=norm_groups),
            AttnFFNBlock(dim * 4, use_attn=False, window_size=window_size, num_heads=1, shift=False, norm_groups=norm_groups),
        )

        # Decoder (mirror)
        self.up2 = Upsample(dim * 4, dim * 2)
        self.dec2 = Stage(dim * 2, depths[1], use_attn=True, window_size=window_size, num_heads=num_heads[0], shift=False, norm_groups=norm_groups)

        self.up1 = Upsample(dim * 2, dim)
        self.dec1 = Stage(dim, depths[0], use_attn=False, window_size=window_size, num_heads=1, shift=False, norm_groups=norm_groups)

        # Frequency + generative texture refinement
        self.texture_head = FrequencyTextureHead(dim, norm_groups=norm_groups, noise_warmup=noise_warmup)

        # Upsample / recon
        self.head = PixelShuffleHead(dim, out_ch, scale=scale)

    def switch_to_deploy(self) -> None:
        """Fuse re-parameterizable layers for inference."""
        self.texture_head.switch_to_deploy()

    def forward(self, x: Tensor) -> Tensor:
        residual = x

        x = self.stem(x)

        e1 = self.enc1(x)
        e2 = self.enc2(self.down1(e1))
        e3 = self.enc3(self.down2(e2))

        # Bottleneck enrich
        e3 = self.global_mixer(e3)
        e3 = self.token_moe(e3)
        e3 = self.bneck(e3)

        d2 = self.dec2(self.up2(e3) + e2)
        d1 = self.dec1(self.up1(d2) + e1)

        # final texture / detail refinement
        d1 = self.texture_head(d1)

        out = self.head(d1)

        # Restoration residual for scale=1
        if self.scale == 1 and out.shape == residual.shape:
            out = out + residual
        return out
