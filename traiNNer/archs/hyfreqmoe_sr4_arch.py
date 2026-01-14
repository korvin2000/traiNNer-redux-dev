#!/usr/bin/env python3
"""
HyFreqMoE-SR — v3 architecture incorporating gemini.md best-practice fixes.

This revision incorporates a set of expert-review driven changes:
- PixelShuffle head follows canonical SR upsampler pattern (supports 2^n and 3).
- Linear op chains use nn.Sequential / nn.Sequential-style modules where appropriate.
- Consistent module style (nn.GELU instead of F.gelu in module graphs).
- Normalization defaults to RMSNorm2d (norm_groups=0). Set norm_groups<0 to disable, or norm_groups>0 for GroupNorm.
- Window attention uses Swin-style alternating shift inside a Stage and applies an attention mask for shifted windows.
- Frequency split uses a fixed *Gaussian* (binomial) low-pass and a learnable HF scale for a stable start.
- Fixed blur kernels avoid per-forward weight replication; implemented as separable depthwise Gaussian for speed.
- MoE includes anti-collapse hooks: router noise + auxiliary stats (importance/load/z-loss) exposed for training code.
- Re-parameterizable large-kernel conv is fused explicitly via switch_to_deploy() (not tied to eval()).

Notes
- The forward() of the main model still returns only the restored SR image.
- MoE auxiliary losses are *exposed* (self.get_aux_loss()) but not automatically added to the image output.
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch
from torch import Tensor, nn
from torch.nn import functional as F  # noqa: N812

from traiNNer.utils.registry import ARCH_REGISTRY


# -------------------------
# Helpers
# -------------------------

def _pad_to_window(x: Tensor, window_size: int) -> Tuple[Tensor, Tuple[int, int]]:
    """Pad H/W to be multiples of window_size."""
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
    k = conv.kernel_size[0]
    if k != conv.kernel_size[1]:
        raise ValueError("Expected square kernel")
    with torch.no_grad():
        conv.weight.zero_()
        conv.weight[:, 0, k // 2, k // 2] = 1.0
        if conv.bias is not None:
            conv.bias.zero_()


def _is_power_of_two(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0



class LayerNorm2d(nn.Module):
    """Channel-first LayerNorm for [B,C,H,W]."""

    def __init__(self, channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        return self.weight[:, None, None] * x + self.bias[:, None, None]


class RMSNorm2d(nn.Module):
    """Spatial RMSNorm for SR: stable with batch=1 and FP16 (variance in FP32)."""

    def __init__(self, channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.ones(channels, 1, 1))
        self.bias = nn.Parameter(torch.zeros(channels, 1, 1))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        x_f32 = x.float()
        variance = x_f32.pow(2).mean(dim=1, keepdim=True)
        rms = torch.sqrt(variance + self.eps).to(x.dtype)
        return self.scale * x / rms + self.bias


def make_norm(channels: int, norm_groups: int, eps: float = 1e-6) -> nn.Module:
    """Normalization factory.

    Semantics (kept single-int for config compatibility):
    - norm_groups > 0  -> GroupNorm(norm_groups, channels) with strict divisibility.
    - norm_groups == 0 -> RMSNorm2d(channels) (recommended default for SR stability).
    - norm_groups < 0  -> Identity (explicitly disable normalization).

    Rationale: SR is commonly trained with small batch sizes; RMSNorm/LayerNorm
    are generally more stable than BatchNorm/GroupNorm in that regime.
    """
    if norm_groups < 0:
        return nn.Identity()
    if norm_groups == 0:
        return RMSNorm2d(channels, eps=eps)
    if channels % norm_groups != 0:
        raise ValueError(f"channels ({channels}) must be divisible by norm_groups ({norm_groups})")
    return nn.GroupNorm(norm_groups, channels, eps=eps, affine=True)



# -------------------------
# Core blocks (conv-first)
# -------------------------

class ConvGatedBlock(nn.Module):
    """NAF-like gated conv block (fast, stable) with optional GN."""

    def __init__(self, dim: int, expansion: float = 2.0, norm_groups: int = 0) -> None:
        super().__init__()
        hidden = int(dim * expansion)
        self.norm = make_norm(dim, norm_groups)
        self.pw1 = nn.Conv2d(dim, hidden * 2, 1)
        self.dw = nn.Conv2d(hidden * 2, hidden * 2, 3, padding=1, groups=hidden * 2)
        self.act = nn.GELU()
        self.pw2 = nn.Conv2d(hidden, dim, 1)
        self.beta = nn.Parameter(torch.zeros(1))

    def forward(self, x: Tensor) -> Tensor:
        y = self.norm(x)
        y = self.pw1(y)
        y = self.dw(y)
        u, v = y.chunk(2, dim=1)
        y = u * self.act(v)
        y = self.pw2(y)
        return x + self.beta * y


class GatedFFN(nn.Module):
    """Conv FFN (1x1 -> gate -> 1x1) with residual scale."""

    def __init__(self, dim: int, expansion: float = 2.0, norm_groups: int = 0) -> None:
        super().__init__()
        hidden = int(dim * expansion)
        self.norm = make_norm(dim, norm_groups)
        self.fc1 = nn.Conv2d(dim, hidden * 2, 1)
        self.act = nn.GELU()
        self.fc2 = nn.Conv2d(hidden, dim, 1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x: Tensor) -> Tensor:
        y = self.norm(x)
        y = self.fc1(y)
        u, v = y.chunk(2, dim=1)
        y = u * self.act(v)
        y = self.fc2(y)
        return x + self.gamma * y


# -------------------------
# Window attention (Swin-style mask when shifted)
# -------------------------

class WindowAttention(nn.Module):
    """Window attention for 2D features with optional cyclic shift and correct mask."""

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
        self.head_dim = dim // num_heads
        self.shift_size = shift_size

        self.scale = self.head_dim ** -0.5

        # Relative position bias (Swin-style) for window attention
        ws = self.window_size
        self.rpe_table = nn.Parameter(torch.zeros((2 * ws - 1) * (2 * ws - 1), self.num_heads))
        nn.init.trunc_normal_(self.rpe_table, std=0.02)

        coords_h = torch.arange(ws)
        coords_w = torch.arange(ws)
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing="ij"))  # (2, ws, ws)
        coords_flatten = torch.flatten(coords, 1)  # (2, ws*ws)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # (2, ws*ws, ws*ws)
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # (ws*ws, ws*ws, 2)
        relative_coords[:, :, 0] += ws - 1
        relative_coords[:, :, 1] += ws - 1
        relative_coords[:, :, 0] *= 2 * ws - 1
        self.register_buffer("relative_position_index", relative_coords.sum(-1), persistent=False)

        self.qkv = nn.Conv2d(dim, dim * 3, 1, bias=True)
        self.proj = nn.Linear(dim, dim)
        self._mask_cache: Dict[Tuple[int, int, str, int], Tensor] = {}

    def _window_partition(self, x: Tensor) -> Tensor:
        """
        x: (B, Hp, Wp, C) -> (B*nW, ws*ws, C)
        """
        b, hp, wp, c = x.shape
        ws = self.window_size
        x = x.view(b, hp // ws, ws, wp // ws, ws, c)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, ws * ws, c)
        return x

    def _window_unpartition(self, windows: Tensor, b: int, hp: int, wp: int) -> Tensor:
        """
        windows: (B*nW, ws*ws, C) -> (B, Hp, Wp, C)
        """
        ws = self.window_size
        c = windows.shape[-1]
        x = windows.view(b, hp // ws, wp // ws, ws, ws, c)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(b, hp, wp, c)
        return x

    def _get_attn_mask(self, hp: int, wp: int, device: torch.device) -> Tensor:
        """
        Return mask of shape (B*nW, T, T) for shifted window attention.
        Built for a single (hp,wp) and then broadcast to batch in forward.
        """
        key = (hp, wp, device.type, -1 if device.index is None else int(device.index))
        if key in self._mask_cache:
            return self._mask_cache[key]
        # Keep cache bounded (varied validation sizes can otherwise thrash/grow)
        if len(self._mask_cache) > 8:
            self._mask_cache.clear()

        ws = self.window_size
        ss = self.shift_size
        img_mask = torch.zeros((1, hp, wp, 1), device=device)

        # Swin-style region partitioning
        h_slices = (slice(0, -ws), slice(-ws, -ss), slice(-ss, None))
        w_slices = (slice(0, -ws), slice(-ws, -ss), slice(-ss, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = self._window_partition(img_mask)  # (nW, T, 1)
        mask_windows = mask_windows.view(-1, ws * ws)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)  # (nW, T, T)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float("-inf")).masked_fill(attn_mask == 0, 0.0)
        self._mask_cache[key] = attn_mask
        return attn_mask

    def forward(self, x: Tensor) -> Tensor:
        """
        x: (B, C, H, W) -> (B, C, H, W)
        """
        b, c, h, w = x.shape
        x, (pad_h, pad_w) = _pad_to_window(x, self.window_size)
        _, _, hp, wp = x.shape

        # Cyclic shift
        if self.shift_size > 0:
            x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(2, 3))

        # QKV
        qkv = self.qkv(x)  # (B, 3C, Hp, Wp)
        qkv = qkv.permute(0, 2, 3, 1).contiguous()  # (B, Hp, Wp, 3C)
        qkv = self._window_partition(qkv)  # (B*nW, T, 3C)

        ws2 = self.window_size * self.window_size
        qkv = qkv.view(-1, ws2, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # (BnW, heads, T, head_dim)

        attn_mask: Optional[Tensor] = None
        if self.shift_size > 0:
            base_mask = self._get_attn_mask(hp, wp, x.device)  # (nW, T, T)
            nW = base_mask.shape[0]
            attn_mask = base_mask.unsqueeze(0).repeat(b, 1, 1, 1).view(b * nW, ws2, ws2)
            attn_mask = attn_mask.unsqueeze(1)  # (BnW, 1, T, T)

        use_sdpa = hasattr(F, "scaled_dot_product_attention") and (q.device.type == "cuda")
        if use_sdpa:
            rpe = self.rpe_table[self.relative_position_index.view(-1)].view(
                self.window_size * self.window_size, self.window_size * self.window_size, -1
            ).permute(2, 0, 1).contiguous()  # (heads, T, T)
            attn_bias = rpe.unsqueeze(0)  # broadcast to (BnW, heads, T, T)
            attn_mask_ = attn_bias if attn_mask is None else (attn_mask + attn_bias)
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask_)  # (BnW, heads, T, head_dim)
        else:
            attn = (q @ k.transpose(-2, -1)) * self.scale
            rpe = self.rpe_table[self.relative_position_index.view(-1)].view(
                self.window_size * self.window_size, self.window_size * self.window_size, -1
            ).permute(2, 0, 1).contiguous()  # (heads, T, T)
            attn = attn + rpe.unsqueeze(0)
            if attn_mask is not None:
                attn = attn + attn_mask
            attn = attn.softmax(dim=-1)
            out = attn @ v

        out = out.transpose(1, 2).reshape(out.shape[0], ws2, self.dim)  # (BnW, T, C)
        out = self.proj(out)

        # Unpartition
        out = self._window_unpartition(out, b=b, hp=hp, wp=wp)

        # Reverse shift
        if self.shift_size > 0:
            out = torch.roll(out, shifts=(self.shift_size, self.shift_size), dims=(1, 2))

        # Crop
        if pad_h > 0 or pad_w > 0:
            out = out[:, :h, :w, :]

        return out.permute(0, 3, 1, 2).contiguous()


class AttnFFNBlock(nn.Module):
    """ConvGated + (optional) WindowAttention + FFN, each with residual scale."""

    def __init__(
        self,
        dim: int,
        use_attn: bool,
        window_size: int,
        num_heads: int,
        shift: bool,
        norm_groups: int,
        ffn_expansion: float = 2.0,
    ) -> None:
        super().__init__()
        self.conv = ConvGatedBlock(dim, norm_groups=norm_groups)
        self.use_attn = use_attn
        if use_attn:
            shift_size = window_size // 2 if shift else 0
            self.attn = WindowAttention(dim, window_size=window_size, num_heads=num_heads, shift_size=shift_size)
            self.delta = nn.Parameter(torch.zeros(1))
        else:
            self.attn = None
            self.delta = None
        self.ffn = GatedFFN(dim, expansion=ffn_expansion, norm_groups=norm_groups)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        if self.use_attn and self.attn is not None and self.delta is not None:
            x = x + self.delta * self.attn(x)
        x = self.ffn(x)
        return x


# -------------------------
# Fixed separable depthwise Gaussian (binomial) blur
# -------------------------

def _binomial_1d(kernel_size: int, device: torch.device, dtype: torch.dtype) -> Tensor:
    if kernel_size <= 0 or kernel_size % 2 == 0:
        raise ValueError("kernel_size must be odd and > 0")
    coeffs = [math.comb(kernel_size - 1, i) for i in range(kernel_size)]
    g = torch.tensor(coeffs, device=device, dtype=dtype)
    g = g / g.sum()
    return g


class DepthwiseGaussianBlur(nn.Module):
    """Fixed separable depthwise Gaussian blur (binomial).

    v3-fix: buffers are created with *final* shapes in __init__ (no lazy empty buffers).
    This avoids EMA / deepcopy buffer-shape mismatches (e.g., when EMA initializes before first forward).
    """

    def __init__(self, channels: int, kernel_size: int = 5) -> None:
        super().__init__()
        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd and > 0")

        self.channels = int(channels)
        self.kernel_size = int(kernel_size)
        self.pad = self.kernel_size // 2

        # Binomial coefficients approximate a Gaussian (sigma ~= 1 for k=5).
        coeffs = [math.comb(self.kernel_size - 1, i) for i in range(self.kernel_size)]
        g = torch.tensor(coeffs, dtype=torch.float32)
        g = g / g.sum()

        # Pre-expand once; buffers will be moved/cast with the module (DDP/EMA friendly).
        w_h = g.view(1, 1, self.kernel_size, 1).repeat(self.channels, 1, 1, 1)  # (C,1,k,1)
        w_w = g.view(1, 1, 1, self.kernel_size).repeat(self.channels, 1, 1, 1)  # (C,1,1,k)

        # Not learnable; don\'t save in state_dict (EMA/DDP friendly).
        self.register_buffer("w_h", w_h, persistent=False)
        self.register_buffer("w_w", w_w, persistent=False)
        # Cache casted weights to avoid per-forward allocations under AMP.
        self._cast_cache = {}  # key: (device, dtype) -> (w_h, w_w)

    def forward(self, x: Tensor) -> Tensor:
        key = (x.device, x.dtype)
        cached = self._cast_cache.get(key, None)
        if cached is None:
            w_h = self.w_h.to(device=x.device, dtype=x.dtype)
            w_w = self.w_w.to(device=x.device, dtype=x.dtype)
            self._cast_cache[key] = (w_h, w_w)
        else:
            w_h, w_w = cached

        x = F.conv2d(x, w_h, padding=(self.pad, 0), groups=self.channels)
        x = F.conv2d(x, w_w, padding=(0, self.pad), groups=self.channels)
        return x



# -------------------------
# Frequency split + fusion
# -------------------------

class HybridFreqSplit(nn.Module):
    """
    Interpretable frequency split with stable start.

    low  = GaussianBlur(x)
    high = hf_scale * (x - low)
    """

    def __init__(self, channels: int, blur_ks: int = 5) -> None:
        super().__init__()
        self.blur = DepthwiseGaussianBlur(channels, kernel_size=blur_ks)
        # Learnable HF scale (stable start at 0)
        self.hf_scale = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.lift = nn.Conv2d(channels, channels, 1, bias=True)
        # Prevent bias from injecting constant HF when hf_scale==0.
        nn.init.zeros_(self.lift.bias)
        # Init lift as identity (safe, predictable).
        nn.init.zeros_(self.lift.weight)
        with torch.no_grad():
            for i in range(channels):
                self.lift.weight[i, i, 0, 0] = 1.0

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        low = self.blur(x)
        high = self.lift(x - low) * self.hf_scale
        return low, high





class BottleneckFreqSplit(nn.Module):
    """Gaussian LF/HF split for bottleneck features (for MoE isolation).

    low  = GaussianBlur(x)
    high = x - low   (no additional scaling; preserves gradients and avoids disabling MoE)
    """

    def __init__(self, channels: int, blur_ks: int = 5) -> None:
        super().__init__()
        self.blur = DepthwiseGaussianBlur(channels, kernel_size=blur_ks)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        low = self.blur(x)
        high = x - low
        return low, high

class CrossFrequencyFusion(nn.Module):
    """Light cross-gating LF<->HF."""

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

    def forward(self, lf: Tensor, hf: Tensor) -> Tuple[Tensor, Tensor]:
        lf = lf + self.hf_proj(hf) * self.hf_gate(hf)
        hf = hf + self.lf_proj(lf) * self.lf_gate(lf)
        return lf, hf


# -------------------------
# Sparse Token MoE (anti-collapse hooks)
# -------------------------

class TokenMLPExpert(nn.Module):
    def __init__(self, dim: int, hidden_mult: float = 2.0) -> None:
        super().__init__()
        hidden = int(dim * hidden_mult)
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, dim)

    def forward(self, x: Tensor) -> Tensor:
        return self.fc2(self.act(self.fc1(x)))


class SparseTopKMoE(nn.Module):
    """
    Sparse MoE on tokens (evaluates only selected experts).

    Anti-collapse hooks:
    - optional router noise during training
    - auxiliary statistics (importance/load/z-loss) exposed via last_aux_loss
    """

    def __init__(
        self,
        dim: int,
        num_experts: int = 4,
        top_k: int = 2,
        hidden_mult: float = 2.0,
        router_noise_std: float = 0.0,
        aux_loss_weight: float = 0.0,
    ) -> None:
        super().__init__()
        if top_k > num_experts:
            raise ValueError("top_k must be <= num_experts")
        self.num_experts = num_experts
        self.top_k = top_k
        self.router_noise_std = float(router_noise_std)
        self.aux_loss_weight = float(aux_loss_weight)

        self.router = nn.Linear(dim, num_experts, bias=True)
        self.experts = nn.ModuleList([TokenMLPExpert(dim, hidden_mult=hidden_mult) for _ in range(num_experts)])

        # Residual scale (stable start)
        self.scale = nn.Parameter(torch.zeros(1))

        # Exposed stats for training loops
        # last_aux_loss keeps graph when aux_loss_weight>0 and training; consume via model.get_aux_loss(clear=True)
        self.last_aux_loss: Optional[Tensor] = None
        self.last_aux_loss_detached: Optional[Tensor] = None  # for logging

    def _compute_aux(self, logits: Tensor, topi: Tensor) -> Tensor:
        """
        Switch-style load balancing + z-loss.
        logits: (B, N, E), topi: (B, N, K)
        """
        # Importance: mean softmax probability per expert
        probs = F.softmax(logits, dim=-1)  # (B, N, E)
        importance = probs.mean(dim=(0, 1))  # (E,)

        # Load: fraction of tokens routed to each expert (top-1)
        top1 = topi[..., 0]  # (B, N)
        load = F.one_hot(top1, num_classes=self.num_experts).float().mean(dim=(0, 1))  # (E,)

        # Switch transformer aux (scaled)
        aux = (importance * load).sum() * (self.num_experts ** 2)

        # z-loss to keep router logits bounded
        z = torch.logsumexp(logits, dim=-1)
        z_loss = (z ** 2).mean()

        return aux + 0.1 * z_loss

    def forward(self, x: Tensor) -> Tensor:
        """
        x: (B, C, H, W)
        """
        b, c, h, w = x.shape
        tokens = x.permute(0, 2, 3, 1).contiguous().view(b, h * w, c)  # (B, N, C)

        logits = self.router(tokens)  # (B, N, E)
        if self.training and self.router_noise_std > 0:
            logits = logits + torch.randn_like(logits) * self.router_noise_std

        topv, topi = logits.topk(self.top_k, dim=-1)  # (B, N, K)
        wts = F.softmax(topv, dim=-1)  # (B, N, K)

        # Compute MoE aux loss (load-balance + z-loss) and, if enabled, *inject its gradient*
        # into the router path so the training loop does not need to be modified.
        #
        # Rationale:
        # - traiNNer-style SR loops usually assume `forward(x) -> sr` and do not add model-specific
        #   regularizers automatically.
        # - We emulate adding `aux_loss_weight * aux_loss` to the total objective by adding
        #   `aux_loss_weight * d(aux)/d(logits)` to the gradient flowing through `logits`.
        aux = self._compute_aux(logits, topi)
        self.last_aux_loss_detached = aux.detach()

        if self.training and self.aux_loss_weight > 0:
            # Compute d(aux)/d(logits) once (no higher-order grads), then add it to the upstream grad.
            aux_grad = torch.autograd.grad(
                aux,
                logits,
                retain_graph=True,
                create_graph=False,
                allow_unused=False,
            )[0]
            aux_w = float(self.aux_loss_weight)
            logits.register_hook(lambda g, ag=aux_grad, aw=aux_w: g + aw * ag)

            # Keep a detached scalar for logging/monitoring.
            self.last_aux_loss = (aux.detach() * aux_w)
        else:
            self.last_aux_loss = None

        out = torch.zeros_like(tokens)
        for e_idx, expert in enumerate(self.experts):
            for k in range(self.top_k):
                sel = topi[..., k] == e_idx  # (B, N)
                if not sel.any():
                    continue
                x_e = tokens[sel]  # (T, C)
                y_e = expert(x_e)  # (T, C)
                out[sel] += y_e * wts[..., k][sel].unsqueeze(-1)

        out = out.view(b, h, w, c).permute(0, 3, 1, 2).contiguous()
        y = x + self.scale * out

        # Aux loss (if enabled) is applied internally via router-gradient injection.
        return y


# -------------------------
# Re-parameterizable large-kernel conv (explicit deploy switch)
# -------------------------

class ReparamLargeKernelConv(nn.Module):
    """
    Train-time: sum of multiple depthwise conv branches (identity + 1x1 + 3x3 + kxk).
    Deploy: fuse into a single depthwise kxk conv.

    Important: fusing is explicit (switch_to_deploy) to avoid breaking train<->eval loops.
    """

    def __init__(self, channels: int, kernel_size: int = 9) -> None:
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd")
        self.channels = channels
        self.kernel_size = kernel_size
        self.pad = kernel_size // 2

        self.deploy = False

        self.dw_k = nn.Conv2d(channels, channels, kernel_size, padding=self.pad, groups=channels, bias=True)
        self.dw_3 = nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False)
        self.dw_1 = nn.Conv2d(channels, channels, 1, padding=0, groups=channels, bias=False)

        self.alpha = nn.Parameter(torch.ones(4))  # [id, 1x1, 3x3, kxk]

    def _fuse_kernel(self) -> Tuple[Tensor, Tensor]:
        """
        Return fused (weight, bias) for a single depthwise kxk conv.
        """
        c = self.channels
        k = self.kernel_size
        device = self.dw_k.weight.device
        dtype = self.dw_k.weight.dtype

        # Start with kxk branch
        w = self.alpha[3] * self.dw_k.weight
        b = self.alpha[3] * (self.dw_k.bias if self.dw_k.bias is not None else torch.zeros(c, device=device, dtype=dtype))

        # Add 3x3 into center
        w3 = self.dw_3.weight
        pad = (k - 3) // 2
        w[:, :, pad:pad + 3, pad:pad + 3] += self.alpha[2] * w3

        # Add 1x1 into center
        w1 = self.dw_1.weight
        mid = k // 2
        w[:, :, mid:mid + 1, mid:mid + 1] += self.alpha[1] * w1

        # Add identity (depthwise) into center
        eye = torch.zeros((c, 1, k, k), device=device, dtype=dtype)
        eye[:, 0, mid, mid] = 1.0
        w += self.alpha[0] * eye

        return w, b

    @torch.no_grad()
    def switch_to_deploy(self) -> None:
        if self.deploy:
            return
        w, b = self._fuse_kernel()
        self.reparam = nn.Conv2d(self.channels, self.channels, self.kernel_size, padding=self.pad, groups=self.channels, bias=True)
        self.reparam.weight.copy_(w)
        self.reparam.bias.copy_(b)

        # Remove branches
        del self.dw_k
        del self.dw_3
        del self.dw_1
        del self.alpha
        self.deploy = True

    def forward(self, x: Tensor) -> Tensor:
        if self.deploy:
            return self.reparam(x)

        y = 0.0
        y = y + self.alpha[3] * self.dw_k(x)
        y = y + self.alpha[2] * self.dw_3(x)
        y = y + self.alpha[1] * self.dw_1(x)
        y = y + self.alpha[0] * x
        return y


class AdaptiveNoiseInjection(nn.Module):
    """
    Conditioned noise injection inside the HF branch.
    - Warmup gating (strength starts low).
    - Optional colored noise via cheap Gaussian blur (k=3).
    """

    def __init__(
        self,
        channels: int,
        cond_channels: int,
        warmup_iters: int = 4000,
        colored_noise: bool = True,
    ) -> None:
        super().__init__()
        self.channels = channels
        self.warmup_iters = int(warmup_iters)
        self.colored_noise = bool(colored_noise)

        self.noise_proj = nn.Conv2d(1, channels, 1)

        hidden = max(cond_channels // 4, 8)
        self.amp = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(cond_channels, hidden, 1),
            nn.GELU(),
            nn.Conv2d(hidden, channels, 1),
            nn.Sigmoid(),
        )
        self.gate = nn.Sequential(
            nn.Conv2d(cond_channels, hidden, 1),
            nn.GELU(),
            nn.Conv2d(hidden, channels, 1),
            nn.Sigmoid(),
        )

        self.strength = nn.Parameter(torch.tensor(-2.0))  # sigmoid -> small initially
        self._step: int = 0  # python counter; avoid EMA buffer copy issues

        if self.colored_noise:
            self.blur = DepthwiseGaussianBlur(1, kernel_size=3)

    def _warmup_scale(self, device: torch.device, dtype: torch.dtype) -> Tensor:
        if (not self.training) or (self.warmup_iters <= 0):
            return torch.ones((), device=device, dtype=dtype)
        t = min(float(self._step) / float(self.warmup_iters), 1.0)
        return torch.tensor(t, device=device, dtype=dtype)

    def forward(self, x: Tensor, cond: Tensor, noise: Optional[Tensor] = None) -> Tensor:
        if self.training:
            self._step += 1

        if noise is None:
            noise = torch.randn(x.shape[0], 1, x.shape[2], x.shape[3], device=x.device, dtype=x.dtype)
        if self.colored_noise:
            noise = self.blur(noise)

        proj = self.noise_proj(noise)
        amp = self.amp(cond)
        gate = self.gate(cond)

        s = torch.sigmoid(self.strength) * self._warmup_scale(device=x.device, dtype=x.dtype)
        return x + (gate * amp) * proj * s


class FrequencyTextureHead(nn.Module):
    """
    Frequency head:
    - LF refine keeps structure.
    - HF branch adds controllable texture (conditioned noise + large-kernel depthwise conv).
    - Cross-frequency gating prevents HF from breaking geometry.
    """

    def __init__(self, dim: int, norm_groups: int = 0, noise_warmup: int = 4000) -> None:
        super().__init__()
        self.split = HybridFreqSplit(dim, blur_ks=5)

        self.lf_ref = nn.Sequential(
            ConvGatedBlock(dim, norm_groups=norm_groups),
            ConvGatedBlock(dim, norm_groups=norm_groups),
        )

        self.hf_pre = ConvGatedBlock(dim, norm_groups=norm_groups)
        self.noise = AdaptiveNoiseInjection(dim, cond_channels=dim, warmup_iters=noise_warmup, colored_noise=True)
        self.hf_lk = ReparamLargeKernelConv(dim, kernel_size=9)
        self.hf_post = ConvGatedBlock(dim, norm_groups=norm_groups)

        self.xfuse = CrossFrequencyFusion(dim)
        self.merge = nn.Conv2d(dim * 2, dim, 1)

    def forward(self, x: Tensor) -> Tensor:
        lf, hf = self.split(x)
        lf = self.lf_ref(lf)

        hf = self.hf_pre(hf)
        hf = self.noise(hf, cond=lf)
        hf = self.hf_lk(hf)
        hf = self.hf_post(hf)

        lf, hf = self.xfuse(lf, hf)
        return self.merge(torch.cat([lf, hf], dim=1))


# -------------------------
# Encoder / Decoder
# -------------------------

class Downsample(nn.Sequential):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__(nn.Conv2d(in_ch, out_ch, 3, stride=2, padding=1))


class Upsample(nn.Sequential):
    """Conv -> PixelShuffle(2) expressed as a simple Sequential chain."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__(
            nn.Conv2d(in_ch, out_ch * 4, 3, padding=1),
            nn.PixelShuffle(2),
        )


class Stage(nn.Module):
    """
    Stage of repeated blocks.
    If shift=True and use_attn=True, shift alternates by depth: [no-shift, shift, no-shift, ...].
    """

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
        blocks = []
        for i in range(depth):
            shift_i = bool(shift and use_attn and (i % 2 == 1))
            blocks.append(
                AttnFFNBlock(
                    dim,
                    use_attn=use_attn,
                    window_size=window_size,
                    num_heads=num_heads,
                    shift=shift_i,
                    norm_groups=norm_groups,
                )
            )
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x: Tensor) -> Tensor:
        return self.blocks(x)


class GlobalMixer(nn.Module):
    """Cheap global context at bottleneck (large-kernel depthwise + pointwise)."""

    def __init__(self, dim: int, kernel_size: int = 9) -> None:
        super().__init__()
        pad = kernel_size // 2
        self.dw = nn.Conv2d(dim, dim, kernel_size, padding=pad, groups=dim)
        self.pw = nn.Conv2d(dim, dim, 1)
        self.scale = nn.Parameter(torch.zeros(1))

    def forward(self, x: Tensor) -> Tensor:
        y = self.pw(self.dw(x))
        return x + self.scale * y


# -------------------------
# Canonical PixelShuffle upsampler head
# -------------------------

class PixelShuffleHead(nn.Sequential):
    """
    Canonical SR upsampler:
      pre: Conv(in->mid) + LeakyReLU
      up:  (Conv(mid->4mid) + PS(2)) x n   if scale=2^n
           Conv(mid->9mid) + PS(3)         if scale=3
      out: Conv(mid->out)
    """

    def __init__(self, in_dim: int, out_dim: int, scale: int, mid_dim: Optional[int] = None) -> None:
        if scale != 1 and (not _is_power_of_two(scale)) and scale != 3:
            raise ValueError("scale must be 1, 3, or 2^n")

        if mid_dim is None:
            mid_dim = in_dim

        m = []
        if scale == 1:
            m.append(nn.Conv2d(in_dim, out_dim, 3, 1, 1))
            super().__init__(*m)
            return

        m.extend([nn.Conv2d(in_dim, mid_dim, 3, 1, 1), nn.LeakyReLU(0.2, inplace=True)])

        if _is_power_of_two(scale):
            for _ in range(int(math.log2(scale))):
                m.extend([nn.Conv2d(mid_dim, 4 * mid_dim, 3, 1, 1), nn.PixelShuffle(2)])
        elif scale == 3:
            m.extend([nn.Conv2d(mid_dim, 9 * mid_dim, 3, 1, 1), nn.PixelShuffle(3)])

        m.append(nn.Conv2d(mid_dim, out_dim, 3, 1, 1))
        super().__init__(*m)



# -------------------------
# Modern upsampling option: DySample + MultiMode head
# -------------------------

class DySample(nn.Module):
    """Adapted from 'Learning to Upsample by Learning to Sample':
    https://arxiv.org/abs/2308.15085
    https://github.com/tiny-smart/dysample
    """

    def __init__(
        self,
        in_channels: int = 64,
        out_ch: int = 3,
        scale: int = 2,
        groups: int = 4,
        end_convolution: bool = True,
        end_kernel=1,
    ) -> None:
        super().__init__()

        if in_channels <= groups or in_channels % groups != 0:
            msg = "Incorrect in_channels and groups values."
            raise ValueError(msg)

        out_channels = 2 * groups * scale**2
        self.scale = scale
        self.groups = groups
        self.end_convolution = end_convolution
        if end_convolution:
            self.end_conv = nn.Conv2d(
                in_channels, out_ch, end_kernel, 1, end_kernel // 2
            )
        self.offset = nn.Conv2d(in_channels, out_channels, 1, bias=True)
        self.scope = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        # Start close to identity sampling: offset==0, scope==0 => stable & smooth early training.
        nn.init.zeros_(self.offset.weight)
        nn.init.zeros_(self.offset.bias)
        nn.init.zeros_(self.scope.weight)
        self.register_buffer("init_pos", self._init_pos(), persistent=False)

    def _init_pos(self) -> Tensor:
        h = torch.arange((-self.scale + 1) / 2, (self.scale - 1) / 2 + 1) / self.scale
        return (
            torch.stack(torch.meshgrid([h, h], indexing="ij"))
            .transpose(1, 2)
            .repeat(1, self.groups, 1)
            .reshape(1, -1, 1, 1)
        )

    def forward(self, x: Tensor) -> Tensor:
        offset = self.offset(x) * self.scope(x).sigmoid() * 0.5 + self.init_pos
        offset = offset.float()
        B, _, H, W = offset.shape
        offset = offset.view(B, 2, -1, H, W)
        coords_h = torch.arange(H, device=x.device, dtype=torch.float32) + 0.5
        coords_w = torch.arange(W, device=x.device, dtype=torch.float32) + 0.5

        coords = (
            torch.stack(torch.meshgrid([coords_w, coords_h], indexing="ij"))
            .transpose(1, 2)
            .unsqueeze(1)
            .unsqueeze(0)
        )
        normalizer = torch.tensor([W, H], dtype=torch.float32, device=x.device).view(1, 2, 1, 1, 1)
        coords = 2 * (coords + offset) / normalizer - 1

        coords = (
            F.pixel_shuffle(coords.reshape(B, -1, H, W), self.scale)
            .view(B, 2, -1, self.scale * H, self.scale * W)
            .permute(0, 2, 3, 4, 1)
            .contiguous()
            .flatten(0, 1)
        )
        output = F.grid_sample(
            x.reshape(B * self.groups, -1, H, W),
            coords,
            mode="bilinear",
            align_corners=False,
            padding_mode="border",
        ).view(B, -1, self.scale * H, self.scale * W)

        if self.end_convolution:
            output = self.end_conv(output)

        return output

def _pick_dysample_groups(in_channels: int, requested_groups: int) -> int:
    """Pick a valid DySample groups value for the given channel width."""
    g = int(requested_groups)
    g = max(g, 1)
    # DySample requires: in_channels > groups and divisible by groups
    if in_channels <= g:
        g = max(1, in_channels // 2)
    # Reduce until divisible
    while g > 1 and (in_channels % g != 0):
        g -= 1
    if in_channels <= g:
        g = 1
    return g


class UpsampleHead(nn.Module):
    """Configurable output head: PixelShuffle (canonical) or DySample."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        scale: int,
        mode: str = "dysample",
        dysample_groups: int = 4,
        ps_mid_dim: Optional[int] = None,
    ) -> None:
        super().__init__()
        mode = str(mode).lower()
        self.mode = mode
        self.scale = int(scale)

        if self.scale == 1:
            self.head = nn.Conv2d(in_dim, out_dim, 3, 1, 1)
            return

        if mode in ("pixelshuffle", "ps"):
            self.head = PixelShuffleHead(in_dim, out_dim, scale=self.scale, mid_dim=ps_mid_dim)
        elif mode in ("dysample", "dy"):
            g = _pick_dysample_groups(in_dim, dysample_groups)
            self.head = DySample(in_channels=in_dim, out_ch=out_dim, scale=self.scale, groups=g, end_convolution=True, end_kernel=1)
        else:
            raise ValueError("mode must be 'dysample' or 'pixelshuffle'")

    def forward(self, x: Tensor) -> Tensor:
        return self.head(x)


# -------------------------
# Main model
# -------------------------

@ARCH_REGISTRY.register()
class HyFreqMoE_SR4(nn.Module):
    """
    Revised combined architecture:
    - Conv-first multi-scale encoder/decoder
    - Window attention only at 1/2 and 1/4 scales with alternating shift inside stages
    - Sparse Token-MoE at bottleneck with anti-collapse hooks
    - Frequency + conditioned-noise texture head
    """

    def __init__(
        self,
        scale: int = 4,
        in_ch: int = 3,
        out_ch: int = 3,
        dim: int = 64,
        depths: Tuple[int, int, int] = (4, 4, 6),
        window_size: int = 8,
        num_heads: Tuple[int, int] = (4, 8),  # stage2, stage3
        norm_groups: int = 1,
        moe_experts: int = 6,
        moe_top_k: int = 2,
        moe_router_noise: float = 0.04,
        moe_aux_weight: float = 0.0075,
        noise_warmup: int = 4000,
        upsampler: str = "pixelshuffle",
        dysample_groups: int = 4,
    ) -> None:
        super().__init__()
        if scale != 1 and (not _is_power_of_two(scale)) and scale != 3:
            raise ValueError("scale must be 1, 3, or 2^n")
        self.scale = scale
        self.norm_groups = int(norm_groups)

        # Stem
        self.stem = nn.Conv2d(in_ch, dim, 3, padding=1)

        # Encoder
        self.enc1 = Stage(dim, depths[0], use_attn=False, window_size=window_size, num_heads=1, shift=False, norm_groups=self.norm_groups)
        self.down1 = Downsample(dim, dim * 2)

        self.enc2 = Stage(dim * 2, depths[1], use_attn=True, window_size=window_size, num_heads=num_heads[0], shift=True, norm_groups=self.norm_groups)
        self.down2 = Downsample(dim * 2, dim * 4)

        self.enc3 = Stage(dim * 4, depths[2], use_attn=True, window_size=window_size, num_heads=num_heads[1], shift=True, norm_groups=self.norm_groups)

        # Bottleneck
        self.global_mixer = GlobalMixer(dim * 4, kernel_size=9)
        self.bottleneck_split = BottleneckFreqSplit(dim * 4, blur_ks=5)
        self.token_moe = SparseTopKMoE(
            dim * 4,
            num_experts=moe_experts,
            top_k=moe_top_k,
            hidden_mult=2.0,
            router_noise_std=moe_router_noise,
            aux_loss_weight=moe_aux_weight,
        )
        self.bneck = nn.Sequential(
            AttnFFNBlock(dim * 4, use_attn=True, window_size=window_size, num_heads=num_heads[1], shift=False, norm_groups=self.norm_groups),
            AttnFFNBlock(dim * 4, use_attn=False, window_size=window_size, num_heads=1, shift=False, norm_groups=self.norm_groups),
        )

        # Decoder
        self.up2 = Upsample(dim * 4, dim * 2)
        self.dec2 = Stage(dim * 2, depths[1], use_attn=True, window_size=window_size, num_heads=num_heads[0], shift=True, norm_groups=self.norm_groups)

        self.up1 = Upsample(dim * 2, dim)
        self.dec1 = Stage(dim, depths[0], use_attn=False, window_size=window_size, num_heads=1, shift=False, norm_groups=self.norm_groups)

        # Texture head (pre-upsampling)
        self.texture_head = FrequencyTextureHead(dim, norm_groups=self.norm_groups, noise_warmup=noise_warmup)

        # Upsampling head (DySample or PixelShuffle)
        self.head = UpsampleHead(dim, out_ch, scale=scale, mode=upsampler, dysample_groups=dysample_groups)

        # Exposed aux loss (MoE)
        self._last_aux_loss: Optional[Tensor] = None  # EMA-safe (not a buffer)
        self.moe_aux_weight = float(moe_aux_weight)

    def get_aux_loss(self, clear: bool = True) -> Tensor:
        """
        Return the last MoE auxiliary loss (0 if disabled or not run).
        By default clears internal reference to avoid retaining autograd graph.
        """
        if self._last_aux_loss is None:
            return torch.zeros((), device=next(self.parameters()).device)
        aux = self._last_aux_loss
        if clear:
            self._last_aux_loss = None
            if hasattr(self, "token_moe"):
                self.token_moe.last_aux_loss = None
        return aux

    @torch.no_grad()
    def switch_to_deploy(self) -> None:
        """
        Explicitly fuse re-parameterizable modules for inference/export.
        Safe to call once after training; do not call if you need to resume training.
        """
        for m in self.modules():
            if isinstance(m, ReparamLargeKernelConv):
                m.switch_to_deploy()

    def forward(self, x: Tensor) -> Tensor:
        inp = x
        x = self.stem(x)

        e1 = self.enc1(x)
        e2 = self.enc2(self.down1(e1))
        e3 = self.enc3(self.down2(e2))

        e3 = self.global_mixer(e3)

        # Isolate MoE to HF tokens (BP-03)
        lf_b, hf_b = self.bottleneck_split(e3)
        hf_b = self.token_moe(hf_b)
        self._last_aux_loss = self.token_moe.last_aux_loss  # exposed to trainer
        e3 = lf_b + hf_b

        e3 = self.bneck(e3)

        d2 = self.up2(e3) + e2
        d2 = self.dec2(d2)

        d1 = self.up1(d2) + e1
        d1 = self.dec1(d1)

        d1 = self.texture_head(d1)

        out = self.head(d1)
        if self.scale == 1 and out.shape == inp.shape:
            out = out + inp
        return out