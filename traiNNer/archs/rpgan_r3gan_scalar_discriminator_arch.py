# noqa: D100,D101,D102,D103,D104
"""RpGAN / R3GAN Scalar Discriminator (Modern ResNet / ConvNeXt-style)

This file provides a traiNNer-redux compatible discriminator that matches the
authors' reference implementation used for RpGAN/R3GAN (see `Networks.py`,
`FusedOperators.py`, `Resamplers.py`).

Key properties (matching the reference):
- Fully-convolutional *encoder* built from "Modern ResNet / ConvNeXt-style"
  residual blocks: 1x1 expand -> grouped conv -> 1x1 project, with per-channel
  learned bias + LeakyReLU between linear layers.
- Anti-aliased downsampling by 2 using a low-pass FIR kernel created from the
  1D weights (default [1,2,1]) via the exact CreateLowpassKernel logic.
- Final 4x4 spatial map is reduced by a depthwise 4x4 conv, then a linear layer
  produces a scalar score per sample (or an embedding for projection conditioning).
- Optional projection conditioning (same as reference): dot(score_vec, emb(y)).

Important for R3GAN:
R1/R2 gradient penalties require higher-order autograd through D. This
architecture is intentionally simple (no U-Net decoder, no attention, no FFT)
to keep double-backprop practical.

Usage:
- Drop this file into your `traiNNer/archs/` (or equivalent) and ensure it is imported
  so the registry sees the class.
- Configure `type: RpGANDiscriminatorScalar` in your YAML.

References:
- authors' Networks.py Discriminator, DiscriminatorStage, ResidualBlock, Convolution,
  DiscriminativeBasis, MSRInitializer
- authors' FusedOperators.py BiasedActivation (bias_act)
- authors' Resamplers.py InterpolativeDownsampler (upfirdn2d) and CreateLowpassKernel
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F  # noqa: N812

from traiNNer.utils.registry import ARCH_REGISTRY


# -----------------------------------------------------------------------------
# Fused operators (optional) - match authors' FusedOperators.py semantics
# -----------------------------------------------------------------------------

try:
    # trainner-redux commonly ships StyleGAN2 ops in torch_utils.ops
    from torch_utils.ops import bias_act  # type: ignore
    _HAS_BIAS_ACT = True
except Exception:  # pragma: no cover
    bias_act = None
    _HAS_BIAS_ACT = False


class BiasedActivation(nn.Module):
    """Per-channel learned bias + LeakyReLU (gain=1 in forward), matching reference.

    Note: The reference defines a constant Gain = sqrt(2/(1+0.2^2)) used ONLY for
    initialization scaling. The forward does NOT multiply by Gain (gain=1 in bias_act).
    """

    Gain: float = math.sqrt(2.0 / (1.0 + 0.2**2))
    _NEG_SLOPE: float = 0.2

    def __init__(self, channels: int, use_fused: bool = True) -> None:
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(channels))
        self.use_fused = bool(use_fused) and _HAS_BIAS_ACT

    def forward(self, x: Tensor) -> Tensor:
        if self.use_fused:
            # exact reference: bias_act.bias_act(... act='lrelu', gain=1)
            return bias_act.bias_act(x, self.bias.to(x.dtype), act="lrelu", gain=1)  # type: ignore[attr-defined]
        # fallback: pure PyTorch (supports higher-order autograd)
        b = self.bias.to(dtype=x.dtype)
        if x.ndim == 4:
            x = x + b.view(1, -1, 1, 1)
        else:
            x = x + b.view(1, -1)
        return F.leaky_relu(x, negative_slope=self._NEG_SLOPE, inplace=False)


# -----------------------------------------------------------------------------
# Resamplers - match authors' Resamplers.py CreateLowpassKernel + downsample2d
# -----------------------------------------------------------------------------

try:
    from torch_utils.ops import upfirdn2d  # type: ignore
    _HAS_UPFIRDN = True
except Exception:  # pragma: no cover
    upfirdn2d = None
    _HAS_UPFIRDN = False


def _convolve_with_ones(weights: Sequence[float]) -> list[float]:
    """Equivalent to numpy.convolve(weights, [1, 1]) for 1D weights."""
    if len(weights) == 0:
        raise ValueError("weights must be non-empty")
    out = [float(weights[0])]
    for i in range(1, len(weights)):
        out.append(float(weights[i - 1]) + float(weights[i]))
    out.append(float(weights[-1]))
    return out


def create_lowpass_kernel(weights: Sequence[float], inplace: bool) -> Tensor:
    """Match authors' CreateLowpassKernel.

    - if inplace: use weights directly (1D)
    - else: convolve weights with [1,1], then outer-product to 2D
    - normalize by sum
    """
    w1d = list(map(float, weights))
    if not inplace:
        w1d = _convolve_with_ones(w1d)
    w = torch.tensor(w1d, dtype=torch.float32)
    k2d = torch.outer(w, w)
    k2d = k2d / k2d.sum()
    return k2d


class InterpolativeDownsampler(nn.Module):
    """Anti-aliased /2 downsampling with a fixed low-pass kernel.

    Default semantics match the CUDA version of authors' Resamplers.py:
    upfirdn2d.downsample2d(x, kernel)

    Fallback matches InterpolativeDownsamplerReference (per-channel conv2d stride=2).
    """

    def __init__(self, filt: Sequence[float] = (1, 2, 1), *, use_cuda_impl: bool = True) -> None:
        super().__init__()
        self.use_cuda_impl = bool(use_cuda_impl) and _HAS_UPFIRDN
        k = create_lowpass_kernel(filt, inplace=False)
        self.register_buffer("kernel", k)

        # reference uses padding = len(Filter)//2 (FilterRadius)
        self.filter_radius = len(filt) // 2

    def forward(self, x: Tensor) -> Tensor:
        if self.use_cuda_impl and x.is_cuda:
            return upfirdn2d.downsample2d(x, self.kernel)  # type: ignore[attr-defined]

        # Reference fallback: per-channel conv2d on (B*C,1,H,W)
        b, c, h, w = x.shape
        k = self.kernel.to(device=x.device, dtype=x.dtype).view(1, 1, *self.kernel.shape)
        y = F.conv2d(x.reshape(b * c, 1, h, w), k, stride=2, padding=self.filter_radius)
        return y.reshape(b, c, y.shape[-2], y.shape[-1])


# -----------------------------------------------------------------------------
# Reference-like MSR initializer + convolution wrapper (weights cast to x.dtype)
# -----------------------------------------------------------------------------

def msr_init_(layer: nn.Module, activation_gain: float = 1.0) -> nn.Module:
    """Match authors' MSRInitializer for Conv2d/Linear."""
    if isinstance(layer, nn.Conv2d):
        fan_in = layer.weight.data.size(1) * layer.weight.data[0][0].numel()
    elif isinstance(layer, nn.Linear):
        fan_in = layer.weight.data.size(1)
    else:
        return layer
    layer.weight.data.normal_(0.0, float(activation_gain) / math.sqrt(fan_in))
    if getattr(layer, "bias", None) is not None:
        layer.bias.data.zero_()
    return layer


class Convolution(nn.Module):
    """Conv2d wrapper matching authors' Convolution.

    - Internally stores an nn.Conv2d without bias.
    - Forward uses F.conv2d with weights cast to x.dtype (important under AMP).
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: int,
        *,
        groups: int = 1,
        activation_gain: float = 1.0,
    ) -> None:
        super().__init__()
        conv = nn.Conv2d(
            in_ch,
            out_ch,
            kernel_size=kernel_size,
            stride=1,
            padding=(kernel_size - 1) // 2,
            groups=groups,
            bias=False,
        )
        self.layer = msr_init_(conv, activation_gain=activation_gain)

    @property
    def padding(self) -> tuple[int, int]:
        return self.layer.padding  # type: ignore[return-value]

    @property
    def groups(self) -> int:
        return int(self.layer.groups)

    def forward(self, x: Tensor) -> Tensor:
        w = self.layer.weight.to(dtype=x.dtype)
        return F.conv2d(x, w, padding=self.padding, groups=self.groups)


# -----------------------------------------------------------------------------
# Core RpGAN discriminator blocks (1:1 with Networks.py semantics)
# -----------------------------------------------------------------------------

class ResidualBlock(nn.Module):
    """Modern ResNet / ConvNeXt-style residual block (authors' ResidualBlock)."""

    def __init__(
        self,
        in_ch: int,
        *,
        cardinality: int,
        expansion_factor: int,
        kernel_size: int,
        variance_scaling_param: int,
        use_fused_ops: bool = True,
    ) -> None:
        super().__init__()
        n_linear = 3
        expanded = int(in_ch) * int(expansion_factor)
        card = int(cardinality)
        if expanded % card != 0:
            raise ValueError(f"ExpandedChannels={expanded} must be divisible by Cardinality={card}.")

        # Reference:
        # ActivationGain = BiasedActivation.Gain * VSP ** (-1/(2*n_linear-2))
        activation_gain = BiasedActivation.Gain * float(variance_scaling_param) ** (
            -1.0 / (2 * n_linear - 2)
        )

        self.lin1 = Convolution(in_ch, expanded, kernel_size=1, activation_gain=activation_gain)
        self.lin2 = Convolution(
            expanded,
            expanded,
            kernel_size=kernel_size,
            groups=card,
            activation_gain=activation_gain,
        )
        # Third linear layer uses ActivationGain=0 (=> zero init, start close to identity)
        self.lin3 = Convolution(expanded, in_ch, kernel_size=1, activation_gain=0.0)

        self.act1 = BiasedActivation(expanded, use_fused=use_fused_ops)
        self.act2 = BiasedActivation(expanded, use_fused=use_fused_ops)

    def forward(self, x: Tensor) -> Tensor:
        y = self.lin1(x)
        y = self.lin2(self.act1(y))
        y = self.lin3(self.act2(y))
        return x + y


class DownsampleLayer(nn.Module):
    """Authors' DownsampleLayer: InterpolativeDownsampler then optional 1x1 conv."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        resampling_filter: Sequence[float],
        *,
        use_cuda_resampler: bool = True,
    ) -> None:
        super().__init__()
        self.resampler = InterpolativeDownsampler(resampling_filter, use_cuda_impl=use_cuda_resampler)
        self.linear = None
        if int(in_ch) != int(out_ch):
            self.linear = Convolution(in_ch, out_ch, kernel_size=1, activation_gain=1.0)

    def forward(self, x: Tensor) -> Tensor:
        x = self.resampler(x)
        if self.linear is not None:
            x = self.linear(x)
        return x


class DiscriminativeBasis(nn.Module):
    """Authors' DiscriminativeBasis: depthwise 4x4 conv -> linear."""

    def __init__(self, in_ch: int, out_dim: int) -> None:
        super().__init__()
        basis = nn.Conv2d(in_ch, in_ch, kernel_size=4, stride=1, padding=0, groups=in_ch, bias=False)
        self.basis = msr_init_(basis, activation_gain=1.0)
        linear = nn.Linear(in_ch, out_dim, bias=False)
        self.linear = msr_init_(linear, activation_gain=1.0)

    def forward(self, x: Tensor) -> Tensor:
        # Reference expects x to be (B,C,4,4) so basis(x) is (B,C,1,1).
        # In SR training the patch size may vary; to keep the exact Basis (4x4 depthwise)
        # while remaining robust, adaptively pool to 4x4 when needed.
        if x.shape[-2:] != (4, 4):
            x = F.adaptive_avg_pool2d(x, (4, 4))
        x = self.basis(x)
        x = x.reshape(x.shape[0], -1)
        return self.linear(x)


class DiscriminatorStage(nn.Module):
    """Authors' DiscriminatorStage: N residual blocks then transition (downsample or basis)."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        *,
        cardinality: int,
        num_blocks: int,
        expansion_factor: int,
        kernel_size: int,
        variance_scaling_param: int,
        resampling_filter: Optional[Sequence[float]] = (1, 2, 1),
        stage_dtype: Optional[torch.dtype] = None,
        use_fused_ops: bool = True,
        use_cuda_resampler: bool = True,
    ) -> None:
        super().__init__()
        self.stage_dtype = stage_dtype

        blocks = [
            ResidualBlock(
                in_ch,
                cardinality=cardinality,
                expansion_factor=expansion_factor,
                kernel_size=kernel_size,
                variance_scaling_param=variance_scaling_param,
                use_fused_ops=use_fused_ops,
            )
            for _ in range(int(num_blocks))
        ]

        if resampling_filter is None:
            transition: nn.Module = DiscriminativeBasis(in_ch, out_ch)
        else:
            transition = DownsampleLayer(in_ch, out_ch, resampling_filter, use_cuda_resampler=use_cuda_resampler)

        self.layers = nn.ModuleList(blocks + [transition])

    def forward(self, x: Tensor) -> Tensor:
        if self.stage_dtype is not None:
            x = x.to(self.stage_dtype)
        for layer in self.layers:
            x = layer(x)
        return x


# -----------------------------------------------------------------------------
# Public traiNNer-redux discriminator
# -----------------------------------------------------------------------------

@ARCH_REGISTRY.register()
class RpGANDiscriminatorScalar(nn.Module):
    """RpGAN/R3GAN scalar discriminator (Modern ResNet / ConvNeXt-style).

    This is a clean reimplementation of the authors' `Discriminator` in Networks.py.

    Input:
        x: Tensor (B, C, H, W) typically C=3.
        y: optional condition tensor (B, condition_dim) for projection conditioning.

    Output:
        Tensor (B,) - scalar score per sample.
    """

    def __init__(
        self,
        num_in_ch: int = 3,
        width_per_stage: Sequence[int] = (32, 64, 128, 256, 256),
        cardinality_per_stage: Sequence[int] = (1, 1, 2, 8, 8),
        blocks_per_stage: Sequence[int] = (1, 1, 1, 1, 1),
        expansion_factor: int = 2,
        kernel_size: int = 3,
        resampling_filter: Sequence[float] = (1, 2, 1),
        # dtype control: match reference default (float32). Set None to respect AMP.
        stage_dtype: Optional[torch.dtype] = torch.float32,
        # conditioning
        condition_dim: Optional[int] = None,
        condition_embedding_dim: int = 0,
        # implementation toggles (keep architecture the same)
        use_fused_ops: bool = True,
        use_cuda_resampler: bool = True,
        # safety: if True, adaptively pool the final feature map to 4x4 before DiscriminativeBasis.
        # This is OFF by default to match the reference strictly.
        force_4x4: bool = False,
    ) -> None:
        super().__init__()
        self.num_in_ch = int(num_in_ch)
        self.force_4x4 = bool(force_4x4)

        w = [int(x) for x in width_per_stage]
        c = [int(x) for x in cardinality_per_stage]
        b = [int(x) for x in blocks_per_stage]
        if not (len(w) == len(c) == len(b)):
            raise ValueError(
                "width_per_stage, cardinality_per_stage, blocks_per_stage must have the same length."
            )
        if len(w) < 1:
            raise ValueError("width_per_stage must be non-empty")

        self.stage_dtype = stage_dtype

        # VarianceScalingParameter = sum(BlocksPerStage) in reference
        vsp = sum(b)

        # ExtractionLayer: Convolution(3 -> width0, 1x1)
        self.extraction = Convolution(self.num_in_ch, w[0], kernel_size=1, activation_gain=1.0)

        # Main layers:
        # for x in range(len(WidthPerStage)-1): stage(width[x] -> width[x+1], resampling_filter)
        # plus final stage: stage(width[-1] -> out_dim, resampling_filter=None)
        stages: list[nn.Module] = []

        for i in range(len(w) - 1):
            stages.append(
                DiscriminatorStage(
                    w[i],
                    w[i + 1],
                    cardinality=c[i],
                    num_blocks=b[i],
                    expansion_factor=expansion_factor,
                    kernel_size=kernel_size,
                    variance_scaling_param=vsp,
                    resampling_filter=resampling_filter,
                    stage_dtype=stage_dtype,
                    use_fused_ops=use_fused_ops,
                    use_cuda_resampler=use_cuda_resampler,
                )
            )

        out_dim = 1 if condition_dim is None else int(condition_embedding_dim)
        if condition_dim is not None and out_dim <= 0:
            raise ValueError("condition_embedding_dim must be > 0 when condition_dim is provided.")

        stages.append(
            DiscriminatorStage(
                w[-1],
                out_dim,
                cardinality=c[-1],
                num_blocks=b[-1],
                expansion_factor=expansion_factor,
                kernel_size=kernel_size,
                variance_scaling_param=vsp,
                resampling_filter=None,
                stage_dtype=stage_dtype,
                use_fused_ops=use_fused_ops,
                use_cuda_resampler=use_cuda_resampler,
            )
        )

        self.main_layers = nn.ModuleList(stages)

        self.condition_dim = condition_dim
        if condition_dim is not None:
            emb = nn.Linear(int(condition_dim), int(condition_embedding_dim), bias=False)
            # Reference: ActivationGain=1/sqrt(ConditionEmbeddingDimension)
            self.embedding = msr_init_(emb, activation_gain=1.0 / math.sqrt(int(condition_embedding_dim)))

    def forward(self, x: Tensor, y: Optional[Tensor] = None) -> Tensor:
        # Reference: x = ExtractionLayer(x.to(MainLayers[0].DataType))
        if self.stage_dtype is not None:
            x = x.to(self.stage_dtype)
        x = self.extraction(x)

        for layer in self.main_layers[:-1]:
            x = layer(x)

        # Final stage produces (B,1) or (B,embed_dim) via DiscriminativeBasis
        if self.force_4x4 and (x.shape[-2:] != (4, 4)):
            x = F.adaptive_avg_pool2d(x, (4, 4))
        x = self.main_layers[-1](x)

        if self.condition_dim is not None:
            if y is None:
                raise ValueError("RpGANDiscriminatorScalar is conditional but y=None was passed.")
            y_emb = self.embedding(y)
            x = (x * y_emb).sum(dim=1, keepdim=True)

        return x.reshape(x.shape[0])
