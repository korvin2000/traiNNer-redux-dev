from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Tuple

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class HumArchConfig:
    base_channels: int = 48
    style_dim: int = 128
    noise_stride: int = 16
    pretex_blocks: int = 8
    trunk_blocks: tuple[int, int, int] = (2, 2, 3)
    trunk_refine_blocks: int = 6
    hrtex_blocks: int = 6
    thin_alpha: tuple[float, float, float] = (0.08, 0.1, 0.12)  # scale 1/2/4
    mid_alpha: tuple[float, float, float] = (0.05, 0.07, 0.08)
    edge_gain: float = 3.0
    line_kernel_size: int = 7
    enable_confidence_brake: bool = True
    conf_brake_power: float = 1.5
    conf_use_edge_gate: bool = True
    conf_orient_weight: float = 0.5
    split_struct_alpha: bool = True
    thin_hair_weight: float = 1.0
    thin_lash_weight: float = 0.85

    def alpha_for_scale(self, scale: int) -> tuple[float, float]:
        if scale == 1:
            return self.thin_alpha[0], self.mid_alpha[0]
        if scale == 2:
            return self.thin_alpha[1], self.mid_alpha[1]
        if scale == 4:
            return self.thin_alpha[2], self.mid_alpha[2]
        raise ValueError(f"Unsupported scale: {scale}")


def make_norm(channels: int) -> nn.GroupNorm:
    groups = 8
    if channels < groups:
        groups = 1
    elif channels % groups != 0:
        groups = math.gcd(channels, groups)
        if groups == 0:
            groups = 1
    return nn.GroupNorm(groups, channels)


class FiLM(nn.Module):
    def __init__(self, style_dim: int, channels: int) -> None:
        super().__init__()
        self.proj = nn.Linear(style_dim, channels * 2)

    def forward(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        scale, shift = self.proj(z).chunk(2, dim=1)
        scale = scale[:, :, None, None]
        shift = shift[:, :, None, None]
        return x * (1 + scale) + shift


class GatedDWBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 3) -> None:
        super().__init__()
        self.norm = make_norm(channels)
        self.pw = nn.Conv2d(channels, channels * 2, 1)
        self.dw = nn.Conv2d(
            channels, channels, kernel_size, padding=kernel_size // 2, groups=channels
        )
        self.act = nn.GELU()
        self.out = nn.Conv2d(channels, channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm(x)
        x = self.pw(x)
        gate, feat = x.chunk(2, dim=1)
        feat = self.dw(feat)
        feat = self.act(feat)
        x = self.out(gate * feat)
        return residual + x * self.gamma


class StyleEncoder(nn.Module):
    def __init__(self, in_channels: int, base_channels: int, style_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(base_channels, base_channels * 2, 3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(base_channels * 2, base_channels * 4, 3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(base_channels * 4, base_channels * 4, 3, stride=2, padding=1),
            nn.GELU(),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(base_channels * 4, base_channels * 4),
            nn.GELU(),
            nn.Linear(base_channels * 4, style_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.net(x)
        x = self.pool(x)
        return self.mlp(x)


def gradient_magnitude(x: torch.Tensor) -> torch.Tensor:
    gray = x.mean(dim=1, keepdim=True)
    kernel_x = torch.tensor(
        [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
        device=x.device,
        dtype=x.dtype,
    )
    kernel_y = kernel_x.t()
    kernel_x = kernel_x.view(1, 1, 3, 3)
    kernel_y = kernel_y.view(1, 1, 3, 3)
    grad_x = F.conv2d(gray, kernel_x, padding=1)
    grad_y = F.conv2d(gray, kernel_y, padding=1)
    return torch.sqrt(grad_x**2 + grad_y**2 + 1e-6)


def make_border_mask(height: int, width: int, device: torch.device) -> torch.Tensor:
    wy = torch.hann_window(height, periodic=False, device=device)
    wx = torch.hann_window(width, periodic=False, device=device)
    mask = wy[:, None] * wx[None, :]
    return mask.clamp_min(1e-3)[None, None]


def _pixel_unshuffle(x: torch.Tensor, scale: int) -> torch.Tensor:
    return F.pixel_unshuffle(x, scale)


def _pixel_shuffle(x: torch.Tensor, scale: int) -> torch.Tensor:
    return F.pixel_shuffle(x, scale)


def haar_decompose(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if x.shape[-1] % 2 != 0 or x.shape[-2] % 2 != 0:
        raise ValueError("Haar decomposition requires even spatial dimensions.")
    x = _pixel_unshuffle(x, 2)
    b, c4, h, w = x.shape
    x = x.view(b, c4 // 4, 4, h, w)
    a, b1, c1, d = x[:, :, 0], x[:, :, 1], x[:, :, 2], x[:, :, 3]
    ll = (a + b1 + c1 + d) * 0.5
    lh = (a + b1 - c1 - d) * 0.5
    hl = (a - b1 + c1 - d) * 0.5
    hh = (a - b1 - c1 + d) * 0.5
    return ll, lh, hl, hh


def haar_reconstruct(
    ll: torch.Tensor, lh: torch.Tensor, hl: torch.Tensor, hh: torch.Tensor
) -> torch.Tensor:
    a = (ll + lh + hl + hh) * 0.5
    b1 = (ll + lh - hl - hh) * 0.5
    c1 = (ll - lh + hl - hh) * 0.5
    d = (ll - lh - hl + hh) * 0.5
    x = torch.stack([a, b1, c1, d], dim=2)
    b, c, _, h, w = x.shape
    x = x.view(b, c * 4, h, w)
    return _pixel_shuffle(x, 2)


def haar_from_highbands(lh: torch.Tensor, hl: torch.Tensor, hh: torch.Tensor) -> torch.Tensor:
    ll = torch.zeros_like(lh)
    return haar_reconstruct(ll, lh, hl, hh)


def downsample_avg(x: torch.Tensor) -> torch.Tensor:
    return F.avg_pool2d(x, 2)


def upsample_nearest(x: torch.Tensor, scale: int) -> torch.Tensor:
    return F.interpolate(x, scale_factor=scale, mode="nearest")


def bandpass_level2(x: torch.Tensor) -> torch.Tensor:
    low1 = downsample_avg(x)
    low2 = downsample_avg(low1)
    up2 = F.interpolate(low2, size=low1.shape[-2:], mode="bilinear", align_corners=False)
    band2 = low1 - up2
    return band2


def bandpass_level1(x: torch.Tensor) -> torch.Tensor:
    low = downsample_avg(x)
    up = F.interpolate(low, size=x.shape[-2:], mode="bilinear", align_corners=False)
    return x - up


def upsample_band_to_full(band: torch.Tensor, full_size: Iterable[int]) -> torch.Tensor:
    return F.interpolate(band, size=full_size, mode="bilinear", align_corners=False)


def build_oriented_kernels(size: int, device: torch.device) -> torch.Tensor:
    if size % 2 == 0:
        raise ValueError("Kernel size must be odd.")
    radius = size // 2
    coords = torch.linspace(-radius, radius, steps=size, device=device)
    yy, xx = torch.meshgrid(coords, coords, indexing="ij")
    kernels = []
    for angle in torch.arange(8, device=device, dtype=coords.dtype) * (math.pi / 8.0):
        cos_a = torch.cos(angle)
        sin_a = torch.sin(angle)
        x_rot = xx * cos_a + yy * sin_a
        y_rot = -xx * sin_a + yy * cos_a
        sigma_x = 1.0
        sigma_y = 2.5
        kernel = torch.exp(-(x_rot**2) / (2 * sigma_x**2)) * torch.exp(
            -(y_rot**2) / (2 * sigma_y**2)
        )
        kernel = kernel - kernel.mean()
        kernel = kernel / (kernel.abs().sum() + 1e-6)
        kernels.append(kernel)
    weight = torch.stack(kernels, dim=0).unsqueeze(1)
    return weight


def apply_oriented_kernels(seed: torch.Tensor, kernels: torch.Tensor) -> torch.Tensor:
    return F.conv2d(seed, kernels, padding=kernels.shape[-1] // 2)


def high_pass(x: torch.Tensor) -> torch.Tensor:
    blurred = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
    return x - blurred


class BaseHead(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.to_rgb = nn.Conv2d(channels, 3, 3, padding=1)
        self.up2 = nn.Sequential(
            nn.Conv2d(channels, channels * 4, 3, padding=1),
            nn.PixelShuffle(2),
            GatedDWBlock(channels),
        )
        self.up4_1 = nn.Sequential(
            nn.Conv2d(channels, channels * 4, 3, padding=1),
            nn.PixelShuffle(2),
            GatedDWBlock(channels),
        )
        self.up4_2 = nn.Sequential(
            nn.Conv2d(channels, channels * 4, 3, padding=1),
            nn.PixelShuffle(2),
            GatedDWBlock(channels),
        )
        self.out = nn.Conv2d(channels, 3, 3, padding=1)

    def forward(self, x: torch.Tensor, scale: int) -> torch.Tensor:
        if scale == 1:
            return self.to_rgb(x)
        if scale == 2:
            x = self.up2(x)
            return self.out(x)
        if scale == 4:
            x = self.up4_1(x)
            x = self.up4_2(x)
            return self.out(x)
        raise ValueError(f"Unsupported scale {scale}")


class PreTex(nn.Module):
    def __init__(self, channels: int, blocks: int) -> None:
        super().__init__()
        self.net = nn.Sequential(*[GatedDWBlock(channels) for _ in range(blocks)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class HRTextureHead(nn.Module):
    def __init__(self, in_channels: int, channels: int, style_dim: int, blocks: int) -> None:
        super().__init__()
        self.conv_in = nn.Conv2d(in_channels, channels, 3, padding=1)
        self.blocks = nn.ModuleList([GatedDWBlock(channels) for _ in range(blocks)])
        self.film = FiLM(style_dim, channels)
        self.unshuffle = nn.PixelUnshuffle(2)
        self.conv_out = nn.Conv2d(channels * 4, 9, 3, padding=1)

    def forward(self, x: torch.Tensor, z_g: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.conv_in(x)
        for block in self.blocks:
            x = block(x)
            x = self.film(x, z_g)
        x = self.unshuffle(x)
        x = self.conv_out(x)
        lh, hl, hh = x.chunk(3, dim=1)
        return lh, hl, hh


class TextureMaskHead(nn.Module):
    def __init__(self, in_channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(in_channels, 1, 3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ThinStructureHead(nn.Module):
    def __init__(
        self,
        in_channels: int,
        channels: int,
        style_dim: int,
        split_struct_alpha: bool = True,
    ) -> None:
        super().__init__()
        alpha_channels = 2 if split_struct_alpha else 1
        self.conv_in = nn.Conv2d(in_channels, channels, 3, padding=1)
        self.blocks = nn.Sequential(GatedDWBlock(channels), GatedDWBlock(channels))
        self.film = FiLM(style_dim, channels)
        self.to_alpha = nn.Conv2d(channels, alpha_channels, 3, padding=1)
        self.to_orient = nn.Conv2d(channels, 8, 3, padding=1)
        self.to_amp = nn.Conv2d(channels, 1, 3, padding=1)

    def forward(self, x: torch.Tensor, z_g: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.conv_in(x)
        x = self.blocks(x)
        x = self.film(x, z_g)
        alpha = torch.sigmoid(self.to_alpha(x))
        orient = torch.softmax(self.to_orient(x), dim=1)
        amp = torch.sigmoid(self.to_amp(x))
        return alpha, orient, amp


class MidBandHead(nn.Module):
    def __init__(self, in_channels: int, channels: int, style_dim: int) -> None:
        super().__init__()
        self.conv_in = nn.Conv2d(in_channels, channels, 3, padding=1)
        self.blocks = nn.Sequential(GatedDWBlock(channels), GatedDWBlock(channels))
        self.film = FiLM(style_dim, channels)
        self.out = nn.Conv2d(channels, 3, 3, padding=1)

    def forward(self, x: torch.Tensor, z_g: torch.Tensor) -> torch.Tensor:
        x = self.conv_in(x)
        x = self.blocks(x)
        x = self.film(x, z_g)
        return self.out(x)


class StructureRefiner2(nn.Module):
    def __init__(
        self,
        in_channels: int,
        feature_channels: int,
        style_dim: int,
        kernels: torch.Tensor,
        split_struct_alpha: bool = True,
        enable_confidence_brake: bool = True,
        conf_brake_power: float = 1.5,
        conf_use_edge_gate: bool = True,
        conf_orient_weight: float = 0.5,
        thin_hair_weight: float = 1.0,
        thin_lash_weight: float = 0.85,
    ) -> None:
        super().__init__()
        self.split_struct_alpha = split_struct_alpha
        self.enable_confidence_brake = enable_confidence_brake
        self.conf_brake_power = conf_brake_power
        self.conf_use_edge_gate = conf_use_edge_gate
        self.conf_orient_weight = conf_orient_weight
        self.thin_hair_weight = thin_hair_weight
        self.thin_lash_weight = thin_lash_weight
        self.thin_head = ThinStructureHead(
            in_channels,
            feature_channels,
            style_dim,
            split_struct_alpha=split_struct_alpha,
        )
        self.mid_head = MidBandHead(in_channels, feature_channels, style_dim)
        self.seed_conv = nn.Conv2d(in_channels, 1, 3, padding=1)
        self.register_buffer("kernels", kernels, persistent=False)

    def _confidence_brake(
        self, alpha_combined: torch.Tensor, orient: torch.Tensor, edge_gate: torch.Tensor
    ) -> torch.Tensor:
        if not self.enable_confidence_brake:
            return torch.ones_like(alpha_combined)
        c_base = alpha_combined.clamp(0.0, 1.0)
        if self.conf_use_edge_gate:
            c_base = c_base * edge_gate
        pmax = orient.max(dim=1, keepdim=True).values
        uniform = 1.0 / orient.shape[1]
        orient_conf = ((pmax - uniform) / (1.0 - uniform)).clamp(0.0, 1.0)
        c_struct = c_base * ((1.0 - self.conf_orient_weight) + self.conf_orient_weight * orient_conf)
        if self.conf_brake_power != 1.0:
            c_struct = c_struct.pow(self.conf_brake_power)
        return c_struct

    def forward(
        self,
        y1: torch.Tensor,
        y_base: torch.Tensor,
        f_hr: torch.Tensor,
        t_up: torch.Tensor,
        z_l: torch.Tensor,
        z_g: torch.Tensor,
        border_mask: torch.Tensor,
        edge_gain: float,
        alpha_thin: float,
        alpha_mid: float,
        mid_band_level: int = 2,
        grad_y1: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if grad_y1 is None:
            grad_y1 = gradient_magnitude(y1)
        edge_gate = torch.clamp(edge_gain * grad_y1, 0.0, 1.0)

        ref_input = torch.cat([y1, y_base, grad_y1, f_hr, t_up, z_l], dim=1)

        alpha_struct, orient, amp = self.thin_head(ref_input, z_g)
        if self.split_struct_alpha:
            alpha_hair = alpha_struct[:, 0:1]
            alpha_lash = alpha_struct[:, 1:2]
            alpha_combined = torch.maximum(alpha_hair, alpha_lash)
            alpha_hair_g = alpha_hair * edge_gate * border_mask
            alpha_lash_g = alpha_lash * edge_gate * border_mask
            alpha_mix = self.thin_hair_weight * alpha_hair_g + self.thin_lash_weight * alpha_lash_g
        else:
            alpha_combined = alpha_struct
            alpha_mix = alpha_struct * edge_gate * border_mask
        amp = amp * alpha_thin

        seed = self.seed_conv(ref_input)
        strokes = apply_oriented_kernels(seed, self.kernels)
        strokes = (strokes * orient).sum(dim=1, keepdim=True)
        delta_thin = alpha_mix * torch.tanh(strokes) * amp

        mid_raw = self.mid_head(ref_input, z_g)
        mid_input = torch.tanh(mid_raw) * alpha_mid
        if mid_band_level == 1:
            mid_coeffs = bandpass_level1(mid_input)
        elif mid_band_level == 2:
            mid_coeffs = bandpass_level2(mid_input)
        else:
            raise ValueError(f"Unsupported mid_band_level {mid_band_level}")
        gate = border_mask * alpha_combined
        if gate.shape[-2:] != mid_coeffs.shape[-2:]:
            gate = downsample_avg(gate)
        delta_mid = mid_coeffs * gate

        c_struct = self._confidence_brake(alpha_combined, orient, edge_gate)
        if c_struct.shape[-2:] != delta_mid.shape[-2:]:
            c_struct_mid = downsample_avg(c_struct)
        else:
            c_struct_mid = c_struct
        delta_thin = delta_thin * c_struct
        delta_mid = delta_mid * c_struct_mid

        return delta_thin, delta_mid, alpha_struct, alpha_combined


def assemble_wavelet_highbands(lh: torch.Tensor, hl: torch.Tensor, hh: torch.Tensor) -> torch.Tensor:
    return haar_from_highbands(lh, hl, hh)


def mid_band_integration(delta_mid: torch.Tensor, output_size: Tuple[int, int]) -> torch.Tensor:
    if delta_mid.shape[-2:] == output_size:
        return delta_mid
    return upsample_band_to_full(delta_mid, output_size)


def bandpass_from_rgb(x: torch.Tensor) -> torch.Tensor:
    return bandpass_level2(x)


def apply_high_pass(x: torch.Tensor) -> torch.Tensor:
    return high_pass(x)


class HumTrunk(nn.Module):
    def __init__(
        self,
        in_channels: int,
        base_channels: int,
        blocks: tuple[int, int, int],
        refine_blocks: int,
    ) -> None:
        super().__init__()
        c1 = base_channels
        c2 = base_channels * 2
        c3 = base_channels * 4
        c4 = base_channels * 6

        self.stem = nn.Conv2d(in_channels, c1, 3, padding=1)

        self.down1 = nn.Conv2d(c1, c2, 4, stride=2, padding=1)
        self.down2 = nn.Conv2d(c2, c3, 4, stride=2, padding=1)
        self.down3 = nn.Conv2d(c3, c4, 4, stride=2, padding=1)

        self.stage1 = nn.Sequential(*[GatedDWBlock(c2) for _ in range(blocks[0])])
        self.stage2 = nn.Sequential(*[GatedDWBlock(c3) for _ in range(blocks[1])])
        self.stage3 = nn.Sequential(*[GatedDWBlock(c4, kernel_size=7) for _ in range(blocks[2])])

        self.up3 = nn.Conv2d(c4, c3, 3, padding=1)
        self.up2 = nn.Conv2d(c3, c2, 3, padding=1)
        self.up1 = nn.Conv2d(c2, c1, 3, padding=1)

        self.refine = nn.Sequential(*[GatedDWBlock(c1) for _ in range(refine_blocks)])

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        f1 = self.stem(x)

        x2 = self.down1(f1)
        f2 = self.stage1(x2)

        x3 = self.down2(f2)
        f3 = self.stage2(x3)

        x4 = self.down3(f3)
        f4 = self.stage3(x4)

        u3 = F.interpolate(f4, scale_factor=2, mode="nearest")
        u3 = self.up3(u3)
        u3 = u3 + f3

        u2 = F.interpolate(u3, scale_factor=2, mode="nearest")
        u2 = self.up2(u2)
        u2 = u2 + f2

        u1 = F.interpolate(u2, scale_factor=2, mode="nearest")
        u1 = self.up1(u1)
        u1 = u1 + f1

        refined = self.refine(u1)

        return {
            "stem": f1,
            "s2": f2,
            "s4": f3,
            "s8": f4,
            "refined": refined,
        }


@dataclass
class ForwardCache:
    y_base: torch.Tensor
    y1: torch.Tensor
    f_hr: torch.Tensor
    t_up: torch.Tensor
    grad: torch.Tensor
    mask_tex: torch.Tensor


class HumArchGenerator(nn.Module):
    def __init__(self, config: HumArchConfig) -> None:
        super().__init__()
        self.config = config
        c = config.base_channels
        self.style_encoder = StyleEncoder(3, c // 2, config.style_dim)
        self.trunk = HumTrunk(3, c, config.trunk_blocks, config.trunk_refine_blocks)
        self.base_head = BaseHead(c)
        self.pretex = PreTex(c, config.pretex_blocks)
        hrtex_in_channels = 3 + 1 + c + c + 1
        self.hrtex = HRTextureHead(hrtex_in_channels, c, config.style_dim, config.hrtex_blocks)
        mask_channels = c + 1
        self.mask_head = PreTex(mask_channels, 2)
        self.mask_out = nn.Sequential(
            nn.Conv2d(mask_channels, 1, 3, padding=1),
            nn.Sigmoid(),
        )
        kernels = build_oriented_kernels(config.line_kernel_size, torch.device("cpu"))
        self.refiner2 = StructureRefiner2(
            in_channels=3 + 3 + 1 + c + c + 1,
            feature_channels=c,
            style_dim=config.style_dim,
            kernels=kernels,
            split_struct_alpha=config.split_struct_alpha,
            enable_confidence_brake=config.enable_confidence_brake,
            conf_brake_power=config.conf_brake_power,
            conf_use_edge_gate=config.conf_use_edge_gate,
            conf_orient_weight=config.conf_orient_weight,
            thin_hair_weight=config.thin_hair_weight,
            thin_lash_weight=config.thin_lash_weight,
        )

    def compute_style(self, x: torch.Tensor, max_side: int = 320) -> torch.Tensor:
        b, _, h, w = x.shape
        scale = max_side / max(h, w)
        if scale < 1.0:
            x = F.interpolate(x, scale_factor=scale, mode="bilinear", align_corners=False)
        return self.style_encoder(x)

    def _upsample_feature(self, feat: torch.Tensor, scale: int) -> torch.Tensor:
        if scale == 1:
            return feat
        return upsample_nearest(feat, scale)

    def _prepare_border_mask(self, y: torch.Tensor, border_mask: Optional[torch.Tensor]) -> torch.Tensor:
        if border_mask is not None:
            return border_mask
        return make_border_mask(y.shape[-2], y.shape[-1], y.device)

    def forward(
        self,
        x: torch.Tensor,
        scale: int,
        z_g: Optional[torch.Tensor] = None,
        z_l: Optional[torch.Tensor] = None,
        border_mask: Optional[torch.Tensor] = None,
        return_debug: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if scale not in (1, 2, 4):
            raise ValueError("scale must be 1, 2, or 4")
        if z_g is None:
            z_g = self.compute_style(x)
        trunk_feats = self.trunk(x)
        refined = trunk_feats["refined"]

        y_base = self.base_head(refined, scale)
        f_hr = self._upsample_feature(refined, scale)

        t = self.pretex(refined)
        t_up = self._upsample_feature(t, scale)

        grad_base = gradient_magnitude(y_base)

        if z_l is None:
            z_l = torch.zeros_like(grad_base)

        pad_h = y_base.shape[-2] % 2
        pad_w = y_base.shape[-1] % 2
        if pad_h or pad_w:
            pad = (0, pad_w, 0, pad_h)
            y_base = F.pad(y_base, pad, mode="reflect")
            grad_base = F.pad(grad_base, pad, mode="reflect")
            f_hr = F.pad(f_hr, pad, mode="reflect")
            t_up = F.pad(t_up, pad, mode="reflect")
            z_l = F.pad(z_l, pad, mode="reflect")
            if border_mask is not None:
                border_mask = F.pad(border_mask, pad, mode="reflect")

        hr_input = torch.cat([y_base, grad_base, f_hr, t_up, z_l], dim=1)
        lh, hl, hh = self.hrtex(hr_input, z_g)

        border_mask = self._prepare_border_mask(y_base, border_mask)
        mask_in = torch.cat([f_hr, grad_base], dim=1)
        mask_feat = self.mask_head(mask_in)
        mask_tex = self.mask_out(mask_feat)
        mask_tex_half = F.avg_pool2d(mask_tex, 2)
        border_half = make_border_mask(
            y_base.shape[-2] // 2,
            y_base.shape[-1] // 2,
            y_base.device,
        )
        lh = lh * mask_tex_half * border_half
        hl = hl * mask_tex_half * border_half
        hh = hh * mask_tex_half * border_half

        delta_hf = assemble_wavelet_highbands(lh, hl, hh)
        y1 = y_base + delta_hf
        if pad_h or pad_w:
            h_slice = slice(None, -pad_h) if pad_h else slice(None)
            w_slice = slice(None, -pad_w) if pad_w else slice(None)
            y_base = y_base[..., h_slice, w_slice]
            y1 = y1[..., h_slice, w_slice]
            f_hr = f_hr[..., h_slice, w_slice]
            t_up = t_up[..., h_slice, w_slice]
            grad_base = grad_base[..., h_slice, w_slice]
            border_mask = border_mask[..., h_slice, w_slice]
            z_l = z_l[..., h_slice, w_slice]

        cache = ForwardCache(
            y_base=y_base,
            y1=y1,
            f_hr=f_hr,
            t_up=t_up,
            grad=gradient_magnitude(y1),
            mask_tex=mask_tex,
        )

        y2, debug = self._forward_pass2(
            cache,
            z_g,
            z_l,
            border_mask,
            scale,
            return_debug=return_debug,
        )
        if return_debug:
            debug.update({"y_base": y_base, "y1": y1})
            return y2, debug
        return y2

    def _forward_pass2(
        self,
        cache: ForwardCache,
        z_g: torch.Tensor,
        z_l: torch.Tensor,
        border_mask: torch.Tensor,
        scale: int,
        return_debug: bool,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        alpha_thin, alpha_mid = self.config.alpha_for_scale(scale)
        debug: Dict[str, torch.Tensor] = {}
        if scale == 4:
            y1 = F.avg_pool2d(cache.y1, 2)
            y_base = F.avg_pool2d(cache.y_base, 2)
            f_hr = F.avg_pool2d(cache.f_hr, 2)
            t_up = F.avg_pool2d(cache.t_up, 2)
            z_l = F.avg_pool2d(z_l, 2)
            border = make_border_mask(y1.shape[-2], y1.shape[-1], y1.device)
            grad_y1 = gradient_magnitude(y1)
        else:
            y1 = cache.y1
            y_base = cache.y_base
            f_hr = cache.f_hr
            t_up = cache.t_up
            border = border_mask
            grad_y1 = cache.grad

        delta_thin, delta_mid, alpha_struct, alpha_combined = self.refiner2(
            y1,
            y_base,
            f_hr,
            t_up,
            z_l,
            z_g,
            border,
            self.config.edge_gain,
            alpha_thin,
            alpha_mid,
            mid_band_level=1 if scale == 4 else 2,
            grad_y1=grad_y1,
        )

        if scale == 4:
            delta_thin = F.interpolate(delta_thin, scale_factor=2, mode="bilinear", align_corners=False)
            delta_thin = apply_high_pass(delta_thin) * border_mask
            delta_mid = mid_band_integration(delta_mid, cache.y1.shape[-2:]) * border_mask
        else:
            delta_mid = mid_band_integration(delta_mid, cache.y1.shape[-2:])

        y2 = cache.y1 + delta_thin + delta_mid

        if return_debug:
            debug.update(
                {
                    "delta_thin": delta_thin,
                    "delta_mid": delta_mid,
                    "alpha_struct": alpha_struct,
                    "alpha_struct_combined": alpha_combined,
                }
            )
        return y2, debug


class HumArch(nn.Module):
    """Human anatomy restoration/SR generator with two-pass detail synthesis."""

    def __init__(
        self,
        base_channels: int = 48,
        style_dim: int = 128,
        noise_stride: int = 16,
        pretex_blocks: int = 8,
        trunk_blocks: tuple[int, int, int] = (2, 2, 3),
        trunk_refine_blocks: int = 6,
        hrtex_blocks: int = 6,
        thin_alpha: tuple[float, float, float] = (0.08, 0.1, 0.12),
        mid_alpha: tuple[float, float, float] = (0.05, 0.07, 0.08),
        edge_gain: float = 3.0,
        line_kernel_size: int = 7,
    ) -> None:
        super().__init__()
        config = HumArchConfig(
            base_channels=base_channels,
            style_dim=style_dim,
            noise_stride=noise_stride,
            pretex_blocks=pretex_blocks,
            trunk_blocks=trunk_blocks,
            trunk_refine_blocks=trunk_refine_blocks,
            hrtex_blocks=hrtex_blocks,
            thin_alpha=thin_alpha,
            mid_alpha=mid_alpha,
            edge_gain=edge_gain,
            line_kernel_size=line_kernel_size,
        )
        self.generator = HumArchGenerator(config)

    @property
    def config(self) -> HumArchConfig:
        return self.generator.config

    def compute_style(self, image: torch.Tensor, max_side: int = 320) -> torch.Tensor:
        return self.generator.compute_style(image, max_side=max_side)

    def forward(
        self,
        image: torch.Tensor,
        scale: int,
        z_g: Optional[torch.Tensor] = None,
        z_l: Optional[torch.Tensor] = None,
        border_mask: Optional[torch.Tensor] = None,
        return_debug: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        return self.generator(
            image,
            scale=scale,
            z_g=z_g,
            z_l=z_l,
            border_mask=border_mask,
            return_debug=return_debug,
        )


def make_global_noise_map(
    output_shape: Tuple[int, int],
    stride: int,
    seed: int,
    device: torch.device,
) -> torch.Tensor:
    h, w = output_shape
    h_lr = math.ceil(h / stride)
    w_lr = math.ceil(w / stride)
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    return torch.randn(1, 1, h_lr, w_lr, generator=gen, device=device)


def crop_and_upsample_noise(
    noise_map: torch.Tensor,
    out_y0: int,
    out_x0: int,
    out_h: int,
    out_w: int,
    stride: int,
) -> torch.Tensor:
    y0 = out_y0 // stride
    x0 = out_x0 // stride
    y1 = math.ceil((out_y0 + out_h) / stride)
    x1 = math.ceil((out_x0 + out_w) / stride)
    crop = noise_map[..., y0:y1, x0:x1]
    return F.interpolate(crop, size=(out_h, out_w), mode="bilinear", align_corners=False)


def tiled_inference(
    model: HumArchGenerator,
    image: torch.Tensor,
    scale: int,
    seed: int = 42,
    tile_size: int = 192,
    overlap: int = 32,
    style_long_side: int = 320,
    config: HumArchConfig | None = None,
) -> torch.Tensor:
    if scale not in (1, 2, 4):
        raise ValueError("scale must be 1, 2, or 4")
    if image.dim() != 4:
        raise ValueError("image must be BCHW")
    if overlap >= tile_size:
        raise ValueError("overlap must be smaller than tile_size")

    b, _, h, w = image.shape
    device = image.device
    if config is None:
        config = model.config

    out_h = h * scale
    out_w = w * scale

    z_g = model.compute_style(image, max_side=style_long_side)
    noise_full = make_global_noise_map((out_h, out_w), config.noise_stride, seed, device)

    acc = torch.zeros(b, 3, out_h, out_w, device=device)
    wacc = torch.zeros(b, 1, out_h, out_w, device=device)

    stride = tile_size - overlap
    for y in range(0, h, stride):
        for x in range(0, w, stride):
            tile = image[..., y : y + tile_size, x : x + tile_size]
            tile_h, tile_w = tile.shape[-2:]
            out_y0 = y * scale
            out_x0 = x * scale
            out_h_tile = tile_h * scale
            out_w_tile = tile_w * scale

            z_l = crop_and_upsample_noise(
                noise_full, out_y0, out_x0, out_h_tile, out_w_tile, config.noise_stride
            )
            border_mask = make_border_mask(out_h_tile, out_w_tile, device)
            y_tile = model(tile, scale=scale, z_g=z_g, z_l=z_l, border_mask=border_mask)

            acc[..., out_y0 : out_y0 + out_h_tile, out_x0 : out_x0 + out_w_tile] += (
                y_tile * border_mask
            )
            wacc[..., out_y0 : out_y0 + out_h_tile, out_x0 : out_x0 + out_w_tile] += border_mask

    out = acc / wacc.clamp_min(1e-6)
    return out

# TODO: what does it block __all__ means, should it be removed ?
__all__ = [
    "HumArch",
    "HumArchConfig",
    "HumArchGenerator",
    "StructureRefiner2",
    "bandpass_level1",
    "bandpass_level2",
    "build_oriented_kernels",
    "tiled_inference",
    "upsample_band_to_full",
]
