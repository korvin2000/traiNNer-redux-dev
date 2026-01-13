# LLM-Optimized Best Practices for Image Restoration & SISR

> **Purpose**: Provide a machine-readable, LLM-friendly guide for building modern SISR / image-restoration architectures.
> The document is derived from this repository's reference implementations and is tuned to guide future code generation.

---

## 0. Document Metadata (Machine-Readable)

```yaml
doc_id: llm_sisr_best_practices_v1
target_domain: [SISR, image_restoration, low_vision]
primary_language: python
framework: pytorch
minimum_expected_line_count: 2048
line_budget_max: 10000
reasoning_profile: high
intended_reader: LLM
source_priority:
  - README.md (found)
  - overview.md (missing)
  - ideas/idea-05.md (missing)
  - ideas/idea-08.md (missing)
source_arches_focus:
  - traiNNer/archs/realplksr_arch.py
  - traiNNer/archs/plksr_arch.py
  - traiNNer/archs/gfisrv2_arch.py
  - traiNNer/archs/gaterv3_arch.py
  - traiNNer/archs/spanpp_arch.py
  - traiNNer/archs/seemore_arch.py
  - traiNNer/archs/paragonsr2_arch.py
  - traiNNer/archs/escreal_arch.py
  - traiNNer/archs/hit_srf_arch.py
  - traiNNer/archs/mosrv2_arch.py
  - traiNNer/archs/moesr_arch.py
notes:
  - Missing files are explicitly acknowledged and treated as constraints.
  - The guide prioritizes robustness, clarity, and inference efficiency.
```

---

## 1. Source Availability & Assumptions

- **Found**: `README.md` provides a global overview and registry taxonomy.
- **Missing**: `/overview.md`, `/ideas/idea-05.md`, `/ideas/idea-08.md` were not present in the repository path.
- **Assumption**: The requested ideas describe modern SISR / restoration architectures (likely hybrid conv + attention + efficient upsamplers).
- **Action**: This guide is based on the available architecture implementations and highlights how to map their patterns into new designs.

---

## 2. Evidence Map (Architectures & Patterns)

| Architecture | Key Patterns | Notes |
|---|---|---|
| RealPLKSR | Partial large-kernel conv, mixed norm, DySample or PixelShuffle | Efficient large-kernel CNNs with optional dynamic upsampling |
| PLKSR | Registry wrapper for spandrel PLKSR | Useful for parameter choices and options |
| GFISRv2 | Dynamic sampling (DySample), LDA attention, channels_last LayerNorm | Modern SR with dynamic sampling and attention |
| GateRV3 | Re-parameterizable conv blocks, attention, SDPA | Train-time multi-branch, inference-time fused |
| SPANPP | Multi-branch reparameterization, fused conv | Lightweight SR with structural reparam |
| SeemoRe | MoE-like routing (from spandrel), global kernel | Expert routing and recursive usage |
| ParagonSR2 | Classical base + learned residual, stable norm, window attention | Deployment-first design |
| ESCReal | Attention variants (naive/SDPA/flex), LK + dynamic conv | Attention selection for OS and hardware |
| HiT-SRF | Window attention utilities, conv-FFN | Transformer-friendly SR |
| MoSRv2 | Mamba-like gated CNN, inception depthwise conv | Efficient gated CNN core |
| MoESR2 | MSG down/upscale w gated blocks | Multi-stage gating and rescale path |

---

## 3. Evidence Snippets (Selected, Annotated)

> The following snippets are **directly extracted** from the repository and annotated for LLM guidance.

### RealPLKSR: LayerNorm + PLK Block

```python
class LayerNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        return self.weight[:, None, None] * x + self.bias[:, None, None]


class DCCM(nn.Sequential):
    "Doubled Convolutional Channel Mixer"

    def __init__(self, dim: int) -> None:
        super().__init__(
            nn.Conv2d(dim, dim * 2, 3, 1, 1),
            nn.Mish(),
            nn.Conv2d(dim * 2, dim, 3, 1, 1),
        )
        assert isinstance(self[-1].weight, Tensor)
        trunc_normal_(self[-1].weight, std=0.02)


class PLKConv2d(nn.Module):
    "Partial Large Kernel Convolutional Layer"

    def __init__(self, dim: int, kernel_size: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(dim, dim, kernel_size, 1, kernel_size // 2)
        trunc_normal_(self.conv.weight, std=0.02)
        self.idx = dim

    def forward(self, x: Tensor) -> Tensor:
        if self.training:
            x1, x2 = torch.split(x, [self.idx, x.size(1) - self.idx], dim=1)
            x1 = self.conv(x1)
            return torch.cat([x1, x2], dim=1)
        x[:, : self.idx] = self.conv(x[:, : self.idx])
        return x


class EA(nn.Module):
    "Element-wise Attention"

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.f = nn.Sequential(nn.Conv2d(dim, dim, 3, 1, 1), nn.Sigmoid())
        assert isinstance(self.f[0].weight, Tensor)
        trunc_normal_(self.f[0].weight, std=0.02)

    def forward(self, x: Tensor) -> Tensor:
        return x * self.f(x)


class PLKBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        kernel_size: int,
        split_ratio: float,
        use_ea: bool = True,
        norm_groups: int = 4,
        use_layer_norm: bool = False,
    ) -> None:
        super().__init__()

        # Layer Norm
        self.layer_norm = LayerNorm(dim) if use_layer_norm else nn.Identity()

        # Local Texture
        self.channel_mixer = DCCM(dim)

        # Long-range Dependency
        pdim = int(dim * split_ratio)

        # Conv Layer
        self.lk = PLKConv2d(pdim, kernel_size)

        # Instance-dependent modulation
        if use_ea:
            self.attn = EA(dim)
        else:
            self.attn = nn.Identity()

        # Refinement
        self.refine = nn.Conv2d(dim, dim, 1, 1, 0)
        trunc_normal_(self.refine.weight, std=0.02)
        if not use_layer_norm:
            self.norm = nn.GroupNorm(norm_groups, dim)
            nn.init.constant_(self.norm.bias, 0)
            nn.init.constant_(self.norm.weight, 1.0)
        else:
            self.norm = nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        x_skip = x
        x = self.layer_norm(x)
        x = self.channel_mixer(x)
        x = self.lk(x)
        x = self.attn(x)
        x = self.refine(x)
        x = self.norm(x)

        return x + x_skip


```

### GFISRv2: DySample (dynamic resampling)

```python
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
        self.offset = nn.Conv2d(in_channels, out_channels, 1)
        self.scope = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        if self.training:
            nn.init.trunc_normal_(self.offset.weight, std=0.02)
            nn.init.constant_(self.scope.weight, val=0)

        self.register_buffer("init_pos", self._init_pos())

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
        B, _, H, W = offset.shape
        offset = offset.view(B, 2, -1, H, W)
        coords_h = torch.arange(H) + 0.5
        coords_w = torch.arange(W) + 0.5

        coords = (
            torch.stack(torch.meshgrid([coords_w, coords_h], indexing="ij"))
            .transpose(1, 2)
            .unsqueeze(1)
            .unsqueeze(0)
            .type(x.dtype)
            .to(x.device, non_blocking=True)
        )
        normalizer = torch.tensor(
            [W, H], dtype=x.dtype, device=x.device, pin_memory=True
        ).view(1, 2, 1, 1, 1)
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


```

### GateRV3: Re-parameterizable Conv3XC

```python
class Conv3XC(nn.Module):
    def __init__(
        self, c_in: int, c_out: int, gain1: int = 1, s: int = 1, bias: bool = True
    ) -> None:
        super().__init__()
        self.bias = bias
        self.weight_concat = None
        self.bias_concat = None
        self.update_params_flag = False
        self.stride = s
        gain = gain1

        self.sk = nn.Conv2d(
            in_channels=c_in,
            out_channels=c_out,
            kernel_size=1,
            padding=0,
            stride=s,
            bias=bias,
        )
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels=c_in,
                out_channels=c_in * gain,
                kernel_size=1,
                padding=0,
                bias=bias,
            ),
            nn.Conv2d(
                in_channels=c_in * gain,
                out_channels=c_out * gain,
                kernel_size=3,
                stride=s,
                padding=0,
                bias=bias,
            ),
            nn.Conv2d(
                in_channels=c_out * gain,
                out_channels=c_out,
                kernel_size=1,
                padding=0,
                bias=bias,
            ),
        )

        self.eval_conv = nn.Conv2d(
            in_channels=c_in,
            out_channels=c_out,
            kernel_size=3,
            padding=1,
            stride=s,
            bias=bias,
        )
        nn.init.trunc_normal_(self.sk.weight, std=0.02)
        if self.training is False:
            self.eval_conv.weight.requires_grad = False
            self.eval_conv.bias.requires_grad = False  # pyright: ignore[reportOptionalMemberAccess]
            self.update_params()

    def update_params(self) -> None:
        w1 = self.conv[0].weight.data.clone().detach()  # pyright: ignore[reportCallIssue]
        w2 = self.conv[1].weight.data.clone().detach()  # pyright: ignore[reportCallIssue]
        w3 = self.conv[2].weight.data.clone().detach()  # pyright: ignore[reportCallIssue]
        w = (
            F.conv2d(w1.flip(2, 3).permute(1, 0, 2, 3), w2, padding=2, stride=1)
            .flip(2, 3)
            .permute(1, 0, 2, 3)
        )

        self.weight_concat = (
            F.conv2d(w.flip(2, 3).permute(1, 0, 2, 3), w3, padding=0, stride=1)
            .flip(2, 3)
            .permute(1, 0, 2, 3)
        )

        sk_w = self.sk.weight.data.clone().detach()

        if self.bias:
            b1 = self.conv[0].bias.data.clone().detach()  # pyright: ignore[reportCallIssue]
            b2 = self.conv[1].bias.data.clone().detach()  # pyright: ignore[reportCallIssue]
            b3 = self.conv[2].bias.data.clone().detach()  # pyright: ignore[reportCallIssue]
            b = (w2 * b1.reshape(1, -1, 1, 1)).sum((1, 2, 3)) + b2
            self.bias_concat = (w3 * b.reshape(1, -1, 1, 1)).sum((1, 2, 3)) + b3
            sk_b = self.sk.bias.data.clone().detach()  # pyright: ignore[reportOptionalMemberAccess]

        target_kernel_size = 3

        h_pixels_to_pad = (target_kernel_size - 1) // 2
        w_pixels_to_pad = (target_kernel_size - 1) // 2
        sk_w = F.pad(
            sk_w, [h_pixels_to_pad, h_pixels_to_pad, w_pixels_to_pad, w_pixels_to_pad]
        )
        self.weight_concat = self.weight_concat + sk_w
        self.eval_conv.weight.data = self.weight_concat
        if self.bias:
            self.bias_concat = self.bias_concat + sk_b  # pyright: ignore[reportOperatorIssue,reportPossiblyUnboundVariable]
            self.eval_conv.bias.data = self.bias_concat  # pyright: ignore[reportOptionalMemberAccess]

    def train(self, mode: bool = True) -> Self:
        super().train(mode)
        if not mode:
            self.update_params()
        return self

    def forward(self, x: Tensor) -> Tensor:
        if self.training:
            pad = 1
            x_pad = F.pad(x, (pad, pad, pad, pad), "constant", 0)
            out = self.conv(x_pad) + self.sk(x)
        else:
            out = self.eval_conv(x)
        return out


```

### SPANPP: RepConv fusion mechanics

```python
class RepConv(nn.Module):
    def __init__(self, in_dim=3, out_dim=32) -> None:
        super().__init__()
        self.conv1 = SeqConv3x3(in_dim, out_dim, 2)
        self.conv2 = nn.Conv2d(in_dim, out_dim, 3, 1, 1)
        self.conv3 = Conv3XC(in_dim, out_dim)
        self.conv_3x3_rep = nn.Conv2d(in_dim, out_dim, 3, 1, 1)
        self.alpha = nn.Parameter(torch.randn(3), requires_grad=True)
        # self.alpha.register_hook(lambda grad: grad * 100)
        self.forward_module = self.train_forward

        nn.init.constant_(self.alpha, 1.0)

    def fuse(self) -> None:
        conv1_w, conv1_b = self.conv1.rep_params()
        conv2_w, conv2_b = self.conv2.weight, self.conv2.bias
        self.conv3.update_params()
        conv3_w, conv3_b = self.conv3.eval_conv.weight, self.conv3.eval_conv.bias
        device = self.conv_3x3_rep.weight.device
        sum_weight = (
            self.alpha[0] * conv1_w + self.alpha[1] * conv2_w + self.alpha[2] * conv3_w
        ).to(device)
        sum_bias = (
            self.alpha[0] * conv1_b + self.alpha[1] * conv2_b + self.alpha[2] * conv3_b
        ).to(device)
        self.conv_3x3_rep.weight = nn.Parameter(sum_weight)
        self.conv_3x3_rep.bias = nn.Parameter(sum_bias)

    def train_forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)
        return self.alpha[0] * x1 + self.alpha[1] * x2 + self.alpha[2] * x3

    def train(self, mode: bool = True):
        super().train(mode)
        if not mode:
            self.fuse()
        return self

    def forward(self, x):
        if self.training:
            return self.train_forward(x)
        else:
            return self.conv_3x3_rep(x)


```

### ParagonSR2: Magic kernel upsampler + RMSNorm

```python
class MagicKernelSharp2021Upsample(nn.Module):
    """
    Fixed classical upsampler based on Magic Kernel Sharp 2021.

    Provides a strong low-frequency base reconstruction that the neural
    network refines with learned high-frequency detail.
    """

    def __init__(self, in_ch: int, scale: int, alpha: float) -> None:
        super().__init__()
        self.scale = scale
        self.alpha = alpha

        self.sharp = SeparableConv(in_ch, get_magic_sharp_2021_kernel_weights())
        self.blur = SeparableConv(in_ch, get_magic_kernel_weights())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Optional pre-sharpening
        if self.alpha > 0:
            x = x + self.alpha * (self.sharp(x) - x)

        # Nearest-neighbor upsampling
        x = F.interpolate(x, scale_factor=self.scale, mode="nearest")

        # Reconstruction blur
        return self.blur(x)


# ============================================================================
# 2. NORMALIZATION & RESIDUAL SCALING
# ============================================================================


class RMSNorm(nn.Module):
    """
    Spatial RMS Normalization.

    More stable than BatchNorm for SR and safe in FP16.
    Computes variance in FP32 for numerical stability.
    """

    def __init__(self, channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.ones(channels, 1, 1))
        self.bias = nn.Parameter(torch.zeros(channels, 1, 1))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Calculate variance in FP32 for stability, then cast back
        x_f32 = x.float()
        variance = x_f32.pow(2).mean(dim=1, keepdim=True)
        rms = torch.sqrt(variance + self.eps).to(x.dtype)
        return self.scale * x / rms + self.bias


class LayerScale(nn.Module):
    """
    LayerScale: Learnable per-channel scaling factor initialized to a small value.
    Crucial for stabilizing deep networks, especially in FP16.
    """

    def __init__(self, dim: int, init_values: float = 1e-5) -> None:
        super().__init__()
        self.gamma = nn.Parameter(init_values * torch.ones(1, dim, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.gamma


# ============================================================================
# 3. CORE BLOCKS
# ============================================================================


```

### ESCReal: ConvolutionalAttention (LK + dynamic)

```python
class ConvolutionalAttention(nn.Module):
    def __init__(self, pdim: int) -> None:
        super().__init__()
        self.pdim = pdim
        self.sk_size = 3
        self.dwc_proj = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(pdim, pdim // 2, 1, 1, 0),
            nn.GELU(),
            nn.Conv2d(pdim // 2, pdim * self.sk_size * self.sk_size, 1, 1, 0),
        )
        nn.init.zeros_(self.dwc_proj[-1].weight)  # pyright: ignore[reportArgumentType]
        nn.init.zeros_(self.dwc_proj[-1].bias)  # pyright: ignore[reportArgumentType]

    def forward(self, x: Tensor, lk_filter: Tensor) -> Tensor:
        if self.training:
            x1, x2 = torch.split(x, [self.pdim, x.shape[1] - self.pdim], dim=1)

            # Dynamic Conv
            bs = x1.shape[0]
            dynamic_kernel = self.dwc_proj(x[:, : self.pdim]).reshape(
                -1, 1, self.sk_size, self.sk_size
            )
            x1_ = rearrange(x1, "b c h w -> 1 (b c) h w")
            x1_ = F.conv2d(
                x1_,
                dynamic_kernel,
                stride=1,
                padding=self.sk_size // 2,
                groups=bs * self.pdim,
            )
            x1_ = rearrange(x1_, "1 (b c) h w -> b c h w", b=bs, c=self.pdim)

            # Static LK Conv + Dynamic Conv
            x1 = (
                F.conv2d(x1, lk_filter, stride=1, padding=lk_filter.shape[-1] // 2)
                + x1_
            )

            x = torch.cat([x1, x2], dim=1)
        else:
            # for GPU
            dynamic_kernel = self.dwc_proj(x[:, : self.pdim]).reshape(
                -1, 1, self.sk_size, self.sk_size
            )
            x[:, : self.pdim] = F.conv2d(
                x[:, : self.pdim], lk_filter, stride=1, padding=13 // 2
            ) + F.conv2d(
                x[:, : self.pdim],
                dynamic_kernel,
                stride=1,
                padding=self.sk_size // 2,
                groups=self.pdim,
            )

            # For Mobile Conversion, uncomment the following code
            # x_1, x_2 = torch.split(x, [self.pdim, x.shape[1]-self.pdim], dim=1)
            # dynamic_kernel = self.dwc_proj(x_1).reshape(16, 1, 3, 3)
            # x_1 = F.conv2d(x_1, lk_filter, stride=1, padding=13 // 2) + F.conv2d(x_1, dynamic_kernel, stride=1, padding=1, groups=16)
            # x = torch.cat([x_1, x_2], dim=1)
        return x

    def extra_repr(self) -> str:
        return f"pdim={self.pdim}"


```

### HiT-SRF: Window partition utils + ConvFFN

```python
def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (tuple): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(
        B, H // window_size[0], window_size[0], W // window_size[1], window_size[1], C
    )
    windows = (
        x.permute(0, 1, 3, 2, 4, 5)
        .contiguous()
        .view(-1, window_size[0], window_size[1], C)
    )
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (tuple): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] * (window_size[0] * window_size[1]) / (H * W))
    x = windows.view(
        B, H // window_size[0], W // window_size[1], window_size[0], window_size[1], -1
    )
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


```

### MoSRv2: GatedCNNBlock + pad-to-multiple

```python
class GatedCNNBlock(nn.Module):
    r"""
    modernized mambaout main unit
    https://github.com/yuweihao/MambaOut/blob/main/models/mambaout.py#L119
    """

    def __init__(
        self, dim: int = 64, expansion_ratio: float = 8 / 3, rms_norm: bool = True
    ) -> None:
        super().__init__()
        self.norm = RMSNorm(dim) if rms_norm else LayerNorm(dim)
        hidden = int(expansion_ratio * dim)
        self.fc1 = nn.Conv2d(dim, hidden * 2, 3, 1, 1)

        self.act = nn.Mish()
        conv_channels = dim
        self.split_indices = [hidden, hidden - conv_channels, conv_channels]

        self.conv = InceptionDWConv2d(conv_channels)
        self.fc2 = nn.Conv2d(hidden, dim, 3, 1, 1)
        self.gamma = nn.Parameter(torch.ones([1, dim, 1, 1]), requires_grad=True)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m: nn.Module) -> None:
        if isinstance(m, nn.Conv2d | nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x: Tensor) -> Tensor:
        shortcut = x
        x = self.norm(x)
        g, i, c = torch.split(self.fc1(x), self.split_indices, dim=1)
        c = self.conv(c)
        x = self.act(self.fc2(self.act(g) * torch.cat((i, c), dim=1)))
        return x * self.gamma + shortcut


@ARCH_REGISTRY.register()
```

### MoESR2: MSG block and residual

```python
class MSG(nn.Module):
    def __init__(self, dim, expansion_msg=1.5) -> None:
        super().__init__()
        self.down = nn.Sequential(
            nn.Conv2d(dim, dim // 4, 3, 1, 1),
            nn.PixelUnshuffle(2),
            nn.LeakyReLU(0.1, True),
        )
        self.gated = nn.Sequential(
            *[GatedCNNBlock(dim, expansion_ratio=expansion_msg) for _ in range(3)]
        )
        self.up = nn.Sequential(
            nn.Conv2d(dim, dim * 4, 3, 1, 1),
            nn.PixelShuffle(2),
            nn.LeakyReLU(0.1, True),
        )

    def forward(self, x):
        out = self.down(x)
        out = self.gated(out)
        return self.up(out) + x


```

---

## 4. Best-Practice SISR/Restoration Architecture Principles

### 4.1 Core Invariants (Non-Negotiable)

- **INV-001**: Input tensor shape is [B, C, H, W] with contiguous memory unless explicitly using channels_last.
- **INV-002**: Preserve spatial alignment: avoid odd padding that shifts content unless explicitly modeling shifts.
- **INV-003**: Upscale factor must be an integer; output dimensions must be (H*scale, W*scale).
- **INV-004**: Residual connections must preserve dtype and device; do not mix CPU/GPU tensors.
- **INV-005**: Normalization layers must be safe for small batch sizes (often batch=1 in SR).
- **INV-006**: Inference path should avoid training-only branches (no dropout, no stochastic kernels).
- **INV-007**: Memory growth must be bounded: avoid O(HW)^2 attention for large inputs without windowing.
- **INV-008**: All parameterized kernels should be initialized for stability (trunc_normal or scaled init).
- **INV-009**: If using dynamic sampling, ensure offsets are normalized to [-1, 1] for grid_sample.
- **INV-010**: If using reparameterization, ensure a deterministic fuse path before deployment.

### 4.2 Critical Failure Modes (Avoid)

- **FAIL-001**: Shape drift after PixelShuffle/Unshuffle due to wrong channel multiples.
- **FAIL-002**: Numerical instability in FP16 without RMSNorm or LayerScale.
- **FAIL-003**: Over-parameterized attention layers causing OOM on 4K inputs.
- **FAIL-004**: Dynamic kernel generation without clamping or normalization leading to exploding activations.
- **FAIL-005**: Hard-coded padding that breaks non-multiple sizes.
- **FAIL-006**: Using BatchNorm in SR tasks causing brightness shifts with batch=1.
- **FAIL-007**: Inconsistent upsamplers between training and inference.
- **FAIL-008**: Missing fusing of reparam blocks for inference leads to slower runtime and mismatched outputs.
- **FAIL-009**: Using align_corners=True where interpolation should be consistent with SR benchmarks.
- **FAIL-010**: Non-deterministic device movement due to implicit CPU tensors in weight buffers.

---

## 5. LLM Rulebook (Concise, Machine-Oriented)

> These rules are distilled from the repository's architectures and are intended for direct use in code generation.

- **RULE-001**
  - context: SISR
  - focus: RMSNorm
  - rule: Prefer RMSNorm or LayerNorm for SR stability; avoid BatchNorm.
  - verify: ensure invariants in Section 4 are satisfied.

- **RULE-002**
  - context: SISR
  - focus: Residual
  - rule: Use residual connections for feature preservation; scale residual if deep.
  - verify: ensure invariants in Section 4 are satisfied.

- **RULE-003**
  - context: SISR
  - focus: Upsample
  - rule: PixelShuffle or DySample are preferred; ensure channel math is correct.
  - verify: ensure invariants in Section 4 are satisfied.

- **RULE-004**
  - context: SISR
  - focus: DynamicSampling
  - rule: If using grid_sample, normalize offsets to [-1, 1].
  - verify: ensure invariants in Section 4 are satisfied.

- **RULE-005**
  - context: SISR
  - focus: Padding
  - rule: Pad to multiples of scale or window size, then crop output.
  - verify: ensure invariants in Section 4 are satisfied.

- **RULE-006**
  - context: SISR
  - focus: Attention
  - rule: Use windowed attention for large images; avoid global attention unless small.
  - verify: ensure invariants in Section 4 are satisfied.

- **RULE-007**
  - context: SISR
  - focus: Reparam
  - rule: Use train-time multi-branch, inference-time fused convs where possible.
  - verify: ensure invariants in Section 4 are satisfied.

- **RULE-008**
  - context: SISR
  - focus: Init
  - rule: Initialize conv weights with trunc_normal_ or similar stable init.
  - verify: ensure invariants in Section 4 are satisfied.

- **RULE-009**
  - context: SISR
  - focus: Activation
  - rule: Mish/SiLU/GELU are commonly stable; keep activations consistent.
  - verify: ensure invariants in Section 4 are satisfied.

- **RULE-010**
  - context: SISR
  - focus: Precision
  - rule: Compute variance in FP32 if using FP16 to avoid overflow.
  - verify: ensure invariants in Section 4 are satisfied.

- **RULE-011**
  - context: Restoration
  - focus: RMSNorm
  - rule: Prefer RMSNorm or LayerNorm for SR stability; avoid BatchNorm.
  - verify: ensure invariants in Section 4 are satisfied.

- **RULE-012**
  - context: Restoration
  - focus: Residual
  - rule: Use residual connections for feature preservation; scale residual if deep.
  - verify: ensure invariants in Section 4 are satisfied.

- **RULE-013**
  - context: Restoration
  - focus: Upsample
  - rule: PixelShuffle or DySample are preferred; ensure channel math is correct.
  - verify: ensure invariants in Section 4 are satisfied.

- **RULE-014**
  - context: Restoration
  - focus: DynamicSampling
  - rule: If using grid_sample, normalize offsets to [-1, 1].
  - verify: ensure invariants in Section 4 are satisfied.

- **RULE-015**
  - context: Restoration
  - focus: Padding
  - rule: Pad to multiples of scale or window size, then crop output.
  - verify: ensure invariants in Section 4 are satisfied.

- **RULE-016**
  - context: Restoration
  - focus: Attention
  - rule: Use windowed attention for large images; avoid global attention unless small.
  - verify: ensure invariants in Section 4 are satisfied.

- **RULE-017**
  - context: Restoration
  - focus: Reparam
  - rule: Use train-time multi-branch, inference-time fused convs where possible.
  - verify: ensure invariants in Section 4 are satisfied.

- **RULE-018**
  - context: Restoration
  - focus: Init
  - rule: Initialize conv weights with trunc_normal_ or similar stable init.
  - verify: ensure invariants in Section 4 are satisfied.

- **RULE-019**
  - context: Restoration
  - focus: Activation
  - rule: Mish/SiLU/GELU are commonly stable; keep activations consistent.
  - verify: ensure invariants in Section 4 are satisfied.

- **RULE-020**
  - context: Restoration
  - focus: Precision
  - rule: Compute variance in FP32 if using FP16 to avoid overflow.
  - verify: ensure invariants in Section 4 are satisfied.

- **RULE-021**
  - context: JPEG
  - focus: RMSNorm
  - rule: Prefer RMSNorm or LayerNorm for SR stability; avoid BatchNorm.
  - verify: ensure invariants in Section 4 are satisfied.

- **RULE-022**
  - context: JPEG
  - focus: Residual
  - rule: Use residual connections for feature preservation; scale residual if deep.
  - verify: ensure invariants in Section 4 are satisfied.

- **RULE-023**
  - context: JPEG
  - focus: Upsample
  - rule: PixelShuffle or DySample are preferred; ensure channel math is correct.
  - verify: ensure invariants in Section 4 are satisfied.

- **RULE-024**
  - context: JPEG
  - focus: DynamicSampling
  - rule: If using grid_sample, normalize offsets to [-1, 1].
  - verify: ensure invariants in Section 4 are satisfied.

- **RULE-025**
  - context: JPEG
  - focus: Padding
  - rule: Pad to multiples of scale or window size, then crop output.
  - verify: ensure invariants in Section 4 are satisfied.

- **RULE-026**
  - context: JPEG
  - focus: Attention
  - rule: Use windowed attention for large images; avoid global attention unless small.
  - verify: ensure invariants in Section 4 are satisfied.

- **RULE-027**
  - context: JPEG
  - focus: Reparam
  - rule: Use train-time multi-branch, inference-time fused convs where possible.
  - verify: ensure invariants in Section 4 are satisfied.

- **RULE-028**
  - context: JPEG
  - focus: Init
  - rule: Initialize conv weights with trunc_normal_ or similar stable init.
  - verify: ensure invariants in Section 4 are satisfied.

- **RULE-029**
  - context: JPEG
  - focus: Activation
  - rule: Mish/SiLU/GELU are commonly stable; keep activations consistent.
  - verify: ensure invariants in Section 4 are satisfied.

- **RULE-030**
  - context: JPEG
  - focus: Precision
  - rule: Compute variance in FP32 if using FP16 to avoid overflow.
  - verify: ensure invariants in Section 4 are satisfied.

- **RULE-031**
  - context: LowLight
  - focus: RMSNorm
  - rule: Prefer RMSNorm or LayerNorm for SR stability; avoid BatchNorm.
  - verify: ensure invariants in Section 4 are satisfied.

- **RULE-032**
  - context: LowLight
  - focus: Residual
  - rule: Use residual connections for feature preservation; scale residual if deep.
  - verify: ensure invariants in Section 4 are satisfied.

- **RULE-033**
  - context: LowLight
  - focus: Upsample
  - rule: PixelShuffle or DySample are preferred; ensure channel math is correct.
  - verify: ensure invariants in Section 4 are satisfied.

- **RULE-034**
  - context: LowLight
  - focus: DynamicSampling
  - rule: If using grid_sample, normalize offsets to [-1, 1].
  - verify: ensure invariants in Section 4 are satisfied.

- **RULE-035**
  - context: LowLight
  - focus: Padding
  - rule: Pad to multiples of scale or window size, then crop output.
  - verify: ensure invariants in Section 4 are satisfied.

- **RULE-036**
  - context: LowLight
  - focus: Attention
  - rule: Use windowed attention for large images; avoid global attention unless small.
  - verify: ensure invariants in Section 4 are satisfied.

- **RULE-037**
  - context: LowLight
  - focus: Reparam
  - rule: Use train-time multi-branch, inference-time fused convs where possible.
  - verify: ensure invariants in Section 4 are satisfied.

- **RULE-038**
  - context: LowLight
  - focus: Init
  - rule: Initialize conv weights with trunc_normal_ or similar stable init.
  - verify: ensure invariants in Section 4 are satisfied.

- **RULE-039**
  - context: LowLight
  - focus: Activation
  - rule: Mish/SiLU/GELU are commonly stable; keep activations consistent.
  - verify: ensure invariants in Section 4 are satisfied.

- **RULE-040**
  - context: LowLight
  - focus: Precision
  - rule: Compute variance in FP32 if using FP16 to avoid overflow.
  - verify: ensure invariants in Section 4 are satisfied.

- **RULE-041**
  - context: Noise
  - focus: RMSNorm
  - rule: Prefer RMSNorm or LayerNorm for SR stability; avoid BatchNorm.
  - verify: ensure invariants in Section 4 are satisfied.

- **RULE-042**
  - context: Noise
  - focus: Residual
  - rule: Use residual connections for feature preservation; scale residual if deep.
  - verify: ensure invariants in Section 4 are satisfied.

- **RULE-043**
  - context: Noise
  - focus: Upsample
  - rule: PixelShuffle or DySample are preferred; ensure channel math is correct.
  - verify: ensure invariants in Section 4 are satisfied.

- **RULE-044**
  - context: Noise
  - focus: DynamicSampling
  - rule: If using grid_sample, normalize offsets to [-1, 1].
  - verify: ensure invariants in Section 4 are satisfied.

- **RULE-045**
  - context: Noise
  - focus: Padding
  - rule: Pad to multiples of scale or window size, then crop output.
  - verify: ensure invariants in Section 4 are satisfied.

- **RULE-046**
  - context: Noise
  - focus: Attention
  - rule: Use windowed attention for large images; avoid global attention unless small.
  - verify: ensure invariants in Section 4 are satisfied.

- **RULE-047**
  - context: Noise
  - focus: Reparam
  - rule: Use train-time multi-branch, inference-time fused convs where possible.
  - verify: ensure invariants in Section 4 are satisfied.

- **RULE-048**
  - context: Noise
  - focus: Init
  - rule: Initialize conv weights with trunc_normal_ or similar stable init.
  - verify: ensure invariants in Section 4 are satisfied.

- **RULE-049**
  - context: Noise
  - focus: Activation
  - rule: Mish/SiLU/GELU are commonly stable; keep activations consistent.
  - verify: ensure invariants in Section 4 are satisfied.

- **RULE-050**
  - context: Noise
  - focus: Precision
  - rule: Compute variance in FP32 if using FP16 to avoid overflow.
  - verify: ensure invariants in Section 4 are satisfied.

- **RULE-051**
  - context: Motion
  - focus: RMSNorm
  - rule: Prefer RMSNorm or LayerNorm for SR stability; avoid BatchNorm.
  - verify: ensure invariants in Section 4 are satisfied.

- **RULE-052**
  - context: Motion
  - focus: Residual
  - rule: Use residual connections for feature preservation; scale residual if deep.
  - verify: ensure invariants in Section 4 are satisfied.

- **RULE-053**
  - context: Motion
  - focus: Upsample
  - rule: PixelShuffle or DySample are preferred; ensure channel math is correct.
  - verify: ensure invariants in Section 4 are satisfied.

- **RULE-054**
  - context: Motion
  - focus: DynamicSampling
  - rule: If using grid_sample, normalize offsets to [-1, 1].
  - verify: ensure invariants in Section 4 are satisfied.

- **RULE-055**
  - context: Motion
  - focus: Padding
  - rule: Pad to multiples of scale or window size, then crop output.
  - verify: ensure invariants in Section 4 are satisfied.

- **RULE-056**
  - context: Motion
  - focus: Attention
  - rule: Use windowed attention for large images; avoid global attention unless small.
  - verify: ensure invariants in Section 4 are satisfied.

- **RULE-057**
  - context: Motion
  - focus: Reparam
  - rule: Use train-time multi-branch, inference-time fused convs where possible.
  - verify: ensure invariants in Section 4 are satisfied.

- **RULE-058**
  - context: Motion
  - focus: Init
  - rule: Initialize conv weights with trunc_normal_ or similar stable init.
  - verify: ensure invariants in Section 4 are satisfied.

- **RULE-059**
  - context: Motion
  - focus: Activation
  - rule: Mish/SiLU/GELU are commonly stable; keep activations consistent.
  - verify: ensure invariants in Section 4 are satisfied.

- **RULE-060**
  - context: Motion
  - focus: Precision
  - rule: Compute variance in FP32 if using FP16 to avoid overflow.
  - verify: ensure invariants in Section 4 are satisfied.

- **RULE-061**
  - context: TinyModel
  - focus: RMSNorm
  - rule: Prefer RMSNorm or LayerNorm for SR stability; avoid BatchNorm.
  - verify: ensure invariants in Section 4 are satisfied.

- **RULE-062**
  - context: TinyModel
  - focus: Residual
  - rule: Use residual connections for feature preservation; scale residual if deep.
  - verify: ensure invariants in Section 4 are satisfied.

- **RULE-063**
  - context: TinyModel
  - focus: Upsample
  - rule: PixelShuffle or DySample are preferred; ensure channel math is correct.
  - verify: ensure invariants in Section 4 are satisfied.

- **RULE-064**
  - context: TinyModel
  - focus: DynamicSampling
  - rule: If using grid_sample, normalize offsets to [-1, 1].
  - verify: ensure invariants in Section 4 are satisfied.

- **RULE-065**
  - context: TinyModel
  - focus: Padding
  - rule: Pad to multiples of scale or window size, then crop output.
  - verify: ensure invariants in Section 4 are satisfied.

- **RULE-066**
  - context: TinyModel
  - focus: Attention
  - rule: Use windowed attention for large images; avoid global attention unless small.
  - verify: ensure invariants in Section 4 are satisfied.

- **RULE-067**
  - context: TinyModel
  - focus: Reparam
  - rule: Use train-time multi-branch, inference-time fused convs where possible.
  - verify: ensure invariants in Section 4 are satisfied.

- **RULE-068**
  - context: TinyModel
  - focus: Init
  - rule: Initialize conv weights with trunc_normal_ or similar stable init.
  - verify: ensure invariants in Section 4 are satisfied.

- **RULE-069**
  - context: TinyModel
  - focus: Activation
  - rule: Mish/SiLU/GELU are commonly stable; keep activations consistent.
  - verify: ensure invariants in Section 4 are satisfied.

- **RULE-070**
  - context: TinyModel
  - focus: Precision
  - rule: Compute variance in FP32 if using FP16 to avoid overflow.
  - verify: ensure invariants in Section 4 are satisfied.

- **RULE-071**
  - context: Realtime
  - focus: RMSNorm
  - rule: Prefer RMSNorm or LayerNorm for SR stability; avoid BatchNorm.
  - verify: ensure invariants in Section 4 are satisfied.

- **RULE-072**
  - context: Realtime
  - focus: Residual
  - rule: Use residual connections for feature preservation; scale residual if deep.
  - verify: ensure invariants in Section 4 are satisfied.

- **RULE-073**
  - context: Realtime
  - focus: Upsample
  - rule: PixelShuffle or DySample are preferred; ensure channel math is correct.
  - verify: ensure invariants in Section 4 are satisfied.

- **RULE-074**
  - context: Realtime
  - focus: DynamicSampling
  - rule: If using grid_sample, normalize offsets to [-1, 1].
  - verify: ensure invariants in Section 4 are satisfied.

- **RULE-075**
  - context: Realtime
  - focus: Padding
  - rule: Pad to multiples of scale or window size, then crop output.
  - verify: ensure invariants in Section 4 are satisfied.

- **RULE-076**
  - context: Realtime
  - focus: Attention
  - rule: Use windowed attention for large images; avoid global attention unless small.
  - verify: ensure invariants in Section 4 are satisfied.

- **RULE-077**
  - context: Realtime
  - focus: Reparam
  - rule: Use train-time multi-branch, inference-time fused convs where possible.
  - verify: ensure invariants in Section 4 are satisfied.

- **RULE-078**
  - context: Realtime
  - focus: Init
  - rule: Initialize conv weights with trunc_normal_ or similar stable init.
  - verify: ensure invariants in Section 4 are satisfied.

- **RULE-079**
  - context: Realtime
  - focus: Activation
  - rule: Mish/SiLU/GELU are commonly stable; keep activations consistent.
  - verify: ensure invariants in Section 4 are satisfied.

- **RULE-080**
  - context: Realtime
  - focus: Precision
  - rule: Compute variance in FP32 if using FP16 to avoid overflow.
  - verify: ensure invariants in Section 4 are satisfied.

- **RULE-081**
  - context: FP16
  - focus: RMSNorm
  - rule: Prefer RMSNorm or LayerNorm for SR stability; avoid BatchNorm.
  - verify: ensure invariants in Section 4 are satisfied.

- **RULE-082**
  - context: FP16
  - focus: Residual
  - rule: Use residual connections for feature preservation; scale residual if deep.
  - verify: ensure invariants in Section 4 are satisfied.

- **RULE-083**
  - context: FP16
  - focus: Upsample
  - rule: PixelShuffle or DySample are preferred; ensure channel math is correct.
  - verify: ensure invariants in Section 4 are satisfied.

- **RULE-084**
  - context: FP16
  - focus: DynamicSampling
  - rule: If using grid_sample, normalize offsets to [-1, 1].
  - verify: ensure invariants in Section 4 are satisfied.

- **RULE-085**
  - context: FP16
  - focus: Padding
  - rule: Pad to multiples of scale or window size, then crop output.
  - verify: ensure invariants in Section 4 are satisfied.

- **RULE-086**
  - context: FP16
  - focus: Attention
  - rule: Use windowed attention for large images; avoid global attention unless small.
  - verify: ensure invariants in Section 4 are satisfied.

- **RULE-087**
  - context: FP16
  - focus: Reparam
  - rule: Use train-time multi-branch, inference-time fused convs where possible.
  - verify: ensure invariants in Section 4 are satisfied.

- **RULE-088**
  - context: FP16
  - focus: Init
  - rule: Initialize conv weights with trunc_normal_ or similar stable init.
  - verify: ensure invariants in Section 4 are satisfied.

- **RULE-089**
  - context: FP16
  - focus: Activation
  - rule: Mish/SiLU/GELU are commonly stable; keep activations consistent.
  - verify: ensure invariants in Section 4 are satisfied.

- **RULE-090**
  - context: FP16
  - focus: Precision
  - rule: Compute variance in FP32 if using FP16 to avoid overflow.
  - verify: ensure invariants in Section 4 are satisfied.

- **RULE-091**
  - context: ONNX
  - focus: RMSNorm
  - rule: Prefer RMSNorm or LayerNorm for SR stability; avoid BatchNorm.
  - verify: ensure invariants in Section 4 are satisfied.

- **RULE-092**
  - context: ONNX
  - focus: Residual
  - rule: Use residual connections for feature preservation; scale residual if deep.
  - verify: ensure invariants in Section 4 are satisfied.

- **RULE-093**
  - context: ONNX
  - focus: Upsample
  - rule: PixelShuffle or DySample are preferred; ensure channel math is correct.
  - verify: ensure invariants in Section 4 are satisfied.

- **RULE-094**
  - context: ONNX
  - focus: DynamicSampling
  - rule: If using grid_sample, normalize offsets to [-1, 1].
  - verify: ensure invariants in Section 4 are satisfied.

- **RULE-095**
  - context: ONNX
  - focus: Padding
  - rule: Pad to multiples of scale or window size, then crop output.
  - verify: ensure invariants in Section 4 are satisfied.

- **RULE-096**
  - context: ONNX
  - focus: Attention
  - rule: Use windowed attention for large images; avoid global attention unless small.
  - verify: ensure invariants in Section 4 are satisfied.

- **RULE-097**
  - context: ONNX
  - focus: Reparam
  - rule: Use train-time multi-branch, inference-time fused convs where possible.
  - verify: ensure invariants in Section 4 are satisfied.

- **RULE-098**
  - context: ONNX
  - focus: Init
  - rule: Initialize conv weights with trunc_normal_ or similar stable init.
  - verify: ensure invariants in Section 4 are satisfied.

- **RULE-099**
  - context: ONNX
  - focus: Activation
  - rule: Mish/SiLU/GELU are commonly stable; keep activations consistent.
  - verify: ensure invariants in Section 4 are satisfied.

- **RULE-100**
  - context: ONNX
  - focus: Precision
  - rule: Compute variance in FP32 if using FP16 to avoid overflow.
  - verify: ensure invariants in Section 4 are satisfied.

- **RULE-101**
  - context: TensorRT
  - focus: RMSNorm
  - rule: Prefer RMSNorm or LayerNorm for SR stability; avoid BatchNorm.
  - verify: ensure invariants in Section 4 are satisfied.

- **RULE-102**
  - context: TensorRT
  - focus: Residual
  - rule: Use residual connections for feature preservation; scale residual if deep.
  - verify: ensure invariants in Section 4 are satisfied.

- **RULE-103**
  - context: TensorRT
  - focus: Upsample
  - rule: PixelShuffle or DySample are preferred; ensure channel math is correct.
  - verify: ensure invariants in Section 4 are satisfied.

- **RULE-104**
  - context: TensorRT
  - focus: DynamicSampling
  - rule: If using grid_sample, normalize offsets to [-1, 1].
  - verify: ensure invariants in Section 4 are satisfied.

- **RULE-105**
  - context: TensorRT
  - focus: Padding
  - rule: Pad to multiples of scale or window size, then crop output.
  - verify: ensure invariants in Section 4 are satisfied.

- **RULE-106**
  - context: TensorRT
  - focus: Attention
  - rule: Use windowed attention for large images; avoid global attention unless small.
  - verify: ensure invariants in Section 4 are satisfied.

- **RULE-107**
  - context: TensorRT
  - focus: Reparam
  - rule: Use train-time multi-branch, inference-time fused convs where possible.
  - verify: ensure invariants in Section 4 are satisfied.

- **RULE-108**
  - context: TensorRT
  - focus: Init
  - rule: Initialize conv weights with trunc_normal_ or similar stable init.
  - verify: ensure invariants in Section 4 are satisfied.

- **RULE-109**
  - context: TensorRT
  - focus: Activation
  - rule: Mish/SiLU/GELU are commonly stable; keep activations consistent.
  - verify: ensure invariants in Section 4 are satisfied.

- **RULE-110**
  - context: TensorRT
  - focus: Precision
  - rule: Compute variance in FP32 if using FP16 to avoid overflow.
  - verify: ensure invariants in Section 4 are satisfied.

- **RULE-111**
  - context: Mobile
  - focus: RMSNorm
  - rule: Prefer RMSNorm or LayerNorm for SR stability; avoid BatchNorm.
  - verify: ensure invariants in Section 4 are satisfied.

- **RULE-112**
  - context: Mobile
  - focus: Residual
  - rule: Use residual connections for feature preservation; scale residual if deep.
  - verify: ensure invariants in Section 4 are satisfied.

- **RULE-113**
  - context: Mobile
  - focus: Upsample
  - rule: PixelShuffle or DySample are preferred; ensure channel math is correct.
  - verify: ensure invariants in Section 4 are satisfied.

- **RULE-114**
  - context: Mobile
  - focus: DynamicSampling
  - rule: If using grid_sample, normalize offsets to [-1, 1].
  - verify: ensure invariants in Section 4 are satisfied.

- **RULE-115**
  - context: Mobile
  - focus: Padding
  - rule: Pad to multiples of scale or window size, then crop output.
  - verify: ensure invariants in Section 4 are satisfied.

- **RULE-116**
  - context: Mobile
  - focus: Attention
  - rule: Use windowed attention for large images; avoid global attention unless small.
  - verify: ensure invariants in Section 4 are satisfied.

- **RULE-117**
  - context: Mobile
  - focus: Reparam
  - rule: Use train-time multi-branch, inference-time fused convs where possible.
  - verify: ensure invariants in Section 4 are satisfied.

- **RULE-118**
  - context: Mobile
  - focus: Init
  - rule: Initialize conv weights with trunc_normal_ or similar stable init.
  - verify: ensure invariants in Section 4 are satisfied.

- **RULE-119**
  - context: Mobile
  - focus: Activation
  - rule: Mish/SiLU/GELU are commonly stable; keep activations consistent.
  - verify: ensure invariants in Section 4 are satisfied.

- **RULE-120**
  - context: Mobile
  - focus: Precision
  - rule: Compute variance in FP32 if using FP16 to avoid overflow.
  - verify: ensure invariants in Section 4 are satisfied.

- **RULE-121**
  - context: LargeScale
  - focus: RMSNorm
  - rule: Prefer RMSNorm or LayerNorm for SR stability; avoid BatchNorm.
  - verify: ensure invariants in Section 4 are satisfied.

- **RULE-122**
  - context: LargeScale
  - focus: Residual
  - rule: Use residual connections for feature preservation; scale residual if deep.
  - verify: ensure invariants in Section 4 are satisfied.

- **RULE-123**
  - context: LargeScale
  - focus: Upsample
  - rule: PixelShuffle or DySample are preferred; ensure channel math is correct.
  - verify: ensure invariants in Section 4 are satisfied.

- **RULE-124**
  - context: LargeScale
  - focus: DynamicSampling
  - rule: If using grid_sample, normalize offsets to [-1, 1].
  - verify: ensure invariants in Section 4 are satisfied.

- **RULE-125**
  - context: LargeScale
  - focus: Padding
  - rule: Pad to multiples of scale or window size, then crop output.
  - verify: ensure invariants in Section 4 are satisfied.

- **RULE-126**
  - context: LargeScale
  - focus: Attention
  - rule: Use windowed attention for large images; avoid global attention unless small.
  - verify: ensure invariants in Section 4 are satisfied.

- **RULE-127**
  - context: LargeScale
  - focus: Reparam
  - rule: Use train-time multi-branch, inference-time fused convs where possible.
  - verify: ensure invariants in Section 4 are satisfied.

- **RULE-128**
  - context: LargeScale
  - focus: Init
  - rule: Initialize conv weights with trunc_normal_ or similar stable init.
  - verify: ensure invariants in Section 4 are satisfied.

- **RULE-129**
  - context: LargeScale
  - focus: Activation
  - rule: Mish/SiLU/GELU are commonly stable; keep activations consistent.
  - verify: ensure invariants in Section 4 are satisfied.

- **RULE-130**
  - context: LargeScale
  - focus: Precision
  - rule: Compute variance in FP32 if using FP16 to avoid overflow.
  - verify: ensure invariants in Section 4 are satisfied.

- **RULE-131**
  - context: MultiScale
  - focus: RMSNorm
  - rule: Prefer RMSNorm or LayerNorm for SR stability; avoid BatchNorm.
  - verify: ensure invariants in Section 4 are satisfied.

- **RULE-132**
  - context: MultiScale
  - focus: Residual
  - rule: Use residual connections for feature preservation; scale residual if deep.
  - verify: ensure invariants in Section 4 are satisfied.

- **RULE-133**
  - context: MultiScale
  - focus: Upsample
  - rule: PixelShuffle or DySample are preferred; ensure channel math is correct.
  - verify: ensure invariants in Section 4 are satisfied.

- **RULE-134**
  - context: MultiScale
  - focus: DynamicSampling
  - rule: If using grid_sample, normalize offsets to [-1, 1].
  - verify: ensure invariants in Section 4 are satisfied.

- **RULE-135**
  - context: MultiScale
  - focus: Padding
  - rule: Pad to multiples of scale or window size, then crop output.
  - verify: ensure invariants in Section 4 are satisfied.

- **RULE-136**
  - context: MultiScale
  - focus: Attention
  - rule: Use windowed attention for large images; avoid global attention unless small.
  - verify: ensure invariants in Section 4 are satisfied.

- **RULE-137**
  - context: MultiScale
  - focus: Reparam
  - rule: Use train-time multi-branch, inference-time fused convs where possible.
  - verify: ensure invariants in Section 4 are satisfied.

- **RULE-138**
  - context: MultiScale
  - focus: Init
  - rule: Initialize conv weights with trunc_normal_ or similar stable init.
  - verify: ensure invariants in Section 4 are satisfied.

- **RULE-139**
  - context: MultiScale
  - focus: Activation
  - rule: Mish/SiLU/GELU are commonly stable; keep activations consistent.
  - verify: ensure invariants in Section 4 are satisfied.

- **RULE-140**
  - context: MultiScale
  - focus: Precision
  - rule: Compute variance in FP32 if using FP16 to avoid overflow.
  - verify: ensure invariants in Section 4 are satisfied.

- **RULE-141**
  - context: MoE
  - focus: RMSNorm
  - rule: Prefer RMSNorm or LayerNorm for SR stability; avoid BatchNorm.
  - verify: ensure invariants in Section 4 are satisfied.

- **RULE-142**
  - context: MoE
  - focus: Residual
  - rule: Use residual connections for feature preservation; scale residual if deep.
  - verify: ensure invariants in Section 4 are satisfied.

- **RULE-143**
  - context: MoE
  - focus: Upsample
  - rule: PixelShuffle or DySample are preferred; ensure channel math is correct.
  - verify: ensure invariants in Section 4 are satisfied.

- **RULE-144**
  - context: MoE
  - focus: DynamicSampling
  - rule: If using grid_sample, normalize offsets to [-1, 1].
  - verify: ensure invariants in Section 4 are satisfied.

- **RULE-145**
  - context: MoE
  - focus: Padding
  - rule: Pad to multiples of scale or window size, then crop output.
  - verify: ensure invariants in Section 4 are satisfied.

- **RULE-146**
  - context: MoE
  - focus: Attention
  - rule: Use windowed attention for large images; avoid global attention unless small.
  - verify: ensure invariants in Section 4 are satisfied.

- **RULE-147**
  - context: MoE
  - focus: Reparam
  - rule: Use train-time multi-branch, inference-time fused convs where possible.
  - verify: ensure invariants in Section 4 are satisfied.

- **RULE-148**
  - context: MoE
  - focus: Init
  - rule: Initialize conv weights with trunc_normal_ or similar stable init.
  - verify: ensure invariants in Section 4 are satisfied.

- **RULE-149**
  - context: MoE
  - focus: Activation
  - rule: Mish/SiLU/GELU are commonly stable; keep activations consistent.
  - verify: ensure invariants in Section 4 are satisfied.

- **RULE-150**
  - context: MoE
  - focus: Precision
  - rule: Compute variance in FP32 if using FP16 to avoid overflow.
  - verify: ensure invariants in Section 4 are satisfied.

---

## 6. Pattern Cards (LLM-Ready, Structured)

> Each card is a minimal recipe with invariants and failure checks.

### PC-NORM-A-001

- category: Normalization
- purpose: Use LayerNorm/RMSNorm in channels-first SR models.
- variant: baseline
- variant_desc: safe defaults, minimal params
- invariants:
  - input shape: [B, C, H, W]
  - deterministic forward in eval
  - preserves spatial alignment
- failure_modes:
  - shape mismatch after upsampling
  - unstable activations in FP16
  - OOM when attention is global
- references:
  - realplksr_arch.py
  - mosrv2_arch.py
  - gaterv3_arch.py

### PC-NORM-B-002

- category: Normalization
- purpose: Use LayerNorm/RMSNorm in channels-first SR models.
- variant: efficient
- variant_desc: reduced params, inference friendly
- invariants:
  - input shape: [B, C, H, W]
  - deterministic forward in eval
  - preserves spatial alignment
- failure_modes:
  - shape mismatch after upsampling
  - unstable activations in FP16
  - OOM when attention is global
- references:
  - realplksr_arch.py
  - mosrv2_arch.py
  - gaterv3_arch.py

### PC-NORM-C-003

- category: Normalization
- purpose: Use LayerNorm/RMSNorm in channels-first SR models.
- variant: high-capacity
- variant_desc: extra depth or attention
- invariants:
  - input shape: [B, C, H, W]
  - deterministic forward in eval
  - preserves spatial alignment
- failure_modes:
  - shape mismatch after upsampling
  - unstable activations in FP16
  - OOM when attention is global
- references:
  - realplksr_arch.py
  - mosrv2_arch.py
  - gaterv3_arch.py

### PC-NORM-D-004

- category: Normalization
- purpose: Use LayerNorm/RMSNorm in channels-first SR models.
- variant: mixed-precision
- variant_desc: FP16 safe with FP32 norms
- invariants:
  - input shape: [B, C, H, W]
  - deterministic forward in eval
  - preserves spatial alignment
- failure_modes:
  - shape mismatch after upsampling
  - unstable activations in FP16
  - OOM when attention is global
- references:
  - realplksr_arch.py
  - mosrv2_arch.py
  - gaterv3_arch.py

### PC-UP-A-005

- category: Upsampling
- purpose: Prefer PixelShuffle/DySample; handle scale math.
- variant: baseline
- variant_desc: safe defaults, minimal params
- invariants:
  - input shape: [B, C, H, W]
  - deterministic forward in eval
  - preserves spatial alignment
- failure_modes:
  - shape mismatch after upsampling
  - unstable activations in FP16
  - OOM when attention is global
- references:
  - realplksr_arch.py
  - mosrv2_arch.py
  - gaterv3_arch.py

### PC-UP-B-006

- category: Upsampling
- purpose: Prefer PixelShuffle/DySample; handle scale math.
- variant: efficient
- variant_desc: reduced params, inference friendly
- invariants:
  - input shape: [B, C, H, W]
  - deterministic forward in eval
  - preserves spatial alignment
- failure_modes:
  - shape mismatch after upsampling
  - unstable activations in FP16
  - OOM when attention is global
- references:
  - realplksr_arch.py
  - mosrv2_arch.py
  - gaterv3_arch.py

### PC-UP-C-007

- category: Upsampling
- purpose: Prefer PixelShuffle/DySample; handle scale math.
- variant: high-capacity
- variant_desc: extra depth or attention
- invariants:
  - input shape: [B, C, H, W]
  - deterministic forward in eval
  - preserves spatial alignment
- failure_modes:
  - shape mismatch after upsampling
  - unstable activations in FP16
  - OOM when attention is global
- references:
  - realplksr_arch.py
  - mosrv2_arch.py
  - gaterv3_arch.py

### PC-UP-D-008

- category: Upsampling
- purpose: Prefer PixelShuffle/DySample; handle scale math.
- variant: mixed-precision
- variant_desc: FP16 safe with FP32 norms
- invariants:
  - input shape: [B, C, H, W]
  - deterministic forward in eval
  - preserves spatial alignment
- failure_modes:
  - shape mismatch after upsampling
  - unstable activations in FP16
  - OOM when attention is global
- references:
  - realplksr_arch.py
  - mosrv2_arch.py
  - gaterv3_arch.py

### PC-REP-A-009

- category: Reparameterization
- purpose: Fuse multi-branch convs for inference.
- variant: baseline
- variant_desc: safe defaults, minimal params
- invariants:
  - input shape: [B, C, H, W]
  - deterministic forward in eval
  - preserves spatial alignment
- failure_modes:
  - shape mismatch after upsampling
  - unstable activations in FP16
  - OOM when attention is global
- references:
  - realplksr_arch.py
  - mosrv2_arch.py
  - gaterv3_arch.py

### PC-REP-B-010

- category: Reparameterization
- purpose: Fuse multi-branch convs for inference.
- variant: efficient
- variant_desc: reduced params, inference friendly
- invariants:
  - input shape: [B, C, H, W]
  - deterministic forward in eval
  - preserves spatial alignment
- failure_modes:
  - shape mismatch after upsampling
  - unstable activations in FP16
  - OOM when attention is global
- references:
  - realplksr_arch.py
  - mosrv2_arch.py
  - gaterv3_arch.py

### PC-REP-C-011

- category: Reparameterization
- purpose: Fuse multi-branch convs for inference.
- variant: high-capacity
- variant_desc: extra depth or attention
- invariants:
  - input shape: [B, C, H, W]
  - deterministic forward in eval
  - preserves spatial alignment
- failure_modes:
  - shape mismatch after upsampling
  - unstable activations in FP16
  - OOM when attention is global
- references:
  - realplksr_arch.py
  - mosrv2_arch.py
  - gaterv3_arch.py

### PC-REP-D-012

- category: Reparameterization
- purpose: Fuse multi-branch convs for inference.
- variant: mixed-precision
- variant_desc: FP16 safe with FP32 norms
- invariants:
  - input shape: [B, C, H, W]
  - deterministic forward in eval
  - preserves spatial alignment
- failure_modes:
  - shape mismatch after upsampling
  - unstable activations in FP16
  - OOM when attention is global
- references:
  - realplksr_arch.py
  - mosrv2_arch.py
  - gaterv3_arch.py

### PC-ATTN-A-013

- category: Attention
- purpose: Use windowed or local attention when H*W is large.
- variant: baseline
- variant_desc: safe defaults, minimal params
- invariants:
  - input shape: [B, C, H, W]
  - deterministic forward in eval
  - preserves spatial alignment
- failure_modes:
  - shape mismatch after upsampling
  - unstable activations in FP16
  - OOM when attention is global
- references:
  - realplksr_arch.py
  - mosrv2_arch.py
  - gaterv3_arch.py

### PC-ATTN-B-014

- category: Attention
- purpose: Use windowed or local attention when H*W is large.
- variant: efficient
- variant_desc: reduced params, inference friendly
- invariants:
  - input shape: [B, C, H, W]
  - deterministic forward in eval
  - preserves spatial alignment
- failure_modes:
  - shape mismatch after upsampling
  - unstable activations in FP16
  - OOM when attention is global
- references:
  - realplksr_arch.py
  - mosrv2_arch.py
  - gaterv3_arch.py

### PC-ATTN-C-015

- category: Attention
- purpose: Use windowed or local attention when H*W is large.
- variant: high-capacity
- variant_desc: extra depth or attention
- invariants:
  - input shape: [B, C, H, W]
  - deterministic forward in eval
  - preserves spatial alignment
- failure_modes:
  - shape mismatch after upsampling
  - unstable activations in FP16
  - OOM when attention is global
- references:
  - realplksr_arch.py
  - mosrv2_arch.py
  - gaterv3_arch.py

### PC-ATTN-D-016

- category: Attention
- purpose: Use windowed or local attention when H*W is large.
- variant: mixed-precision
- variant_desc: FP16 safe with FP32 norms
- invariants:
  - input shape: [B, C, H, W]
  - deterministic forward in eval
  - preserves spatial alignment
- failure_modes:
  - shape mismatch after upsampling
  - unstable activations in FP16
  - OOM when attention is global
- references:
  - realplksr_arch.py
  - mosrv2_arch.py
  - gaterv3_arch.py

### PC-GATE-A-017

- category: Gating
- purpose: Use gated CNN blocks to mix local/global context.
- variant: baseline
- variant_desc: safe defaults, minimal params
- invariants:
  - input shape: [B, C, H, W]
  - deterministic forward in eval
  - preserves spatial alignment
- failure_modes:
  - shape mismatch after upsampling
  - unstable activations in FP16
  - OOM when attention is global
- references:
  - realplksr_arch.py
  - mosrv2_arch.py
  - gaterv3_arch.py

### PC-GATE-B-018

- category: Gating
- purpose: Use gated CNN blocks to mix local/global context.
- variant: efficient
- variant_desc: reduced params, inference friendly
- invariants:
  - input shape: [B, C, H, W]
  - deterministic forward in eval
  - preserves spatial alignment
- failure_modes:
  - shape mismatch after upsampling
  - unstable activations in FP16
  - OOM when attention is global
- references:
  - realplksr_arch.py
  - mosrv2_arch.py
  - gaterv3_arch.py

### PC-GATE-C-019

- category: Gating
- purpose: Use gated CNN blocks to mix local/global context.
- variant: high-capacity
- variant_desc: extra depth or attention
- invariants:
  - input shape: [B, C, H, W]
  - deterministic forward in eval
  - preserves spatial alignment
- failure_modes:
  - shape mismatch after upsampling
  - unstable activations in FP16
  - OOM when attention is global
- references:
  - realplksr_arch.py
  - mosrv2_arch.py
  - gaterv3_arch.py

### PC-GATE-D-020

- category: Gating
- purpose: Use gated CNN blocks to mix local/global context.
- variant: mixed-precision
- variant_desc: FP16 safe with FP32 norms
- invariants:
  - input shape: [B, C, H, W]
  - deterministic forward in eval
  - preserves spatial alignment
- failure_modes:
  - shape mismatch after upsampling
  - unstable activations in FP16
  - OOM when attention is global
- references:
  - realplksr_arch.py
  - mosrv2_arch.py
  - gaterv3_arch.py

### PC-RES-A-021

- category: Residual
- purpose: Preserve low-frequency content with residual paths.
- variant: baseline
- variant_desc: safe defaults, minimal params
- invariants:
  - input shape: [B, C, H, W]
  - deterministic forward in eval
  - preserves spatial alignment
- failure_modes:
  - shape mismatch after upsampling
  - unstable activations in FP16
  - OOM when attention is global
- references:
  - realplksr_arch.py
  - mosrv2_arch.py
  - gaterv3_arch.py

### PC-RES-B-022

- category: Residual
- purpose: Preserve low-frequency content with residual paths.
- variant: efficient
- variant_desc: reduced params, inference friendly
- invariants:
  - input shape: [B, C, H, W]
  - deterministic forward in eval
  - preserves spatial alignment
- failure_modes:
  - shape mismatch after upsampling
  - unstable activations in FP16
  - OOM when attention is global
- references:
  - realplksr_arch.py
  - mosrv2_arch.py
  - gaterv3_arch.py

### PC-RES-C-023

- category: Residual
- purpose: Preserve low-frequency content with residual paths.
- variant: high-capacity
- variant_desc: extra depth or attention
- invariants:
  - input shape: [B, C, H, W]
  - deterministic forward in eval
  - preserves spatial alignment
- failure_modes:
  - shape mismatch after upsampling
  - unstable activations in FP16
  - OOM when attention is global
- references:
  - realplksr_arch.py
  - mosrv2_arch.py
  - gaterv3_arch.py

### PC-RES-D-024

- category: Residual
- purpose: Preserve low-frequency content with residual paths.
- variant: mixed-precision
- variant_desc: FP16 safe with FP32 norms
- invariants:
  - input shape: [B, C, H, W]
  - deterministic forward in eval
  - preserves spatial alignment
- failure_modes:
  - shape mismatch after upsampling
  - unstable activations in FP16
  - OOM when attention is global
- references:
  - realplksr_arch.py
  - mosrv2_arch.py
  - gaterv3_arch.py

### PC-PAD-A-025

- category: Padding
- purpose: Pad to multiples for windowing and unshuffle.
- variant: baseline
- variant_desc: safe defaults, minimal params
- invariants:
  - input shape: [B, C, H, W]
  - deterministic forward in eval
  - preserves spatial alignment
- failure_modes:
  - shape mismatch after upsampling
  - unstable activations in FP16
  - OOM when attention is global
- references:
  - realplksr_arch.py
  - mosrv2_arch.py
  - gaterv3_arch.py

### PC-PAD-B-026

- category: Padding
- purpose: Pad to multiples for windowing and unshuffle.
- variant: efficient
- variant_desc: reduced params, inference friendly
- invariants:
  - input shape: [B, C, H, W]
  - deterministic forward in eval
  - preserves spatial alignment
- failure_modes:
  - shape mismatch after upsampling
  - unstable activations in FP16
  - OOM when attention is global
- references:
  - realplksr_arch.py
  - mosrv2_arch.py
  - gaterv3_arch.py

### PC-PAD-C-027

- category: Padding
- purpose: Pad to multiples for windowing and unshuffle.
- variant: high-capacity
- variant_desc: extra depth or attention
- invariants:
  - input shape: [B, C, H, W]
  - deterministic forward in eval
  - preserves spatial alignment
- failure_modes:
  - shape mismatch after upsampling
  - unstable activations in FP16
  - OOM when attention is global
- references:
  - realplksr_arch.py
  - mosrv2_arch.py
  - gaterv3_arch.py

### PC-PAD-D-028

- category: Padding
- purpose: Pad to multiples for windowing and unshuffle.
- variant: mixed-precision
- variant_desc: FP16 safe with FP32 norms
- invariants:
  - input shape: [B, C, H, W]
  - deterministic forward in eval
  - preserves spatial alignment
- failure_modes:
  - shape mismatch after upsampling
  - unstable activations in FP16
  - OOM when attention is global
- references:
  - realplksr_arch.py
  - mosrv2_arch.py
  - gaterv3_arch.py

### PC-DYN-A-029

- category: DynamicKernel
- purpose: Generate kernels with bounded magnitude.
- variant: baseline
- variant_desc: safe defaults, minimal params
- invariants:
  - input shape: [B, C, H, W]
  - deterministic forward in eval
  - preserves spatial alignment
- failure_modes:
  - shape mismatch after upsampling
  - unstable activations in FP16
  - OOM when attention is global
- references:
  - realplksr_arch.py
  - mosrv2_arch.py
  - gaterv3_arch.py

### PC-DYN-B-030

- category: DynamicKernel
- purpose: Generate kernels with bounded magnitude.
- variant: efficient
- variant_desc: reduced params, inference friendly
- invariants:
  - input shape: [B, C, H, W]
  - deterministic forward in eval
  - preserves spatial alignment
- failure_modes:
  - shape mismatch after upsampling
  - unstable activations in FP16
  - OOM when attention is global
- references:
  - realplksr_arch.py
  - mosrv2_arch.py
  - gaterv3_arch.py

### PC-DYN-C-031

- category: DynamicKernel
- purpose: Generate kernels with bounded magnitude.
- variant: high-capacity
- variant_desc: extra depth or attention
- invariants:
  - input shape: [B, C, H, W]
  - deterministic forward in eval
  - preserves spatial alignment
- failure_modes:
  - shape mismatch after upsampling
  - unstable activations in FP16
  - OOM when attention is global
- references:
  - realplksr_arch.py
  - mosrv2_arch.py
  - gaterv3_arch.py

### PC-DYN-D-032

- category: DynamicKernel
- purpose: Generate kernels with bounded magnitude.
- variant: mixed-precision
- variant_desc: FP16 safe with FP32 norms
- invariants:
  - input shape: [B, C, H, W]
  - deterministic forward in eval
  - preserves spatial alignment
- failure_modes:
  - shape mismatch after upsampling
  - unstable activations in FP16
  - OOM when attention is global
- references:
  - realplksr_arch.py
  - mosrv2_arch.py
  - gaterv3_arch.py

---

## 7. Implementation Templates (Pseudo + PyTorch)

### 7.1 High-Level Pseudocode (SISR Core)

```text
function SISR(x, scale):
  x = maybe_pad_to_multiple(x, scale)
  base = classical_upsample(x, scale)  # optional
  feat = shallow_conv(x)
  for block in blocks:
      feat = block(feat) + feat
  out = upsampler(feat) + base
  out = crop_to_original(out, scale)
  return out
```

### 7.2 PyTorch Skeleton (LLM-Friendly)

```python
class SISRModel(nn.Module):
    def __init__(self, in_ch=3, scale=4, dim=64, n_blocks=24):
        super().__init__()
        self.scale = scale
        self.head = nn.Conv2d(in_ch, dim, 3, 1, 1)
        self.blocks = nn.Sequential(*[Block(dim) for _ in range(n_blocks)])
        self.tail = nn.Conv2d(dim, in_ch * scale * scale, 3, 1, 1)
        self.upsampler = nn.PixelShuffle(scale)

    def forward(self, x):
        b, c, h, w = x.shape
        feat = self.head(x)
        feat = self.blocks(feat) + feat
        out = self.upsampler(self.tail(feat))
        return out[:, :, : h * self.scale, : w * self.scale]
```

---

## 8. Optimization & Deployment Guidance

- Fuse re-parameterizable convs (e.g., Conv3XC or RepConv) before export.
- Prefer SDPA / Flash attention when supported; fallback to naive for determinism.
- Use channels_last memory format if LayerNorm supports it and GPU benefits.
- Avoid Python control flow in forward when exporting to ONNX (use flags).
- Verify dynamic sampling ops are supported by the target runtime.

---

## 9. Verification Checklist

- [ ] Input/output shapes correct for all supported scales.
- [ ] Eval mode produces deterministic results (no dropout, no random).
- [ ] Reparameterization fused before inference export.
- [ ] Pad/crop logic validated on non-divisible image sizes.
- [ ] Mixed precision verified (FP16 safe norms).
- [ ] Memory usage checked on 1K/2K/4K inputs.

---

## 10. Extended Ruleset (Line Count Extension, Structured)

> The following lines extend the ruleset to ensure the document is highly detailed and LLM-readable.

- EXT-RULE-0001: Preserve feature alignment; avoid implicit reshapes that reorder pixels.
- EXT-RULE-0002: If using pixel unshuffle, adjust padding so that H and W are divisible by factor.
- EXT-RULE-0003: When fusing reparam blocks, validate numerical equivalence within tolerance.
- EXT-RULE-0004: Prefer depthwise conv for spatial mixing when compute is constrained.
- EXT-RULE-0005: Use SiLU/Mish for gating blocks to stabilize multiplicative interactions.
- EXT-RULE-0006: Avoid global attention on >1MP images; prefer windows or local kernels.
- EXT-RULE-0007: When using SDPA, ensure q/k/v shapes are [B, heads, N, dim].
- EXT-RULE-0008: Compute attention scores in FP32 if using half precision in training.
- EXT-RULE-0009: Crop output after upsampling to match original H*scale, W*scale.
- EXT-RULE-0010: When using grid_sample, set align_corners=False for SR consistency.
- EXT-RULE-0011: Always register fixed kernels as buffers (requires_grad=False).
- EXT-RULE-0012: Do not let dynamic kernels exceed stable magnitude; apply sigmoid/tanh if needed.
- EXT-RULE-0013: Normalize positional bias tables if they cause divergence in early epochs.
- EXT-RULE-0014: Use checkpointing sparingly; verify speed vs memory trade-off.
- EXT-RULE-0015: Keep recursion depth small for MoE/recursive blocks to avoid instability.
- EXT-RULE-0016: Use group norm or RMSNorm for batch=1 or small batch regimes.
- EXT-RULE-0017: If using channels_last, ensure LayerNorm supports the layout.
- EXT-RULE-0018: Prefer separable conv for classical filters (sharp/blur) to reduce compute.
- EXT-RULE-0019: Avoid hard-coded device creation inside forward; use x.device.
- EXT-RULE-0020: Prefer pad + crop to handle arbitrary image sizes without artifacts.
- EXT-RULE-0021: Preserve feature alignment; avoid implicit reshapes that reorder pixels.
- EXT-RULE-0022: If using pixel unshuffle, adjust padding so that H and W are divisible by factor.
- EXT-RULE-0023: When fusing reparam blocks, validate numerical equivalence within tolerance.
- EXT-RULE-0024: Prefer depthwise conv for spatial mixing when compute is constrained.
- EXT-RULE-0025: Use SiLU/Mish for gating blocks to stabilize multiplicative interactions.
- EXT-RULE-0026: Avoid global attention on >1MP images; prefer windows or local kernels.
- EXT-RULE-0027: When using SDPA, ensure q/k/v shapes are [B, heads, N, dim].
- EXT-RULE-0028: Compute attention scores in FP32 if using half precision in training.
- EXT-RULE-0029: Crop output after upsampling to match original H*scale, W*scale.
- EXT-RULE-0030: When using grid_sample, set align_corners=False for SR consistency.
- EXT-RULE-0031: Always register fixed kernels as buffers (requires_grad=False).
- EXT-RULE-0032: Do not let dynamic kernels exceed stable magnitude; apply sigmoid/tanh if needed.
- EXT-RULE-0033: Normalize positional bias tables if they cause divergence in early epochs.
- EXT-RULE-0034: Use checkpointing sparingly; verify speed vs memory trade-off.
- EXT-RULE-0035: Keep recursion depth small for MoE/recursive blocks to avoid instability.
- EXT-RULE-0036: Use group norm or RMSNorm for batch=1 or small batch regimes.
- EXT-RULE-0037: If using channels_last, ensure LayerNorm supports the layout.
- EXT-RULE-0038: Prefer separable conv for classical filters (sharp/blur) to reduce compute.
- EXT-RULE-0039: Avoid hard-coded device creation inside forward; use x.device.
- EXT-RULE-0040: Prefer pad + crop to handle arbitrary image sizes without artifacts.
- EXT-RULE-0041: Preserve feature alignment; avoid implicit reshapes that reorder pixels.
- EXT-RULE-0042: If using pixel unshuffle, adjust padding so that H and W are divisible by factor.
- EXT-RULE-0043: When fusing reparam blocks, validate numerical equivalence within tolerance.
- EXT-RULE-0044: Prefer depthwise conv for spatial mixing when compute is constrained.
- EXT-RULE-0045: Use SiLU/Mish for gating blocks to stabilize multiplicative interactions.
- EXT-RULE-0046: Avoid global attention on >1MP images; prefer windows or local kernels.
- EXT-RULE-0047: When using SDPA, ensure q/k/v shapes are [B, heads, N, dim].
- EXT-RULE-0048: Compute attention scores in FP32 if using half precision in training.
- EXT-RULE-0049: Crop output after upsampling to match original H*scale, W*scale.
- EXT-RULE-0050: When using grid_sample, set align_corners=False for SR consistency.
- EXT-RULE-0051: Always register fixed kernels as buffers (requires_grad=False).
- EXT-RULE-0052: Do not let dynamic kernels exceed stable magnitude; apply sigmoid/tanh if needed.
- EXT-RULE-0053: Normalize positional bias tables if they cause divergence in early epochs.
- EXT-RULE-0054: Use checkpointing sparingly; verify speed vs memory trade-off.
- EXT-RULE-0055: Keep recursion depth small for MoE/recursive blocks to avoid instability.
- EXT-RULE-0056: Use group norm or RMSNorm for batch=1 or small batch regimes.
- EXT-RULE-0057: If using channels_last, ensure LayerNorm supports the layout.
- EXT-RULE-0058: Prefer separable conv for classical filters (sharp/blur) to reduce compute.
- EXT-RULE-0059: Avoid hard-coded device creation inside forward; use x.device.
- EXT-RULE-0060: Prefer pad + crop to handle arbitrary image sizes without artifacts.
- EXT-RULE-0061: Preserve feature alignment; avoid implicit reshapes that reorder pixels.
- EXT-RULE-0062: If using pixel unshuffle, adjust padding so that H and W are divisible by factor.
- EXT-RULE-0063: When fusing reparam blocks, validate numerical equivalence within tolerance.
- EXT-RULE-0064: Prefer depthwise conv for spatial mixing when compute is constrained.
- EXT-RULE-0065: Use SiLU/Mish for gating blocks to stabilize multiplicative interactions.
- EXT-RULE-0066: Avoid global attention on >1MP images; prefer windows or local kernels.
- EXT-RULE-0067: When using SDPA, ensure q/k/v shapes are [B, heads, N, dim].
- EXT-RULE-0068: Compute attention scores in FP32 if using half precision in training.
- EXT-RULE-0069: Crop output after upsampling to match original H*scale, W*scale.
- EXT-RULE-0070: When using grid_sample, set align_corners=False for SR consistency.
- EXT-RULE-0071: Always register fixed kernels as buffers (requires_grad=False).
- EXT-RULE-0072: Do not let dynamic kernels exceed stable magnitude; apply sigmoid/tanh if needed.
- EXT-RULE-0073: Normalize positional bias tables if they cause divergence in early epochs.
- EXT-RULE-0074: Use checkpointing sparingly; verify speed vs memory trade-off.
- EXT-RULE-0075: Keep recursion depth small for MoE/recursive blocks to avoid instability.
- EXT-RULE-0076: Use group norm or RMSNorm for batch=1 or small batch regimes.
- EXT-RULE-0077: If using channels_last, ensure LayerNorm supports the layout.
- EXT-RULE-0078: Prefer separable conv for classical filters (sharp/blur) to reduce compute.
- EXT-RULE-0079: Avoid hard-coded device creation inside forward; use x.device.
- EXT-RULE-0080: Prefer pad + crop to handle arbitrary image sizes without artifacts.
- EXT-RULE-0081: Preserve feature alignment; avoid implicit reshapes that reorder pixels.
- EXT-RULE-0082: If using pixel unshuffle, adjust padding so that H and W are divisible by factor.
- EXT-RULE-0083: When fusing reparam blocks, validate numerical equivalence within tolerance.
- EXT-RULE-0084: Prefer depthwise conv for spatial mixing when compute is constrained.
- EXT-RULE-0085: Use SiLU/Mish for gating blocks to stabilize multiplicative interactions.
- EXT-RULE-0086: Avoid global attention on >1MP images; prefer windows or local kernels.
- EXT-RULE-0087: When using SDPA, ensure q/k/v shapes are [B, heads, N, dim].
- EXT-RULE-0088: Compute attention scores in FP32 if using half precision in training.
- EXT-RULE-0089: Crop output after upsampling to match original H*scale, W*scale.
- EXT-RULE-0090: When using grid_sample, set align_corners=False for SR consistency.
- EXT-RULE-0091: Always register fixed kernels as buffers (requires_grad=False).
- EXT-RULE-0092: Do not let dynamic kernels exceed stable magnitude; apply sigmoid/tanh if needed.
- EXT-RULE-0093: Normalize positional bias tables if they cause divergence in early epochs.
- EXT-RULE-0094: Use checkpointing sparingly; verify speed vs memory trade-off.
- EXT-RULE-0095: Keep recursion depth small for MoE/recursive blocks to avoid instability.
- EXT-RULE-0096: Use group norm or RMSNorm for batch=1 or small batch regimes.
- EXT-RULE-0097: If using channels_last, ensure LayerNorm supports the layout.
- EXT-RULE-0098: Prefer separable conv for classical filters (sharp/blur) to reduce compute.
- EXT-RULE-0099: Avoid hard-coded device creation inside forward; use x.device.
- EXT-RULE-0100: Prefer pad + crop to handle arbitrary image sizes without artifacts.
- EXT-RULE-0101: Preserve feature alignment; avoid implicit reshapes that reorder pixels.
- EXT-RULE-0102: If using pixel unshuffle, adjust padding so that H and W are divisible by factor.
- EXT-RULE-0103: When fusing reparam blocks, validate numerical equivalence within tolerance.
- EXT-RULE-0104: Prefer depthwise conv for spatial mixing when compute is constrained.
- EXT-RULE-0105: Use SiLU/Mish for gating blocks to stabilize multiplicative interactions.
- EXT-RULE-0106: Avoid global attention on >1MP images; prefer windows or local kernels.
- EXT-RULE-0107: When using SDPA, ensure q/k/v shapes are [B, heads, N, dim].
- EXT-RULE-0108: Compute attention scores in FP32 if using half precision in training.
- EXT-RULE-0109: Crop output after upsampling to match original H*scale, W*scale.
- EXT-RULE-0110: When using grid_sample, set align_corners=False for SR consistency.
- EXT-RULE-0111: Always register fixed kernels as buffers (requires_grad=False).
- EXT-RULE-0112: Do not let dynamic kernels exceed stable magnitude; apply sigmoid/tanh if needed.
- EXT-RULE-0113: Normalize positional bias tables if they cause divergence in early epochs.
- EXT-RULE-0114: Use checkpointing sparingly; verify speed vs memory trade-off.
- EXT-RULE-0115: Keep recursion depth small for MoE/recursive blocks to avoid instability.
- EXT-RULE-0116: Use group norm or RMSNorm for batch=1 or small batch regimes.
- EXT-RULE-0117: If using channels_last, ensure LayerNorm supports the layout.
- EXT-RULE-0118: Prefer separable conv for classical filters (sharp/blur) to reduce compute.
- EXT-RULE-0119: Avoid hard-coded device creation inside forward; use x.device.
- EXT-RULE-0120: Prefer pad + crop to handle arbitrary image sizes without artifacts.
- EXT-RULE-0121: Preserve feature alignment; avoid implicit reshapes that reorder pixels.
- EXT-RULE-0122: If using pixel unshuffle, adjust padding so that H and W are divisible by factor.
- EXT-RULE-0123: When fusing reparam blocks, validate numerical equivalence within tolerance.
- EXT-RULE-0124: Prefer depthwise conv for spatial mixing when compute is constrained.
- EXT-RULE-0125: Use SiLU/Mish for gating blocks to stabilize multiplicative interactions.
- EXT-RULE-0126: Avoid global attention on >1MP images; prefer windows or local kernels.
- EXT-RULE-0127: When using SDPA, ensure q/k/v shapes are [B, heads, N, dim].
- EXT-RULE-0128: Compute attention scores in FP32 if using half precision in training.
- EXT-RULE-0129: Crop output after upsampling to match original H*scale, W*scale.
- EXT-RULE-0130: When using grid_sample, set align_corners=False for SR consistency.
- EXT-RULE-0131: Always register fixed kernels as buffers (requires_grad=False).
- EXT-RULE-0132: Do not let dynamic kernels exceed stable magnitude; apply sigmoid/tanh if needed.
- EXT-RULE-0133: Normalize positional bias tables if they cause divergence in early epochs.
- EXT-RULE-0134: Use checkpointing sparingly; verify speed vs memory trade-off.
- EXT-RULE-0135: Keep recursion depth small for MoE/recursive blocks to avoid instability.
- EXT-RULE-0136: Use group norm or RMSNorm for batch=1 or small batch regimes.
- EXT-RULE-0137: If using channels_last, ensure LayerNorm supports the layout.
- EXT-RULE-0138: Prefer separable conv for classical filters (sharp/blur) to reduce compute.
- EXT-RULE-0139: Avoid hard-coded device creation inside forward; use x.device.
- EXT-RULE-0140: Prefer pad + crop to handle arbitrary image sizes without artifacts.
- EXT-RULE-0141: Preserve feature alignment; avoid implicit reshapes that reorder pixels.
- EXT-RULE-0142: If using pixel unshuffle, adjust padding so that H and W are divisible by factor.
- EXT-RULE-0143: When fusing reparam blocks, validate numerical equivalence within tolerance.
- EXT-RULE-0144: Prefer depthwise conv for spatial mixing when compute is constrained.
- EXT-RULE-0145: Use SiLU/Mish for gating blocks to stabilize multiplicative interactions.
- EXT-RULE-0146: Avoid global attention on >1MP images; prefer windows or local kernels.
- EXT-RULE-0147: When using SDPA, ensure q/k/v shapes are [B, heads, N, dim].
- EXT-RULE-0148: Compute attention scores in FP32 if using half precision in training.
- EXT-RULE-0149: Crop output after upsampling to match original H*scale, W*scale.
- EXT-RULE-0150: When using grid_sample, set align_corners=False for SR consistency.
- EXT-RULE-0151: Always register fixed kernels as buffers (requires_grad=False).
- EXT-RULE-0152: Do not let dynamic kernels exceed stable magnitude; apply sigmoid/tanh if needed.
- EXT-RULE-0153: Normalize positional bias tables if they cause divergence in early epochs.
- EXT-RULE-0154: Use checkpointing sparingly; verify speed vs memory trade-off.
- EXT-RULE-0155: Keep recursion depth small for MoE/recursive blocks to avoid instability.
- EXT-RULE-0156: Use group norm or RMSNorm for batch=1 or small batch regimes.
- EXT-RULE-0157: If using channels_last, ensure LayerNorm supports the layout.
- EXT-RULE-0158: Prefer separable conv for classical filters (sharp/blur) to reduce compute.
- EXT-RULE-0159: Avoid hard-coded device creation inside forward; use x.device.
- EXT-RULE-0160: Prefer pad + crop to handle arbitrary image sizes without artifacts.
- EXT-RULE-0161: Preserve feature alignment; avoid implicit reshapes that reorder pixels.
- EXT-RULE-0162: If using pixel unshuffle, adjust padding so that H and W are divisible by factor.
- EXT-RULE-0163: When fusing reparam blocks, validate numerical equivalence within tolerance.
- EXT-RULE-0164: Prefer depthwise conv for spatial mixing when compute is constrained.
- EXT-RULE-0165: Use SiLU/Mish for gating blocks to stabilize multiplicative interactions.
- EXT-RULE-0166: Avoid global attention on >1MP images; prefer windows or local kernels.
- EXT-RULE-0167: When using SDPA, ensure q/k/v shapes are [B, heads, N, dim].
- EXT-RULE-0168: Compute attention scores in FP32 if using half precision in training.
- EXT-RULE-0169: Crop output after upsampling to match original H*scale, W*scale.
- EXT-RULE-0170: When using grid_sample, set align_corners=False for SR consistency.
- EXT-RULE-0171: Always register fixed kernels as buffers (requires_grad=False).
- EXT-RULE-0172: Do not let dynamic kernels exceed stable magnitude; apply sigmoid/tanh if needed.
- EXT-RULE-0173: Normalize positional bias tables if they cause divergence in early epochs.
- EXT-RULE-0174: Use checkpointing sparingly; verify speed vs memory trade-off.
- EXT-RULE-0175: Keep recursion depth small for MoE/recursive blocks to avoid instability.
- EXT-RULE-0176: Use group norm or RMSNorm for batch=1 or small batch regimes.
- EXT-RULE-0177: If using channels_last, ensure LayerNorm supports the layout.
- EXT-RULE-0178: Prefer separable conv for classical filters (sharp/blur) to reduce compute.
- EXT-RULE-0179: Avoid hard-coded device creation inside forward; use x.device.
- EXT-RULE-0180: Prefer pad + crop to handle arbitrary image sizes without artifacts.
- EXT-RULE-0181: Preserve feature alignment; avoid implicit reshapes that reorder pixels.
- EXT-RULE-0182: If using pixel unshuffle, adjust padding so that H and W are divisible by factor.
- EXT-RULE-0183: When fusing reparam blocks, validate numerical equivalence within tolerance.
- EXT-RULE-0184: Prefer depthwise conv for spatial mixing when compute is constrained.
- EXT-RULE-0185: Use SiLU/Mish for gating blocks to stabilize multiplicative interactions.
- EXT-RULE-0186: Avoid global attention on >1MP images; prefer windows or local kernels.
- EXT-RULE-0187: When using SDPA, ensure q/k/v shapes are [B, heads, N, dim].
- EXT-RULE-0188: Compute attention scores in FP32 if using half precision in training.
- EXT-RULE-0189: Crop output after upsampling to match original H*scale, W*scale.
- EXT-RULE-0190: When using grid_sample, set align_corners=False for SR consistency.
- EXT-RULE-0191: Always register fixed kernels as buffers (requires_grad=False).
- EXT-RULE-0192: Do not let dynamic kernels exceed stable magnitude; apply sigmoid/tanh if needed.
- EXT-RULE-0193: Normalize positional bias tables if they cause divergence in early epochs.
- EXT-RULE-0194: Use checkpointing sparingly; verify speed vs memory trade-off.
- EXT-RULE-0195: Keep recursion depth small for MoE/recursive blocks to avoid instability.
- EXT-RULE-0196: Use group norm or RMSNorm for batch=1 or small batch regimes.
- EXT-RULE-0197: If using channels_last, ensure LayerNorm supports the layout.
- EXT-RULE-0198: Prefer separable conv for classical filters (sharp/blur) to reduce compute.
- EXT-RULE-0199: Avoid hard-coded device creation inside forward; use x.device.
- EXT-RULE-0200: Prefer pad + crop to handle arbitrary image sizes without artifacts.
- EXT-RULE-0201: Preserve feature alignment; avoid implicit reshapes that reorder pixels.
- EXT-RULE-0202: If using pixel unshuffle, adjust padding so that H and W are divisible by factor.
- EXT-RULE-0203: When fusing reparam blocks, validate numerical equivalence within tolerance.
- EXT-RULE-0204: Prefer depthwise conv for spatial mixing when compute is constrained.
- EXT-RULE-0205: Use SiLU/Mish for gating blocks to stabilize multiplicative interactions.
- EXT-RULE-0206: Avoid global attention on >1MP images; prefer windows or local kernels.
- EXT-RULE-0207: When using SDPA, ensure q/k/v shapes are [B, heads, N, dim].
- EXT-RULE-0208: Compute attention scores in FP32 if using half precision in training.
- EXT-RULE-0209: Crop output after upsampling to match original H*scale, W*scale.
- EXT-RULE-0210: When using grid_sample, set align_corners=False for SR consistency.
- EXT-RULE-0211: Always register fixed kernels as buffers (requires_grad=False).
- EXT-RULE-0212: Do not let dynamic kernels exceed stable magnitude; apply sigmoid/tanh if needed.
- EXT-RULE-0213: Normalize positional bias tables if they cause divergence in early epochs.
- EXT-RULE-0214: Use checkpointing sparingly; verify speed vs memory trade-off.
- EXT-RULE-0215: Keep recursion depth small for MoE/recursive blocks to avoid instability.
- EXT-RULE-0216: Use group norm or RMSNorm for batch=1 or small batch regimes.
- EXT-RULE-0217: If using channels_last, ensure LayerNorm supports the layout.
- EXT-RULE-0218: Prefer separable conv for classical filters (sharp/blur) to reduce compute.
- EXT-RULE-0219: Avoid hard-coded device creation inside forward; use x.device.
- EXT-RULE-0220: Prefer pad + crop to handle arbitrary image sizes without artifacts.
- EXT-RULE-0221: Preserve feature alignment; avoid implicit reshapes that reorder pixels.
- EXT-RULE-0222: If using pixel unshuffle, adjust padding so that H and W are divisible by factor.
- EXT-RULE-0223: When fusing reparam blocks, validate numerical equivalence within tolerance.
- EXT-RULE-0224: Prefer depthwise conv for spatial mixing when compute is constrained.
- EXT-RULE-0225: Use SiLU/Mish for gating blocks to stabilize multiplicative interactions.
- EXT-RULE-0226: Avoid global attention on >1MP images; prefer windows or local kernels.
- EXT-RULE-0227: When using SDPA, ensure q/k/v shapes are [B, heads, N, dim].
- EXT-RULE-0228: Compute attention scores in FP32 if using half precision in training.
- EXT-RULE-0229: Crop output after upsampling to match original H*scale, W*scale.
- EXT-RULE-0230: When using grid_sample, set align_corners=False for SR consistency.
- EXT-RULE-0231: Always register fixed kernels as buffers (requires_grad=False).
- EXT-RULE-0232: Do not let dynamic kernels exceed stable magnitude; apply sigmoid/tanh if needed.
- EXT-RULE-0233: Normalize positional bias tables if they cause divergence in early epochs.
- EXT-RULE-0234: Use checkpointing sparingly; verify speed vs memory trade-off.
- EXT-RULE-0235: Keep recursion depth small for MoE/recursive blocks to avoid instability.
- EXT-RULE-0236: Use group norm or RMSNorm for batch=1 or small batch regimes.
- EXT-RULE-0237: If using channels_last, ensure LayerNorm supports the layout.
- EXT-RULE-0238: Prefer separable conv for classical filters (sharp/blur) to reduce compute.
- EXT-RULE-0239: Avoid hard-coded device creation inside forward; use x.device.
- EXT-RULE-0240: Prefer pad + crop to handle arbitrary image sizes without artifacts.
- EXT-RULE-0241: Preserve feature alignment; avoid implicit reshapes that reorder pixels.
- EXT-RULE-0242: If using pixel unshuffle, adjust padding so that H and W are divisible by factor.
- EXT-RULE-0243: When fusing reparam blocks, validate numerical equivalence within tolerance.
- EXT-RULE-0244: Prefer depthwise conv for spatial mixing when compute is constrained.
- EXT-RULE-0245: Use SiLU/Mish for gating blocks to stabilize multiplicative interactions.
- EXT-RULE-0246: Avoid global attention on >1MP images; prefer windows or local kernels.
- EXT-RULE-0247: When using SDPA, ensure q/k/v shapes are [B, heads, N, dim].
- EXT-RULE-0248: Compute attention scores in FP32 if using half precision in training.
- EXT-RULE-0249: Crop output after upsampling to match original H*scale, W*scale.
- EXT-RULE-0250: When using grid_sample, set align_corners=False for SR consistency.
- EXT-RULE-0251: Always register fixed kernels as buffers (requires_grad=False).
- EXT-RULE-0252: Do not let dynamic kernels exceed stable magnitude; apply sigmoid/tanh if needed.
- EXT-RULE-0253: Normalize positional bias tables if they cause divergence in early epochs.
- EXT-RULE-0254: Use checkpointing sparingly; verify speed vs memory trade-off.
- EXT-RULE-0255: Keep recursion depth small for MoE/recursive blocks to avoid instability.
- EXT-RULE-0256: Use group norm or RMSNorm for batch=1 or small batch regimes.
- EXT-RULE-0257: If using channels_last, ensure LayerNorm supports the layout.
- EXT-RULE-0258: Prefer separable conv for classical filters (sharp/blur) to reduce compute.
- EXT-RULE-0259: Avoid hard-coded device creation inside forward; use x.device.
- EXT-RULE-0260: Prefer pad + crop to handle arbitrary image sizes without artifacts.
- EXT-RULE-0261: Preserve feature alignment; avoid implicit reshapes that reorder pixels.
- EXT-RULE-0262: If using pixel unshuffle, adjust padding so that H and W are divisible by factor.
- EXT-RULE-0263: When fusing reparam blocks, validate numerical equivalence within tolerance.
- EXT-RULE-0264: Prefer depthwise conv for spatial mixing when compute is constrained.
- EXT-RULE-0265: Use SiLU/Mish for gating blocks to stabilize multiplicative interactions.
- EXT-RULE-0266: Avoid global attention on >1MP images; prefer windows or local kernels.
- EXT-RULE-0267: When using SDPA, ensure q/k/v shapes are [B, heads, N, dim].
- EXT-RULE-0268: Compute attention scores in FP32 if using half precision in training.
- EXT-RULE-0269: Crop output after upsampling to match original H*scale, W*scale.
- EXT-RULE-0270: When using grid_sample, set align_corners=False for SR consistency.
- EXT-RULE-0271: Always register fixed kernels as buffers (requires_grad=False).
- EXT-RULE-0272: Do not let dynamic kernels exceed stable magnitude; apply sigmoid/tanh if needed.
- EXT-RULE-0273: Normalize positional bias tables if they cause divergence in early epochs.
- EXT-RULE-0274: Use checkpointing sparingly; verify speed vs memory trade-off.
- EXT-RULE-0275: Keep recursion depth small for MoE/recursive blocks to avoid instability.
- EXT-RULE-0276: Use group norm or RMSNorm for batch=1 or small batch regimes.
- EXT-RULE-0277: If using channels_last, ensure LayerNorm supports the layout.
- EXT-RULE-0278: Prefer separable conv for classical filters (sharp/blur) to reduce compute.
- EXT-RULE-0279: Avoid hard-coded device creation inside forward; use x.device.
- EXT-RULE-0280: Prefer pad + crop to handle arbitrary image sizes without artifacts.
- EXT-RULE-0281: Preserve feature alignment; avoid implicit reshapes that reorder pixels.
- EXT-RULE-0282: If using pixel unshuffle, adjust padding so that H and W are divisible by factor.
- EXT-RULE-0283: When fusing reparam blocks, validate numerical equivalence within tolerance.
- EXT-RULE-0284: Prefer depthwise conv for spatial mixing when compute is constrained.
- EXT-RULE-0285: Use SiLU/Mish for gating blocks to stabilize multiplicative interactions.
- EXT-RULE-0286: Avoid global attention on >1MP images; prefer windows or local kernels.
- EXT-RULE-0287: When using SDPA, ensure q/k/v shapes are [B, heads, N, dim].
- EXT-RULE-0288: Compute attention scores in FP32 if using half precision in training.
- EXT-RULE-0289: Crop output after upsampling to match original H*scale, W*scale.
- EXT-RULE-0290: When using grid_sample, set align_corners=False for SR consistency.
- EXT-RULE-0291: Always register fixed kernels as buffers (requires_grad=False).
- EXT-RULE-0292: Do not let dynamic kernels exceed stable magnitude; apply sigmoid/tanh if needed.
- EXT-RULE-0293: Normalize positional bias tables if they cause divergence in early epochs.
- EXT-RULE-0294: Use checkpointing sparingly; verify speed vs memory trade-off.
- EXT-RULE-0295: Keep recursion depth small for MoE/recursive blocks to avoid instability.
- EXT-RULE-0296: Use group norm or RMSNorm for batch=1 or small batch regimes.
- EXT-RULE-0297: If using channels_last, ensure LayerNorm supports the layout.
- EXT-RULE-0298: Prefer separable conv for classical filters (sharp/blur) to reduce compute.
- EXT-RULE-0299: Avoid hard-coded device creation inside forward; use x.device.
- EXT-RULE-0300: Prefer pad + crop to handle arbitrary image sizes without artifacts.
- EXT-RULE-0301: Preserve feature alignment; avoid implicit reshapes that reorder pixels.
- EXT-RULE-0302: If using pixel unshuffle, adjust padding so that H and W are divisible by factor.
- EXT-RULE-0303: When fusing reparam blocks, validate numerical equivalence within tolerance.
- EXT-RULE-0304: Prefer depthwise conv for spatial mixing when compute is constrained.
- EXT-RULE-0305: Use SiLU/Mish for gating blocks to stabilize multiplicative interactions.
- EXT-RULE-0306: Avoid global attention on >1MP images; prefer windows or local kernels.
- EXT-RULE-0307: When using SDPA, ensure q/k/v shapes are [B, heads, N, dim].
- EXT-RULE-0308: Compute attention scores in FP32 if using half precision in training.
- EXT-RULE-0309: Crop output after upsampling to match original H*scale, W*scale.
- EXT-RULE-0310: When using grid_sample, set align_corners=False for SR consistency.
- EXT-RULE-0311: Always register fixed kernels as buffers (requires_grad=False).
- EXT-RULE-0312: Do not let dynamic kernels exceed stable magnitude; apply sigmoid/tanh if needed.
- EXT-RULE-0313: Normalize positional bias tables if they cause divergence in early epochs.
- EXT-RULE-0314: Use checkpointing sparingly; verify speed vs memory trade-off.
- EXT-RULE-0315: Keep recursion depth small for MoE/recursive blocks to avoid instability.
- EXT-RULE-0316: Use group norm or RMSNorm for batch=1 or small batch regimes.
- EXT-RULE-0317: If using channels_last, ensure LayerNorm supports the layout.
- EXT-RULE-0318: Prefer separable conv for classical filters (sharp/blur) to reduce compute.
- EXT-RULE-0319: Avoid hard-coded device creation inside forward; use x.device.
- EXT-RULE-0320: Prefer pad + crop to handle arbitrary image sizes without artifacts.
- EXT-RULE-0321: Preserve feature alignment; avoid implicit reshapes that reorder pixels.
- EXT-RULE-0322: If using pixel unshuffle, adjust padding so that H and W are divisible by factor.
- EXT-RULE-0323: When fusing reparam blocks, validate numerical equivalence within tolerance.
- EXT-RULE-0324: Prefer depthwise conv for spatial mixing when compute is constrained.
- EXT-RULE-0325: Use SiLU/Mish for gating blocks to stabilize multiplicative interactions.
- EXT-RULE-0326: Avoid global attention on >1MP images; prefer windows or local kernels.
- EXT-RULE-0327: When using SDPA, ensure q/k/v shapes are [B, heads, N, dim].
- EXT-RULE-0328: Compute attention scores in FP32 if using half precision in training.
- EXT-RULE-0329: Crop output after upsampling to match original H*scale, W*scale.
- EXT-RULE-0330: When using grid_sample, set align_corners=False for SR consistency.
- EXT-RULE-0331: Always register fixed kernels as buffers (requires_grad=False).
- EXT-RULE-0332: Do not let dynamic kernels exceed stable magnitude; apply sigmoid/tanh if needed.
- EXT-RULE-0333: Normalize positional bias tables if they cause divergence in early epochs.
- EXT-RULE-0334: Use checkpointing sparingly; verify speed vs memory trade-off.
- EXT-RULE-0335: Keep recursion depth small for MoE/recursive blocks to avoid instability.
- EXT-RULE-0336: Use group norm or RMSNorm for batch=1 or small batch regimes.
- EXT-RULE-0337: If using channels_last, ensure LayerNorm supports the layout.
- EXT-RULE-0338: Prefer separable conv for classical filters (sharp/blur) to reduce compute.
- EXT-RULE-0339: Avoid hard-coded device creation inside forward; use x.device.
- EXT-RULE-0340: Prefer pad + crop to handle arbitrary image sizes without artifacts.
- EXT-RULE-0341: Preserve feature alignment; avoid implicit reshapes that reorder pixels.
- EXT-RULE-0342: If using pixel unshuffle, adjust padding so that H and W are divisible by factor.
- EXT-RULE-0343: When fusing reparam blocks, validate numerical equivalence within tolerance.
- EXT-RULE-0344: Prefer depthwise conv for spatial mixing when compute is constrained.
- EXT-RULE-0345: Use SiLU/Mish for gating blocks to stabilize multiplicative interactions.
- EXT-RULE-0346: Avoid global attention on >1MP images; prefer windows or local kernels.
- EXT-RULE-0347: When using SDPA, ensure q/k/v shapes are [B, heads, N, dim].
- EXT-RULE-0348: Compute attention scores in FP32 if using half precision in training.
- EXT-RULE-0349: Crop output after upsampling to match original H*scale, W*scale.
- EXT-RULE-0350: When using grid_sample, set align_corners=False for SR consistency.
- EXT-RULE-0351: Always register fixed kernels as buffers (requires_grad=False).
- EXT-RULE-0352: Do not let dynamic kernels exceed stable magnitude; apply sigmoid/tanh if needed.
- EXT-RULE-0353: Normalize positional bias tables if they cause divergence in early epochs.
- EXT-RULE-0354: Use checkpointing sparingly; verify speed vs memory trade-off.
- EXT-RULE-0355: Keep recursion depth small for MoE/recursive blocks to avoid instability.
- EXT-RULE-0356: Use group norm or RMSNorm for batch=1 or small batch regimes.
- EXT-RULE-0357: If using channels_last, ensure LayerNorm supports the layout.
- EXT-RULE-0358: Prefer separable conv for classical filters (sharp/blur) to reduce compute.
- EXT-RULE-0359: Avoid hard-coded device creation inside forward; use x.device.
- EXT-RULE-0360: Prefer pad + crop to handle arbitrary image sizes without artifacts.
- EXT-RULE-0361: Preserve feature alignment; avoid implicit reshapes that reorder pixels.
- EXT-RULE-0362: If using pixel unshuffle, adjust padding so that H and W are divisible by factor.
- EXT-RULE-0363: When fusing reparam blocks, validate numerical equivalence within tolerance.
- EXT-RULE-0364: Prefer depthwise conv for spatial mixing when compute is constrained.
- EXT-RULE-0365: Use SiLU/Mish for gating blocks to stabilize multiplicative interactions.
- EXT-RULE-0366: Avoid global attention on >1MP images; prefer windows or local kernels.
- EXT-RULE-0367: When using SDPA, ensure q/k/v shapes are [B, heads, N, dim].
- EXT-RULE-0368: Compute attention scores in FP32 if using half precision in training.
- EXT-RULE-0369: Crop output after upsampling to match original H*scale, W*scale.
- EXT-RULE-0370: When using grid_sample, set align_corners=False for SR consistency.
- EXT-RULE-0371: Always register fixed kernels as buffers (requires_grad=False).
- EXT-RULE-0372: Do not let dynamic kernels exceed stable magnitude; apply sigmoid/tanh if needed.
- EXT-RULE-0373: Normalize positional bias tables if they cause divergence in early epochs.
- EXT-RULE-0374: Use checkpointing sparingly; verify speed vs memory trade-off.
- EXT-RULE-0375: Keep recursion depth small for MoE/recursive blocks to avoid instability.
- EXT-RULE-0376: Use group norm or RMSNorm for batch=1 or small batch regimes.
- EXT-RULE-0377: If using channels_last, ensure LayerNorm supports the layout.
- EXT-RULE-0378: Prefer separable conv for classical filters (sharp/blur) to reduce compute.
- EXT-RULE-0379: Avoid hard-coded device creation inside forward; use x.device.
- EXT-RULE-0380: Prefer pad + crop to handle arbitrary image sizes without artifacts.
- EXT-RULE-0381: Preserve feature alignment; avoid implicit reshapes that reorder pixels.
- EXT-RULE-0382: If using pixel unshuffle, adjust padding so that H and W are divisible by factor.
- EXT-RULE-0383: When fusing reparam blocks, validate numerical equivalence within tolerance.
- EXT-RULE-0384: Prefer depthwise conv for spatial mixing when compute is constrained.
- EXT-RULE-0385: Use SiLU/Mish for gating blocks to stabilize multiplicative interactions.
- EXT-RULE-0386: Avoid global attention on >1MP images; prefer windows or local kernels.
- EXT-RULE-0387: When using SDPA, ensure q/k/v shapes are [B, heads, N, dim].
- EXT-RULE-0388: Compute attention scores in FP32 if using half precision in training.
- EXT-RULE-0389: Crop output after upsampling to match original H*scale, W*scale.
- EXT-RULE-0390: When using grid_sample, set align_corners=False for SR consistency.
- EXT-RULE-0391: Always register fixed kernels as buffers (requires_grad=False).
- EXT-RULE-0392: Do not let dynamic kernels exceed stable magnitude; apply sigmoid/tanh if needed.
- EXT-RULE-0393: Normalize positional bias tables if they cause divergence in early epochs.
- EXT-RULE-0394: Use checkpointing sparingly; verify speed vs memory trade-off.
- EXT-RULE-0395: Keep recursion depth small for MoE/recursive blocks to avoid instability.
- EXT-RULE-0396: Use group norm or RMSNorm for batch=1 or small batch regimes.
- EXT-RULE-0397: If using channels_last, ensure LayerNorm supports the layout.
- EXT-RULE-0398: Prefer separable conv for classical filters (sharp/blur) to reduce compute.
- EXT-RULE-0399: Avoid hard-coded device creation inside forward; use x.device.
- EXT-RULE-0400: Prefer pad + crop to handle arbitrary image sizes without artifacts.
- EXT-RULE-0401: Preserve feature alignment; avoid implicit reshapes that reorder pixels.
- EXT-RULE-0402: If using pixel unshuffle, adjust padding so that H and W are divisible by factor.
- EXT-RULE-0403: When fusing reparam blocks, validate numerical equivalence within tolerance.
- EXT-RULE-0404: Prefer depthwise conv for spatial mixing when compute is constrained.
- EXT-RULE-0405: Use SiLU/Mish for gating blocks to stabilize multiplicative interactions.
- EXT-RULE-0406: Avoid global attention on >1MP images; prefer windows or local kernels.
- EXT-RULE-0407: When using SDPA, ensure q/k/v shapes are [B, heads, N, dim].
- EXT-RULE-0408: Compute attention scores in FP32 if using half precision in training.
- EXT-RULE-0409: Crop output after upsampling to match original H*scale, W*scale.
- EXT-RULE-0410: When using grid_sample, set align_corners=False for SR consistency.
- EXT-RULE-0411: Always register fixed kernels as buffers (requires_grad=False).
- EXT-RULE-0412: Do not let dynamic kernels exceed stable magnitude; apply sigmoid/tanh if needed.
- EXT-RULE-0413: Normalize positional bias tables if they cause divergence in early epochs.
- EXT-RULE-0414: Use checkpointing sparingly; verify speed vs memory trade-off.
- EXT-RULE-0415: Keep recursion depth small for MoE/recursive blocks to avoid instability.
- EXT-RULE-0416: Use group norm or RMSNorm for batch=1 or small batch regimes.
- EXT-RULE-0417: If using channels_last, ensure LayerNorm supports the layout.
- EXT-RULE-0418: Prefer separable conv for classical filters (sharp/blur) to reduce compute.
- EXT-RULE-0419: Avoid hard-coded device creation inside forward; use x.device.
- EXT-RULE-0420: Prefer pad + crop to handle arbitrary image sizes without artifacts.
- EXT-RULE-0421: Preserve feature alignment; avoid implicit reshapes that reorder pixels.
- EXT-RULE-0422: If using pixel unshuffle, adjust padding so that H and W are divisible by factor.
- EXT-RULE-0423: When fusing reparam blocks, validate numerical equivalence within tolerance.
- EXT-RULE-0424: Prefer depthwise conv for spatial mixing when compute is constrained.
- EXT-RULE-0425: Use SiLU/Mish for gating blocks to stabilize multiplicative interactions.
- EXT-RULE-0426: Avoid global attention on >1MP images; prefer windows or local kernels.
- EXT-RULE-0427: When using SDPA, ensure q/k/v shapes are [B, heads, N, dim].
- EXT-RULE-0428: Compute attention scores in FP32 if using half precision in training.
- EXT-RULE-0429: Crop output after upsampling to match original H*scale, W*scale.
- EXT-RULE-0430: When using grid_sample, set align_corners=False for SR consistency.
- EXT-RULE-0431: Always register fixed kernels as buffers (requires_grad=False).
- EXT-RULE-0432: Do not let dynamic kernels exceed stable magnitude; apply sigmoid/tanh if needed.
- EXT-RULE-0433: Normalize positional bias tables if they cause divergence in early epochs.
- EXT-RULE-0434: Use checkpointing sparingly; verify speed vs memory trade-off.
- EXT-RULE-0435: Keep recursion depth small for MoE/recursive blocks to avoid instability.
- EXT-RULE-0436: Use group norm or RMSNorm for batch=1 or small batch regimes.
- EXT-RULE-0437: If using channels_last, ensure LayerNorm supports the layout.
- EXT-RULE-0438: Prefer separable conv for classical filters (sharp/blur) to reduce compute.
- EXT-RULE-0439: Avoid hard-coded device creation inside forward; use x.device.
- EXT-RULE-0440: Prefer pad + crop to handle arbitrary image sizes without artifacts.
- EXT-RULE-0441: Preserve feature alignment; avoid implicit reshapes that reorder pixels.
- EXT-RULE-0442: If using pixel unshuffle, adjust padding so that H and W are divisible by factor.
- EXT-RULE-0443: When fusing reparam blocks, validate numerical equivalence within tolerance.
- EXT-RULE-0444: Prefer depthwise conv for spatial mixing when compute is constrained.
- EXT-RULE-0445: Use SiLU/Mish for gating blocks to stabilize multiplicative interactions.
- EXT-RULE-0446: Avoid global attention on >1MP images; prefer windows or local kernels.
- EXT-RULE-0447: When using SDPA, ensure q/k/v shapes are [B, heads, N, dim].
- EXT-RULE-0448: Compute attention scores in FP32 if using half precision in training.
- EXT-RULE-0449: Crop output after upsampling to match original H*scale, W*scale.
- EXT-RULE-0450: When using grid_sample, set align_corners=False for SR consistency.
- EXT-RULE-0451: Always register fixed kernels as buffers (requires_grad=False).
- EXT-RULE-0452: Do not let dynamic kernels exceed stable magnitude; apply sigmoid/tanh if needed.
- EXT-RULE-0453: Normalize positional bias tables if they cause divergence in early epochs.
- EXT-RULE-0454: Use checkpointing sparingly; verify speed vs memory trade-off.
- EXT-RULE-0455: Keep recursion depth small for MoE/recursive blocks to avoid instability.
- EXT-RULE-0456: Use group norm or RMSNorm for batch=1 or small batch regimes.
- EXT-RULE-0457: If using channels_last, ensure LayerNorm supports the layout.
- EXT-RULE-0458: Prefer separable conv for classical filters (sharp/blur) to reduce compute.
- EXT-RULE-0459: Avoid hard-coded device creation inside forward; use x.device.
- EXT-RULE-0460: Prefer pad + crop to handle arbitrary image sizes without artifacts.
- EXT-RULE-0461: Preserve feature alignment; avoid implicit reshapes that reorder pixels.
- EXT-RULE-0462: If using pixel unshuffle, adjust padding so that H and W are divisible by factor.
- EXT-RULE-0463: When fusing reparam blocks, validate numerical equivalence within tolerance.
- EXT-RULE-0464: Prefer depthwise conv for spatial mixing when compute is constrained.
- EXT-RULE-0465: Use SiLU/Mish for gating blocks to stabilize multiplicative interactions.
- EXT-RULE-0466: Avoid global attention on >1MP images; prefer windows or local kernels.
- EXT-RULE-0467: When using SDPA, ensure q/k/v shapes are [B, heads, N, dim].
- EXT-RULE-0468: Compute attention scores in FP32 if using half precision in training.
- EXT-RULE-0469: Crop output after upsampling to match original H*scale, W*scale.
- EXT-RULE-0470: When using grid_sample, set align_corners=False for SR consistency.
- EXT-RULE-0471: Always register fixed kernels as buffers (requires_grad=False).
- EXT-RULE-0472: Do not let dynamic kernels exceed stable magnitude; apply sigmoid/tanh if needed.
- EXT-RULE-0473: Normalize positional bias tables if they cause divergence in early epochs.
- EXT-RULE-0474: Use checkpointing sparingly; verify speed vs memory trade-off.
- EXT-RULE-0475: Keep recursion depth small for MoE/recursive blocks to avoid instability.
- EXT-RULE-0476: Use group norm or RMSNorm for batch=1 or small batch regimes.
- EXT-RULE-0477: If using channels_last, ensure LayerNorm supports the layout.
- EXT-RULE-0478: Prefer separable conv for classical filters (sharp/blur) to reduce compute.
- EXT-RULE-0479: Avoid hard-coded device creation inside forward; use x.device.
- EXT-RULE-0480: Prefer pad + crop to handle arbitrary image sizes without artifacts.
- EXT-RULE-0481: Preserve feature alignment; avoid implicit reshapes that reorder pixels.
- EXT-RULE-0482: If using pixel unshuffle, adjust padding so that H and W are divisible by factor.
- EXT-RULE-0483: When fusing reparam blocks, validate numerical equivalence within tolerance.
- EXT-RULE-0484: Prefer depthwise conv for spatial mixing when compute is constrained.
- EXT-RULE-0485: Use SiLU/Mish for gating blocks to stabilize multiplicative interactions.
- EXT-RULE-0486: Avoid global attention on >1MP images; prefer windows or local kernels.
- EXT-RULE-0487: When using SDPA, ensure q/k/v shapes are [B, heads, N, dim].
- EXT-RULE-0488: Compute attention scores in FP32 if using half precision in training.
- EXT-RULE-0489: Crop output after upsampling to match original H*scale, W*scale.
- EXT-RULE-0490: When using grid_sample, set align_corners=False for SR consistency.
- EXT-RULE-0491: Always register fixed kernels as buffers (requires_grad=False).
- EXT-RULE-0492: Do not let dynamic kernels exceed stable magnitude; apply sigmoid/tanh if needed.
- EXT-RULE-0493: Normalize positional bias tables if they cause divergence in early epochs.
- EXT-RULE-0494: Use checkpointing sparingly; verify speed vs memory trade-off.
- EXT-RULE-0495: Keep recursion depth small for MoE/recursive blocks to avoid instability.
- EXT-RULE-0496: Use group norm or RMSNorm for batch=1 or small batch regimes.
- EXT-RULE-0497: If using channels_last, ensure LayerNorm supports the layout.
- EXT-RULE-0498: Prefer separable conv for classical filters (sharp/blur) to reduce compute.
- EXT-RULE-0499: Avoid hard-coded device creation inside forward; use x.device.
- EXT-RULE-0500: Prefer pad + crop to handle arbitrary image sizes without artifacts.
- EXT-RULE-0501: Preserve feature alignment; avoid implicit reshapes that reorder pixels.
- EXT-RULE-0502: If using pixel unshuffle, adjust padding so that H and W are divisible by factor.
- EXT-RULE-0503: When fusing reparam blocks, validate numerical equivalence within tolerance.
- EXT-RULE-0504: Prefer depthwise conv for spatial mixing when compute is constrained.
- EXT-RULE-0505: Use SiLU/Mish for gating blocks to stabilize multiplicative interactions.
- EXT-RULE-0506: Avoid global attention on >1MP images; prefer windows or local kernels.
- EXT-RULE-0507: When using SDPA, ensure q/k/v shapes are [B, heads, N, dim].
- EXT-RULE-0508: Compute attention scores in FP32 if using half precision in training.
- EXT-RULE-0509: Crop output after upsampling to match original H*scale, W*scale.
- EXT-RULE-0510: When using grid_sample, set align_corners=False for SR consistency.
- EXT-RULE-0511: Always register fixed kernels as buffers (requires_grad=False).
- EXT-RULE-0512: Do not let dynamic kernels exceed stable magnitude; apply sigmoid/tanh if needed.
- EXT-RULE-0513: Normalize positional bias tables if they cause divergence in early epochs.
- EXT-RULE-0514: Use checkpointing sparingly; verify speed vs memory trade-off.
- EXT-RULE-0515: Keep recursion depth small for MoE/recursive blocks to avoid instability.
- EXT-RULE-0516: Use group norm or RMSNorm for batch=1 or small batch regimes.
- EXT-RULE-0517: If using channels_last, ensure LayerNorm supports the layout.
- EXT-RULE-0518: Prefer separable conv for classical filters (sharp/blur) to reduce compute.
- EXT-RULE-0519: Avoid hard-coded device creation inside forward; use x.device.
- EXT-RULE-0520: Prefer pad + crop to handle arbitrary image sizes without artifacts.
- EXT-RULE-0521: Preserve feature alignment; avoid implicit reshapes that reorder pixels.
- EXT-RULE-0522: If using pixel unshuffle, adjust padding so that H and W are divisible by factor.
- EXT-RULE-0523: When fusing reparam blocks, validate numerical equivalence within tolerance.
- EXT-RULE-0524: Prefer depthwise conv for spatial mixing when compute is constrained.
- EXT-RULE-0525: Use SiLU/Mish for gating blocks to stabilize multiplicative interactions.
- EXT-RULE-0526: Avoid global attention on >1MP images; prefer windows or local kernels.
- EXT-RULE-0527: When using SDPA, ensure q/k/v shapes are [B, heads, N, dim].
- EXT-RULE-0528: Compute attention scores in FP32 if using half precision in training.
- EXT-RULE-0529: Crop output after upsampling to match original H*scale, W*scale.
- EXT-RULE-0530: When using grid_sample, set align_corners=False for SR consistency.
- EXT-RULE-0531: Always register fixed kernels as buffers (requires_grad=False).
- EXT-RULE-0532: Do not let dynamic kernels exceed stable magnitude; apply sigmoid/tanh if needed.
- EXT-RULE-0533: Normalize positional bias tables if they cause divergence in early epochs.
- EXT-RULE-0534: Use checkpointing sparingly; verify speed vs memory trade-off.
- EXT-RULE-0535: Keep recursion depth small for MoE/recursive blocks to avoid instability.
- EXT-RULE-0536: Use group norm or RMSNorm for batch=1 or small batch regimes.
- EXT-RULE-0537: If using channels_last, ensure LayerNorm supports the layout.
- EXT-RULE-0538: Prefer separable conv for classical filters (sharp/blur) to reduce compute.
- EXT-RULE-0539: Avoid hard-coded device creation inside forward; use x.device.
- EXT-RULE-0540: Prefer pad + crop to handle arbitrary image sizes without artifacts.
- EXT-RULE-0541: Preserve feature alignment; avoid implicit reshapes that reorder pixels.
- EXT-RULE-0542: If using pixel unshuffle, adjust padding so that H and W are divisible by factor.
- EXT-RULE-0543: When fusing reparam blocks, validate numerical equivalence within tolerance.
- EXT-RULE-0544: Prefer depthwise conv for spatial mixing when compute is constrained.
- EXT-RULE-0545: Use SiLU/Mish for gating blocks to stabilize multiplicative interactions.
- EXT-RULE-0546: Avoid global attention on >1MP images; prefer windows or local kernels.
- EXT-RULE-0547: When using SDPA, ensure q/k/v shapes are [B, heads, N, dim].
- EXT-RULE-0548: Compute attention scores in FP32 if using half precision in training.
- EXT-RULE-0549: Crop output after upsampling to match original H*scale, W*scale.
- EXT-RULE-0550: When using grid_sample, set align_corners=False for SR consistency.
- EXT-RULE-0551: Always register fixed kernels as buffers (requires_grad=False).
- EXT-RULE-0552: Do not let dynamic kernels exceed stable magnitude; apply sigmoid/tanh if needed.
- EXT-RULE-0553: Normalize positional bias tables if they cause divergence in early epochs.
- EXT-RULE-0554: Use checkpointing sparingly; verify speed vs memory trade-off.
- EXT-RULE-0555: Keep recursion depth small for MoE/recursive blocks to avoid instability.
- EXT-RULE-0556: Use group norm or RMSNorm for batch=1 or small batch regimes.
- EXT-RULE-0557: If using channels_last, ensure LayerNorm supports the layout.
- EXT-RULE-0558: Prefer separable conv for classical filters (sharp/blur) to reduce compute.
- EXT-RULE-0559: Avoid hard-coded device creation inside forward; use x.device.
- EXT-RULE-0560: Prefer pad + crop to handle arbitrary image sizes without artifacts.
- EXT-RULE-0561: Preserve feature alignment; avoid implicit reshapes that reorder pixels.
- EXT-RULE-0562: If using pixel unshuffle, adjust padding so that H and W are divisible by factor.
- EXT-RULE-0563: When fusing reparam blocks, validate numerical equivalence within tolerance.
- EXT-RULE-0564: Prefer depthwise conv for spatial mixing when compute is constrained.
- EXT-RULE-0565: Use SiLU/Mish for gating blocks to stabilize multiplicative interactions.
- EXT-RULE-0566: Avoid global attention on >1MP images; prefer windows or local kernels.
- EXT-RULE-0567: When using SDPA, ensure q/k/v shapes are [B, heads, N, dim].
- EXT-RULE-0568: Compute attention scores in FP32 if using half precision in training.
- EXT-RULE-0569: Crop output after upsampling to match original H*scale, W*scale.
- EXT-RULE-0570: When using grid_sample, set align_corners=False for SR consistency.
- EXT-RULE-0571: Always register fixed kernels as buffers (requires_grad=False).
- EXT-RULE-0572: Do not let dynamic kernels exceed stable magnitude; apply sigmoid/tanh if needed.
- EXT-RULE-0573: Normalize positional bias tables if they cause divergence in early epochs.
- EXT-RULE-0574: Use checkpointing sparingly; verify speed vs memory trade-off.
- EXT-RULE-0575: Keep recursion depth small for MoE/recursive blocks to avoid instability.
- EXT-RULE-0576: Use group norm or RMSNorm for batch=1 or small batch regimes.
- EXT-RULE-0577: If using channels_last, ensure LayerNorm supports the layout.
- EXT-RULE-0578: Prefer separable conv for classical filters (sharp/blur) to reduce compute.
- EXT-RULE-0579: Avoid hard-coded device creation inside forward; use x.device.
- EXT-RULE-0580: Prefer pad + crop to handle arbitrary image sizes without artifacts.
- EXT-RULE-0581: Preserve feature alignment; avoid implicit reshapes that reorder pixels.
- EXT-RULE-0582: If using pixel unshuffle, adjust padding so that H and W are divisible by factor.
- EXT-RULE-0583: When fusing reparam blocks, validate numerical equivalence within tolerance.
- EXT-RULE-0584: Prefer depthwise conv for spatial mixing when compute is constrained.
- EXT-RULE-0585: Use SiLU/Mish for gating blocks to stabilize multiplicative interactions.
- EXT-RULE-0586: Avoid global attention on >1MP images; prefer windows or local kernels.
- EXT-RULE-0587: When using SDPA, ensure q/k/v shapes are [B, heads, N, dim].
- EXT-RULE-0588: Compute attention scores in FP32 if using half precision in training.
- EXT-RULE-0589: Crop output after upsampling to match original H*scale, W*scale.
- EXT-RULE-0590: When using grid_sample, set align_corners=False for SR consistency.
- EXT-RULE-0591: Always register fixed kernels as buffers (requires_grad=False).
- EXT-RULE-0592: Do not let dynamic kernels exceed stable magnitude; apply sigmoid/tanh if needed.
- EXT-RULE-0593: Normalize positional bias tables if they cause divergence in early epochs.
- EXT-RULE-0594: Use checkpointing sparingly; verify speed vs memory trade-off.
- EXT-RULE-0595: Keep recursion depth small for MoE/recursive blocks to avoid instability.
- EXT-RULE-0596: Use group norm or RMSNorm for batch=1 or small batch regimes.
- EXT-RULE-0597: If using channels_last, ensure LayerNorm supports the layout.
- EXT-RULE-0598: Prefer separable conv for classical filters (sharp/blur) to reduce compute.
- EXT-RULE-0599: Avoid hard-coded device creation inside forward; use x.device.
- EXT-RULE-0600: Prefer pad + crop to handle arbitrary image sizes without artifacts.