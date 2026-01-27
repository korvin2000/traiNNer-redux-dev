# TraiNNer-Redux, BasicSR, NeoSR aio image restoration, super-resolution / SISR architecture guide and how-to

---

## Integration Guide (for Codex): Implementing a Super-Resolution Generator Architecture for **traiNNer-redux / BasicSR / NeoSR**

This document is a **rules + FAQ** guide for implementing a **generator architecture** (`network_g`) that integrates cleanly into **BasicSR-family** training stacks (BasicSR, NeoSR, traiNNer-redux). It is intended to be used as a set of strict implementation instructions for **Codex**.

---

## 0) Mental Model: What the Trainer Expects

BasicSR-family trainers generally follow the same contract:

1.  Parse a YAML config.
2.  Build the generator via a **registry lookup** (`ARCH_REGISTRY`) using `network_g.type`.
3.  Call the generator as a standard `nn.Module` in train/val loops (typically `net_g(lq)`).
4.  Own everything else (datasets, losses, optimizers, EMA, validation, checkpointing, tiling utilities).

**Implication:** the architecture should be “boring and stable”: predictable `__init__`, predictable `forward`, deterministic validation unless explicitly configured.

### 0.1 traiNNer-redux call chain (train/test)

High-level flow in traiNNer-redux:

1. `train.py` / `test.py` parse YAML → `Config.load_config_from_file`.
2. `build_model(opt)` chooses `SRModel` (default) or other model wrappers. 【F:traiNNer/models/__init__.py†L1-L42】
3. `SRModel` builds `network_g` via the arch registry (`build_network`) and calls `net_g(lq)` inside mixed-precision `autocast`. 【F:traiNNer/models/sr_model.py†L462-L491】
4. Validation uses `net_g_ema` if present and can switch to tiled inference. 【F:traiNNer/models/sr_model.py†L940-L958】

**Implication:** the generator only needs a standard `forward(lq)`; the trainer handles AMP, EMA, tiling, logging, datasets, and checkpoints.

### 0.2 The scale contract is wired into dataset setup

Training dataloaders derive `gt_size` and `lq_size` from the **top-level** `scale` and explicitly require exactly one of those sizes to be defined. Your architecture must respect that `scale` value (e.g., output size must be `H*scale, W*scale` for SR). 

---

## 1) Registration & Auto-Discovery (ARCH_REGISTRY)

### 1.1 Registration is mandatory

Your architecture must be registered with `ARCH_REGISTRY` so YAML can instantiate it by name. BasicSR-Examples explicitly recommends adding `@ARCH_REGISTRY.register()` before the class. [GitHub](https://github.com/xinntao/BasicSR-examples)

**Pattern A — register the class (recommended):**

```text
pythonfrom torch import nn
from basicsr.utils.registry import ARCH_REGISTRY  # or traiNNer.utils.registry

@ARCH_REGISTRY.register()
class MyGenerator(nn.Module):
    def __init__(self, **opt):
        super().__init__()
        ...
    def forward(self, x):
        ...
```

**Pattern B — register a factory function:**

```text
python@ARCH_REGISTRY.register()
def my_generator(**opt):
    return MyGenerator(**opt)
```

### 1.2 File naming conventions (auto import)

If the framework uses auto-import scanning (common in BasicSR-derived repos), the architecture file should end with `_arch.py`. BasicSR-Examples explicitly calls this out. [GitHub](https://github.com/xinntao/BasicSR-examples)

**Rule:** put your file under `traiNNer/archs/` and name it `*_arch.py`, because traiNNer-redux auto-imports every file ending in `_arch.py` from that folder at startup. 【F:traiNNer/archs/__init__.py†L11-L25】

### 1.2.2 Registry routing is not optional in traiNNer-redux

Because `build_network()` only uses the registry lookups (SPANDREL → ARCH → TESTARCH), **direct class imports are not used**. If your class is not registered, it will not be instantiated by YAML.

### 1.3 YAML usage

BasicSR-Examples shows the standard YAML pattern: `network_g.type` is the class name, and remaining keys become `__init__` kwargs. [GitHub](https://github.com/xinntao/BasicSR-examples)

```text
yamlnetwork_g:
  type: MyGenerator
  num_in_ch: 3
  num_out_ch: 3
  upscale: 4
```

### 1.4 Registry import location is framework-specific (repo pattern)

In this repo, the registry import *varies by framework*:

- BasicSR-style models import from `basicsr.utils.registry`. (e.g., `PFT-SR/pft_arch.py`)  
- traiNNer-redux models import from `traiNNer.utils.registry`. (e.g., `flexnet/flexnet_arch.py`)  
- NeoSR models import from `neosr.utils.registry`. (e.g., `neosr/hma_arch.py`)  

**Rule:** keep the import path aligned with the trainer you target; the class name must match `network_g.type` in YAML, not the Python module path.

### 1.5 Multiple registries in traiNNer-redux

`build_network` resolves registries in this order:

1. `SPANDREL_REGISTRY` (external spandrel models)
2. `ARCH_REGISTRY` (traiNNer-redux native architectures)
3. `TESTARCH_REGISTRY` (test-only)【F:traiNNer/archs/__init__.py†L11-L57】

**Rule:** register in the correct registry for discoverability:

- Use `ARCH_REGISTRY` for in-repo architectures.
- Use `SPANDREL_REGISTRY` for spandrel-backed wrappers.
- Avoid `TESTARCH_REGISTRY` for production architectures.

---

## 2) `__init__` Contract (YAML-Driven, Backward-Compatible)

### 2.1 Must accept kwargs cleanly

Use `__init__(self, **opt)` or explicit keyword args with defaults, because the trainer passes YAML options as keyword args (as in BasicSR-Examples). [GitHub](https://github.com/xinntao/BasicSR-examples)

**Avoid**: positional-only args, required args with no defaults, or strict parsing that breaks older configs.

### 2.2 Keep init pure (no training-loop responsibilities)

`__init__` should:

- build submodules
- store config fields
- register buffers

It should **not**:

- load checkpoints
- read files
- allocate huge tensors
- set global seeds
- call `.to(device)` / `.cuda()` (trainer owns device placement)

### 2.3 Scale handling: prefer init-time scale

In traiNNer-redux, `scale` is a **top-level config option** (not necessarily inside `network_g`). [TraiNNer-Redux](https://trainner-redux.readthedocs.io/en/latest/config_reference.html)

`SRModel` injects the top-level `scale` into `network_g` kwargs (even if the YAML does not list it), so the constructor must accept it without error.

**Best practice:** treat each model instance as a single fixed scale:

```text
pythonself.scale = int(opt.get("scale", opt.get("upscale", 4)))
```

**Why:** traiNNer-redux derives training crop sizes using the **top-level** `scale`, and uses that same scale in tiled inference. Mismatched `scale` in your architecture will desynchronize dataset sizes and outputs. 

### 2.3.2 Preserve train/test model expectations (SRModel contract)

`SRModel` builds the generator with `build_network({**opt.network_g, "scale": opt.scale})` and will **always** call it as a standard module. It does not pass additional arguments to `forward`, so any optional inputs must be computed internally or handled by a wrapper.

### 2.4 Variants are commonly registered as factory functions

Many architectures in this repo expose multiple model sizes (tiny/small/medium/etc.) by **registering factory functions** rather than subclasses, which keeps YAML simple and avoids duplication. Examples include DIS variants in `arches/dis_arch.py` and NEXUS-Lite variants in `Nexus/nexus_lite_arch.py`.  

**Rule:** if your architecture family has several presets, prefer registered factory functions that return a base class with different hyperparameters.

### 2.5 Input-size constraints: document, assert, or pad

Transformer-style SR models often require spatial sizes to be divisible by a window or patch size. If your model has such constraints:

- **document the constraint** in the class docstring and in this guide’s YAML example
- **assert early** in `forward()` with a clear error (fail fast)
- **or** internal-pad/unpad deterministically (but keep it explicit and stable)

Do not rely on the trainer to magically fix shape constraints unless your model wrapper explicitly does so.

### 2.6 Avoid dynamic parameters or shape-dependent module creation

traiNNer-redux uses EMA and strict checkpoint loading. Dynamic module creation (e.g., creating layers inside `forward()`) breaks EMA weight shape consistency and `strict_load_g` behavior. Keep your module graph fixed in `__init__` for stable state dicts. 【F:traiNNer/models/sr_model.py†L835-L846】

---

## 3) `forward()` Contract (Do Not Fight the Trainer)

### 3.1 Forward signature is usually `forward(lq)`

Most SR training loops call the generator with one tensor input: `net_g(lq)`. Therefore, implement:

```text
pythondef forward(self, x: torch.Tensor) -> torch.Tensor:
    ...
```

If your internal architecture supports extra inputs (style vectors, noise maps, border masks), compute them internally or expose **separate** helper methods. Do not require them in the default `forward()`.

### 3.1.1 Pixel format conversions happen outside the net

`SRModel` converts inputs to the configured `input_pixel_format` before calling `net_g`, and converts the output back afterward. Your architecture should assume it receives tensors already in the expected pixel format and should not perform additional RGB↔Y conversions unless explicitly configured. 【F:traiNNer/models/sr_model.py†L462-L491】

### 3.2 Determinism rules (validation should be stable)

traiNNer-redux supports `manual_seed` and a `deterministic` option at config level. [TraiNNer-Redux](https://trainner-redux.readthedocs.io/en/latest/config_reference.html)  
Therefore:

- `eval()` should be deterministic by default
- stochastic branches should be gated by `self.training` or explicit flags

In validation/testing, traiNNer-redux switches to `net_g_ema` if present and runs inference under `torch.inference_mode()`. Avoid relying on training-only state or side effects in `forward()`. 【F:traiNNer/models/sr_model.py†L940-L958】

### 3.3 Return type

Return a **single Tensor** (SR result) unless your framework’s model wrapper explicitly expects a dict/tuple. When you need debug outputs, add an optional flag and keep default behavior returning a Tensor.

### 3.4 Window/padding safety: crop back after padding

Windowed transformer-style models often **pad input** to a multiple of `window_size`, then **crop back** to `H * upscale, W * upscale` at the end. NEXUS-Lite uses this exact pattern (pad → forward → crop).  

**Rule:** if you pad or extend the input, always crop the output back to the true scaled resolution to avoid border growth artifacts.


### 3.5 Validate scale and output shape when upsampling

Several in-repo modules guard against unsupported scale factors (e.g., `2^n` and `3` only) and implement PixelShuffle-based upsampling. This is a common failure point for config mistakes, so it is appropriate to:

- validate `scale` in `__init__`
- fail fast with a clear message if unsupported
- ensure `forward()` returns `NCHW` with spatial size multiplied by `scale`

This matches existing scale checks in upsampling helper modules used by SR architectures. 【F:OSRT/osrt_arch.py†L760-L775】

### 3.6 Shape, dtype, and device invariants (SR / restoration)

The majority of architectures in this repo are pure image-to-image nets and follow NCHW tensor semantics. Practical invariants to keep:

- input: `float32` or `float16/bfloat16` NCHW tensor on the active device
- output: same dtype/device, shape `(N, C, H*scale, W*scale)` for SR or `(N, C, H, W)` for restoration
- avoid implicit CPU tensors inside `forward()` (use buffers or device-aware ops)

If your model can accept non-RGB inputs, **bind it to explicit `num_in_ch`/`in_chans`** and avoid silently repeating channels.

### 3.7 Mixed precision compatibility (fp16/bf16)

Training and validation wrap `net_g(lq)` with `torch.autocast`, and `use_amp`/`amp_bf16` are standard config toggles. Ensure your layers and custom ops are AMP-safe. Avoid integer-only ops that silently upcast or unsupported dtypes. 【F:traiNNer/models/sr_model.py†L462-L491】【F:options/_templates/train/RCAN/RCAN_fidelity.yml†L1-L22】

### 3.8 Pixel format conversion happens outside the network / generator 

`SRModel` converts input/output pixel formats via `rgb2pixelformat_pt` / `pixelformat2rgb_pt` using config fields (`input_pixel_format`, `output_pixel_format`).  
**Rule:** implement the generator assuming the incoming tensor already matches the configured pixel format. Do not perform implicit RGB↔Y or other color transforms inside the architecture unless explicitly exposed as parameters.
In SRModel’s `test()` path, inputs are converted to the configured **pixel format** before the network, and outputs are converted back to RGB afterward. Do not build hard-coded RGB↔Y or pixel format conversions into `forward()` unless explicitly intended; the trainer already owns this conversion step.

---

## 4) Multi-Scale (x1/x2/x4): Recommended Strategy

### 4.1 Most compatible: one checkpoint per scale

Because the trainer typically calls `forward(lq)` without extra args, the cleanest approach is:

- train separate `scale=1`, `scale=2`, `scale=4` experiments/checkpoints

traiNNer-redux explicitly describes what `scale` means in the config reference. [TraiNNer-Redux](https://trainner-redux.readthedocs.io/en/latest/config_reference.html)

### 4.2 If you insist on a single multi-scale checkpoint

You will likely need a **custom model wrapper** (trainer “Model” class) that calls:

```text
pythonout = net_g(lq, scale=...)
```

because the default SR model loop won’t pass `scale`.

---

## 5) Tiling / “Chop” Inference (Keep It OUT of `forward()`)

### 5.1 Why tiling exists

Large images can exceed VRAM. Tiling crops input into tiles, runs `net_g(tile)`, and merges results.

RealESRGAN-style inference helpers commonly expose:

- `tile` (tile size)
- `tile_pad` (overlap padding to remove seams)
- `pre_pad` (global pre-padding)
- `half` (fp16 inference)  
  as documented in RealESRGANer utility code. [Hugging Face](https://huggingface.co/spaces/sczhou/CodeFormer/blob/refs%2Fpr%2F40/CodeFormer/basicsr/utils/realesrgan_utils.py)

### 5.2 Where tiling should live

**Rule:** do not embed tiling logic into the generator’s `forward()`.

Do one of:

- implement tiling in an **inference helper** / CLI
- implement tiling in the framework’s `Model.test()` wrapper
- optionally add `forward_tiled(...)` as an explicit method, but keep `forward()` tile-agnostic

### 5.3 traiNNer-redux validation tiling details

`SRModel` implements tiled validation inference with **batch size 1 only**, reflective padding, and overlap blending. It is enabled when `val.tile_size > 0`, and it auto-falls back to tiling on OOM with a forced `tile_size=256`.
**Rule:** do not assume the trainer will handle batch>1 tiling for you; if you need batch-tiling support, implement it outside `forward()` and document it.

### 5.4 Padding conventions (standard meaning)

- `pre_pad`: pad whole input before processing
- `tile_pad`: pad/overlap each tile; merge to remove tile borders  
  These concepts are explicitly described in the RealESRGANer helper docstring. [Hugging Face](https://huggingface.co/spaces/sczhou/CodeFormer/blob/refs%2Fpr%2F40/CodeFormer/basicsr/utils/realesrgan_utils.py)

### 5.5 traiNNer-redux tiled inference specifics
In traiNNer-redux, tiled inference is implemented in `SRModel.infer_tiled()` and is called only when `val.tile_size > 0`. It **asserts batch size = 1**, uses **reflect padding**, and merges tiles with a weight map. Architectures must therefore support `batch=1` inference and be safe under reflective padding/cropping.

`SRModel.infer_tiled` assumes batch size 1 and uses:

- `tile_size` and `tile_overlap` from the config
- reflect padding for tiles
- a cosine-like weighting map for seam blending
- output cropping back to `H*scale × W*scale`


**Architecture requirement:** the net must return a clean `NCHW` tensor for **every** tile and handle reflect-padded inputs without shape-dependent state.

---

## 6) Data Range, Mean, and Color Handling (Avoid Silent Mismatches)

### 6.1 Expose `img_range` / `rgb_mean` patterns when relevant

Many canonical BasicSR architectures expose parameters like `img_range` and `rgb_mean` (e.g., RCAN API docs show this style of constructor). [basicsr.readthedocs.io](https://basicsr.readthedocs.io/en/latest/api/basicsr.archs.rcan_arch.html)

**Rules:**

- Do not assume `0..255` unless the entire pipeline is fixed that way.
- If inputs are `0..1` floats, prefer `img_range=1.0`.
- Mean shifting should be optional/configurable.

**Repo pattern:** NEXUS-Lite registers a `mean` buffer and applies `img_range` scaling before and after the forward pass. This is the canonical BasicSR-style normalization pattern and is safe with DDP/device moves.  

### 6.2 Don’t silently change color space

traiNNer-redux has config-level pixel format options (input/output formats). [TraiNNer-Redux](https://trainner-redux.readthedocs.io/en/latest/config_reference.html)  
Your generator should not unexpectedly convert RGB↔Y unless explicitly configured and tested.

### 6.3 SR vs. restoration mode (scale=1)

Many architectures in this repository explicitly document that `upscale=1` is used for **denoising** or **compression artifact reduction**, while `upscale=2/3/4/8` targets SR. This allows a single architecture definition to serve both SR and restoration tasks with the same `forward(lq)` contract. 【F:OSRT/osrt_arch.py†L806-L832】

**Rule:** if your model supports restoration, make `upscale=1` a valid (and well-tested) path; do not hard-require PixelShuffle.

### 6.4 Pixel format and range invariants

traiNNer-redux allows configurable input/output pixel formats and performs the conversion outside the architecture. Keep your net focused on learning in the configured color space and respect the provided dtype/range; do **not** clamp or rescale unless explicitly required by the model design. 【F:traiNNer/models/sr_model.py†L462-L491】

---

## 7) Device Safety, Buffers, DDP, EMA

### 7.1 Register constant tensors as buffers

If your architecture uses fixed kernels (e.g., oriented line kernels), register them:

```text
pythonself.register_buffer("kernels", kernels, persistent=False)
```

This guarantees `.to(device)` moves them correctly and they participate properly in state dict behavior.

### 7.2 Avoid creating device-bound constants in `forward()`

Do not repeatedly allocate constant tensors in `forward()`. Precompute and register buffers when possible.

### 7.3 EMA friendliness

#### EMA is actively used in traiNNer-redux SRModel

`SRModel.test()` prefers `net_g_ema` when configured and switches back to training mode afterward. If your network has any training-time switches (e.g., stochastic layers), ensure `eval()` yields deterministic behavior and that EMA snapshots remain compatible with the same `state_dict` structure.

Inference helpers often prefer EMA weights when available (RealESRGANer loads `params_ema` if present). [Hugging Face](https://huggingface.co/spaces/sczhou/CodeFormer/blob/refs%2Fpr%2F40/CodeFormer/basicsr/utils/realesrgan_utils.py)  
Therefore:

- avoid dynamic module creation in forward
- keep stable parameter names/shapes

traiNNer-redux updates `net_g_ema` after generator optimizer steps, unless AMP scaling is skipped. This assumes parameter names and shapes are stable across iterations. 【F:traiNNer/models/sr_model.py†L835-L846】


### 7.4 Minimum spatial size constraints

Some architectures require minimum spatial dimensions (e.g., windowed attention). traiNNer-redux tracks these in `REQUIRE_32_HW` / `REQUIRE_64_HW` lists for training/inference guardrails.
*Rule:** if your model enforces a minimum H/W, add it to the appropriate set and document the constraint in the architecture docstring and template YAML.

---

## 8) Reproducibility & Performance Knobs Belong to the Framework

traiNNer-redux exposes top-level runtime controls like:

- `manual_seed`
- `deterministic`
- AMP (`use_amp`, bf16 options)
- channels-last memory format
- `use_compile` / torch.compile  
  These are explicitly documented in config reference. [TraiNNer-Redux](https://trainner-redux.readthedocs.io/en/latest/config_reference.html)

**Rule:** do not implement these policies inside the generator architecture; keep it a standard `nn.Module`.

### 8.1 Channels-last and memory format

The trainer can move inputs to `channels_last` memory format. Convolutions generally support it, but custom ops or fused CUDA kernels may not. When using `channels_last`, test that your architecture preserves correctness and performance; otherwise document incompatibility. 
SRModel enables AMP via `torch.autocast`, but will **disable fp16** (or fall back to bf16) if your `network_g.type` appears in `ARCHS_WITHOUT_FP16`. If your architecture is not safe in fp16, add its lowercase name to this list.

### 8.2 Channels-last performance list (optional)

Some architectures are known to be slower with channels-last memory format. When adding a model that regresses with `use_channels_last: true`, add it to `ARCHS_WITHOUT_CHANNELS_LAST` so option generation can avoid it.

---

## 9) Scale/upsampler specs observed in this repo

### 9.1 Supported scale set should be explicit

Many registered models **explicitly accept scales {1, 2, 3, 4}** and raise errors otherwise (e.g., DIS).  

**Rule:** validate the scale early (init-time) and fail with a clear error if unsupported. If your model is restoration-only, set `scale=1` and keep the output size identical to input.

Example (DIS): scale validation and scale-dependent upsampling in `__init__`. 【F:traiNNer/archs/dis_arch.py†L88-L152】

### 9.2 Upsampler choices are explicit and configured by name

Transformer-style SR models often expose **`upsampler`** string argument like `"pixelshuffle"`, `"pixelshuffledirect"`, or `"none"` (see NEXUS-Lite).
DRCT, for example, has an `upsampler` parameter in its constructor alongside `upscale` and `img_range`.  

**Rule:** map each upsampler string to a deterministic implementation, and keep the default `forward(lq)` signature independent from the choice.

---

## 10) Registry patterns found in this repo (useful for non-standard integration)

### 10.1 Multiple registries can be stacked

Some architectures register into more than one registry via stacked decorators (e.g., `ARCH_REGISTRY` and `SPANDREL_REGISTRY` in `arches/dis_arch.py`).  

**Rule:** if your model must be discoverable by multiple loaders, stack decorators in the same order.

### 10.2 Registry suffixes disambiguate names

`@ARCH_REGISTRY.register(suffix='traiNNer')` is used in `others/stylegan2_bilinear_arch.py` to avoid name clashes while keeping a recognizable base name.  

**Rule:** use suffixes sparingly and only for explicit name disambiguation across frameworks.

### 10.3 Optional registry fallback for standalone usage

`Nexus/nexus_lite_arch.py` wraps registry imports in a `try/except` and falls back to a dummy registry when traiNNer is unavailable.  

**Rule:** if you want the architecture file to be importable outside trainer repos, provide a small fallback registry that is a no-op decorator.

---

## 11) traiNNer-redux-specific architecture invariants (checklist + rationale)

| Invariant | Why it matters | Evidence |
| --- | --- | --- |
| `forward(lq)` → Tensor | SRModel invokes generator with one tensor input. | SRModel inference call uses `net_g(lq)`.【F:traiNNer/models/sr_model.py†L462-L491】 |
| Output spatial size = `H*scale × W*scale` (SR) | Tiled inference and metrics assume deterministic scaling. | Tiling constructs output shape using `opt.scale`.【F:traiNNer/models/sr_model.py†L849-L919】 |
| Use `opt.scale` as source of truth | Dataset crop sizes are derived from `scale`. | Training uses `opt.scale` for LQ/GT sizing.【F:train.py†L73-L121】 |
| AMP-safe operations | `torch.autocast` is enabled with fp16/bf16. | Autocast wraps generator calls; AMP config exists. 【F:traiNNer/models/sr_model.py†L462-L491】【F:options/_templates/train/RCAN/RCAN_fidelity.yml†L1-L22】 |
| EMA-friendly static parameters | EMA update assumes stable params each step. | EMA update in SRModel. 【F:traiNNer/models/sr_model.py†L835-L846】 |
| Batch size 1 compatibility for tiling | `infer_tiled` enforces `b==1`. | Tiling asserts batch size 1. 【F:traiNNer/models/sr_model.py†L857-L866】 |
| Tile-safe padding | Tiling uses reflect padding. | Reflect padding in tiled inference. 【F:traiNNer/models/sr_model.py†L871-L904】 |
| `pretrain_network_g` compatibility | Testing requires pretrained generator path. | test.py asserts `pretrain_network_g`. 【F:test.py†L17-L30】 |

---

## Recommended Implementation Pattern: “Wrapper Architecture” (Minimal Integration)

If your internal generator has a richer forward signature (e.g., `forward(x, scale, z_g, z_l, border_mask)`), do not change the trainer. Wrap the net so the trainer still calls `forward(lq)`.

```text
pythonfrom torch import nn
from traiNNer.utils.registry import ARCH_REGISTRY  # or basicsr.utils.registry

@ARCH_REGISTRY.register()
class HumArchWrapper(nn.Module):
    def __init__(self, **opt):
        super().__init__()
        # traiNNer-redux has top-level scale; configs often also include upscale.
        self.scale = int(opt.get("scale", opt.get("upscale", 4)))

        # parse your internal config here (keep defaults for backwards compat)
        # cfg = HumArchConfig(...)

        self.net = HumArch(cfg)

        # deterministic inference by default
        self.noise_mode = opt.get("noise_mode", "zero")  # "zero" | "rand"

    def forward(self, lq):
        # keep forward signature simple for trainer compatibility
        if (not self.training) or self.noise_mode == "zero":
            return self.net(lq, scale=self.scale, z_g=None, z_l=None, border_mask=None)
        # if stochastic detail is desired in train, generate z_l inside net or here (but keep API stable)
        return self.net(lq, scale=self.scale, z_g=None, z_l=None, border_mask=None)
```

---

## FAQ (Common Integration Failures)

### Q1: “YAML says `type: MyNet`, but framework can’t find it.”

**Cause:** class not registered or file not auto-imported.  
**Fix:** ensure `@ARCH_REGISTRY.register()` and file ends with `_arch.py` (auto-import pattern noted by BasicSR-Examples). [GitHub](https://github.com/xinntao/BasicSR-examples)

### Q2: “Trainer calls `net_g(lq)` but my model expects `scale/z_g/z_l`.”

**Fix:** compute them internally or use a wrapper that keeps forward signature `forward(lq)`.

### Q3: “GPU crash: CPU kernels used in conv.”

**Cause:** fixed kernels not moved to GPU.  
**Fix:** use `register_buffer()` and avoid CPU-only tensor creation in forward.

### Q4: “Validation is nondeterministic; metrics fluctuate.”

**Cause:** stochastic noise enabled in eval.  
**Fix:** gate noise by `self.training` or force `noise_mode='zero'` in eval; rely on framework seed/deterministic controls. [TraiNNer-Redux](https://trainner-redux.readthedocs.io/en/latest/config_reference.html)

### Q5: “Seams or halos in tiled inference.”

**Fix:** implement standard tiling with `tile_pad` overlap and correct padding/merge; do not put it in `forward()`. RealESRGANer documents standard tile parameters and intent. [Hugging Face](https://huggingface.co/spaces/sczhou/CodeFormer/blob/refs%2Fpr%2F40/CodeFormer/basicsr/utils/realesrgan_utils.py)

### Q6: “Outputs too dark/bright compared to other models.”

**Cause:** mismatch in `img_range` and mean subtraction.  
**Fix:** expose/configure `img_range`/`rgb_mean` style parameters; RCAN API illustrates this convention. [basicsr.readthedocs.io](https://basicsr.readthedocs.io/en/latest/api/basicsr.archs.rcan_arch.html)

### Q7: “I want x1/x2/x4 from one checkpoint.”

**Reality:** default SR loops won’t pass scale.  
**Fix:** train separate scales, or implement a custom Model wrapper that calls scale-specific branch.

### Q8: “Where should AMP / channels-last / torch.compile live?”

In traiNNer-redux these are config-level runtime features (`use_amp`, `use_channels_last`, `use_compile`, etc.). Keep them out of the net. [TraiNNer-Redux](https://trainner-redux.readthedocs.io/en/latest/config_reference.html)

### Q9: “My model fails with tile_size > 0.”

**Cause:** the architecture assumes full-image context, or has state that depends on absolute image size (e.g., positional embeddings without proper interpolation).  
**Fix:** add safe padding/positional embedding interpolation inside the model (not the trainer) so it can process tiles independently. Validate that `forward()` handles reflect-padded inputs and that outputs are cropped correctly. 【F:traiNNer/models/sr_model.py†L849-L919】

---

## Codex Checklist (Pre-PR)

- Architecture file ends with `_arch.py` and is in the auto-imported arch folder. [GitHub](https://github.com/xinntao/BasicSR-examples)
- `@ARCH_REGISTRY.register()` present (class or factory). [GitHub](https://github.com/xinntao/BasicSR-examples)
- `__init__(**opt)` parses YAML keys with defaults. [GitHub](https://github.com/xinntao/BasicSR-examples)
- `forward(lq)` exists, takes one tensor, returns one tensor.
- scale handled at init-time (prefer top-level `scale`). [TraiNNer-Redux](https://trainner-redux.readthedocs.io/en/latest/config_reference.html)
- non-parameter tensors are buffers; no CPU-only kernels in forward.
- eval deterministic (noise disabled in val). [TraiNNer-Redux](https://trainner-redux.readthedocs.io/en/latest/config_reference.html)
- tiling is outside forward; tile options follow standard meaning. [Hugging Face](https://huggingface.co/spaces/sczhou/CodeFormer/blob/refs%2Fpr%2F40/CodeFormer/basicsr/utils/realesrgan_utils.py)
- for AMP safety, add fp16-incompatible nets to `ARCHS_WITHOUT_FP16`.
- validate batch=1 tiled inference and reflect padding compatibility.

---

If you want, I can turn this guide into a **Codex “task prompt”** that tells it exactly what files to create inside a traiNNer-redux repo (e.g., `archs/humarch_arch.py` wrapper + YAML snippet), while keeping your internal all-in-one generator untouched.
