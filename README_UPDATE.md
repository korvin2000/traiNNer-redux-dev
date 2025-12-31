# traiNNer-redux
![redux3](https://github.com/user-attachments/assets/d107b2fc-6b68-4d3e-b08d-82c8231796cb)

Deep learning training framework for **image super-resolution and restoration** (incl. video) built on PyTorch.

- Docs: https://trainner-redux.readthedocs.io/
- Reference: [Architecture reference](https://trainner-redux.readthedocs.io/en/latest/arch_reference.html), [Loss reference](https://trainner-redux.readthedocs.io/en/latest/loss_reference.html)
- Upstream model loader / arch support: **Spandrel** (chaiNNer-org) — see “Spandrel architecture coverage” below.

---

## Quick start (from a clean clone)

> The authoritative, step-by-step setup is in the docs: *Getting Started*.  
> This section is a compact “do the thing” version for README users.

### Requirements

- **Python 3.13 recommended**, **Python 3.12 supported** (3.14 not supported yet).
- A modern GPU is strongly recommended. NVIDIA CUDA is the most common. AMD ROCm works on supported Linux setups.
- Install a matching PyTorch build for your CUDA/ROCm version.

### Install

1. Clone:
   ```bash
   git clone https://github.com/the-database/traiNNer-redux.git
   cd traiNNer-redux
   ```

2. Install dependencies:
   - **Windows**: double-click `install.bat`
   - **Linux**: `chmod +x install.sh && ./install.sh`

### Sanity check training run (tiny test dataset)

```bat
venv\Scripts\activate
copy "options\_templates\train\SPAN\SPAN_S_fidelity.yml" "options\train\SPAN\custom_SPAN_S_fidelity.yml"
python train.py --auto_resume -opt .\options\train\SPAN\custom_SPAN_S_fidelity.yml
```

Stop with `Ctrl+C` after you confirm the training log is progressing.

### Start a real training run

- Pick a template under `options/train/<ARCH>/…yml`
- Set:
  - `name` (unique run name)
  - `scale`
  - dataset paths: `datasets.train.dataroot_gt` / `datasets.train.dataroot_lq`
  - optional pretrain: `path.pretrain_network_g`
  - optional validation: enable `val.val_enabled` + set `datasets.val.*` paths

Single GPU:

```bat
venv\Scripts\activate
python train.py --auto_resume -opt .\options\train\arch\config.yml
```

Multi-GPU:

```bat
venv\Scripts\activate
torchrun --nproc-per-node 4 train.py --launcher pytorch --auto_resume -opt .\options\train\arch\config.yml
```

---

## ✨ Feature matrix (quick reference)

The tables below mirror the project’s **registries** and **config surface** so you can quickly map YAML option strings to concrete implementations.

### Architecture weight classes

Some architectures are grouped into **weight classes** (speed + VRAM) using project benchmark results (see docs benchmark pages and `architecture_categories.json` in this repo). Treat these as **order-of-magnitude** guidance; results depend on GPU, resolution, precision, and settings.

| class | #arches | avg VRAM (GB) | avg sec / 1MP image |
|---|---:|---|---|
| Ultra light | 9 | 0.105–0.215 (med 0.140) | 0.001–0.007 (med 0.005) |
| Light | 12 | 0.215–0.357 (med 0.314) | 0.004–0.111 (med 0.028) |
| Medium | 20 | 0.410–1.073 (med 0.732) | 0.010–0.199 (med 0.107) |
| Medium heavy | 22 | 0.897–2.737 (med 1.368) | 0.025–1.419 (med 0.223) |
| Heavy | 12 | 2.480–6.400 (med 3.575) | 0.988–5.335 (med 2.292) |
| Ultra heavy | 9 | 5.553–13.193 (med 9.665) | 1.559–12.495 (med 3.445) |

### Generator architectures (`network_g.type`)

> **How to read:** set `network_g: { type: <option>, ... }` in a training or testing YAML config.  
> For full parameter blocks per architecture: see the docs “Architecture reference”.

| Family | Options (`network_g.type`) | Typical use | Weight class |
|---|---|---|---|
| ATD | `atd`, `atd_light` | SISR / restoration | Heavy |
| ArtCNN | `artcnn_r16f96`, `artcnn_r3f24`, `artcnn_r5f48`, `artcnn_r8f24`, `artcnn_r8f48`, `artcnn_r8f64` | SISR (fast CNN) | Ultra light → Light |
| AutoEncoder | `autoencoder` | Restoration / autoencoding | (unbenchmarked) |
| CRAFT | `craft` | SISR / restoration | Medium |
| CascadedGaze | `cascadedgaze` | SISR / restoration | (unbenchmarked) |
| DAT | `dat`, `dat_2`, `dat_light`, `dat_s` | SISR / restoration | Heavy |
| DCTLSA | `dctlsa` | SISR / restoration | Medium heavy |
| DIS | `dis_balanced`, `dis_fast` | SISR / restoration | (unbenchmarked) |
| DITN_Real | `ditn_real` | SISR / restoration | Medium |
| DRCT | `drct`, `drct_l`, `drct_xl` | SISR / restoration | Heavy → Ultra heavy |
| DWT | `dwt`, `dwt_s` | SISR / restoration | Heavy → Ultra heavy |
| EIMN | `eimn_a`, `eimn_l` | SISR / restoration | Medium |
| ELAN | `elan`, `elan_light` | SISR / restoration | Medium heavy |
| EMT | `emt` | SISR / restoration | Medium heavy |
| ESCRealM | `escrealm`, `escrealm_xl` | SISR / restoration | (unbenchmarked) |
| FDAT | `fdat`, `fdat_large`, `fdat_light`, `fdat_medium`, `fdat_tiny`, `fdat_xl` | SISR / restoration | (unbenchmarked) |
| FlexNet | `flexnet`, `metaflexnet` | SISR / restoration | Medium heavy |
| GRL | `grl_t`, `grl_s`, `grl_b` | SISR / restoration | Ultra heavy |
| GateRV3 | `gaterv3`, `gaterv3_r`, `gaterv3_s` | SISR / restoration | (unbenchmarked) |
| HAT | `hat_l`, `hat_m`, `hat_s` | SISR / restoration | Ultra heavy |
| HiT_SIR | `hit_sir` | SISR / restoration | Medium heavy |
| HiT_SNG | `hit_sng` | SISR / restoration | Medium heavy |
| HiT_SRF | `hit_srf` | SISR / restoration | Medium heavy |
| LKFMixer | `lkfmixer_t`, `lkfmixer_b`, `lkfmixer_l` | SISR / restoration | (unbenchmarked) |
| LMLT | `lmlt_tiny`, `lmlt_base`, `lmlt_large` | SISR / restoration | Light → Medium |
| MAN | `man_tiny`, `man_light`, `man` | SISR / restoration | Medium → Medium heavy |
| MetaGAN3 | `metagan3` | SISR / restoration | (unbenchmarked) |
| MoESR2 | `moesr2` | SISR / restoration | Medium heavy |
| MoSR | `mosr`, `mosr_t` | SISR / restoration | Light → Medium |
| MoSRv2 | `mosrv2` | SISR / restoration | (unbenchmarked) |
| OmniSR | `omnisr` | SISR / restoration | Medium heavy |
| PLKSR | `plksr`, `plksr_tiny` | SISR / restoration | Light |
| RCAN | `rcan`, `rcan_l`, `rcan_unshuffle` | SISR / restoration | Medium |
| RGT | `rgt`, `rgt_s` | SISR / restoration | (unbenchmarked) |
| RRDBNet (ESRGAN) | `esrgan`, `esrgan_lite` | SISR / restoration | Medium → Medium heavy |
| RTMoSR | `rtmosr`, `rtmosr_l`, `rtmosr_ul` | SISR / restoration | Ultra light → Light |
| RealPLKSR | `realplksr`, `realplksr_large`, `realplksr_tiny` | SISR / restoration | Light → Medium |
| SAFMN | `safmn`, `safmn_l` | SISR / restoration | Light → Medium |
| SCUNet_aaf6aa | `scunet_aaf6aa` | Denoising / restoration | Medium heavy |
| SPAN | `span`, `span_f32`, `span_f64`, `span_f96`, `span_s` | SISR / restoration | Medium |
| SRFormer | `srformer`, `srformer_light` | SISR / restoration | Medium heavy → Heavy |
| SRVGGNetCompact (Real-ESRGAN Compact) | `compact`, `ultracompact`, `superultracompact` | SISR / restoration (fast) | Ultra light |
| Sebica | `sebica`, `sebica_mini` | SISR / restoration (fast) | Ultra light |
| SeemoRe | `seemore_t` | SISR / restoration | Light |
| SpanPlus | `spanplus`, `spanplus_s`, `spanplus_st`, `spanplus_sts` | SISR / restoration | Ultra light → Medium |
| Swin2SR | `swin2sr_l`, `swin2sr_m`, `swin2sr_s` | SISR / restoration | Medium heavy → Heavy |
| SwinIR | `swinir_l`, `swinir_m`, `swinir_s` | SISR / restoration | Medium heavy → Heavy |
| TSCUNet | `tscunet` | Video restoration / denoising | (unbenchmarked) |
| TSPANv2 | `temporalspanv2` | Video super-resolution | (unbenchmarked) |
| TemporalSPAN | `temporalspan` | Video super-resolution | (unbenchmarked) |
| Unet (segmentation) | `unetsegmentation` | Segmentation / masks | (unbenchmarked) |
| UpCunet4x (Real-CUGAN) | `realcugan` | SISR / restoration | Medium heavy |


### Discriminator architectures (`network_d.type`)

| Discriminator | Options (`network_d.type`) |
|---|---|
| DUnet (spectral norm U-Net) | `dunet` |
| MetaGAN v2 discriminator | `metagan2` |
| PatchGANDiscriminatorSN (single-scale) | `patchgandiscriminatorsn` |
| MultiscalePatchGANDiscriminatorSN | `multiscalepatchgandiscriminatorsn` |
| UNetDiscriminatorSN | `unetdiscriminatorsn` |
| VGGStyleDiscriminator (spectral norm) | `vggstylediscriminator` |


### Optimizers and schedulers

> **Schedulers:** `train.scheduler.type` uses PyTorch scheduler names (and scheduler kwargs).  
> **Optimizers:** passed as `train.optim_g` / `train.optim_d` dicts; `type` selects the optimizer.

| Optimizer family | Options (`train.optim_*.type`) | Notes | Options |
|---|---|---|---|
| Built-in (custom) | `adan`, `adanschedulefree`, `adamwschedulefree` | Schedule-free wrappers are useful for long runs / stability; see templates. | nan |
| PyTorch (torch.optim) | nan | Passed through as `type:` + kwargs. | Any torch optimizer name (e.g. `adamw`, `adam`, `nadam`, `sgd`, `rmsprop`) |
| pytorch-optimizer (optional) | nan | Available when dependency is installed; see project templates / install scripts. | Many extra optimizers (e.g. `stableadamw`, `soap`, `adopt`, …) |


### Losses (`train.losses[].type`)

> Loss configs live in `train.losses` (a list). Each entry has at minimum:
> - `type`: loss name (lowercase as shown)
> - `loss_weight`: scalar multiplier

| Category | Loss `type` | What it’s for |
|---|---|---|
| Pixel / distortion | `l1loss` | L1 |
| Pixel / distortion | `mseloss` | L2 / MSE |
| Pixel / distortion | `charbonnierloss` | Robust L1 (default in many SR recipes) |
| Structural | `ssimloss` | SSIM loss |
| Structural | `mssimloss` | MS-SSIM loss |
| Structural | `msssiml1loss` | MS-SSIM + L1 combined loss (common for pretraining) |
| Perceptual | `perceptualloss` | VGG-feature perceptual loss |
| Perceptual | `perceptualfp16loss` | FP16-friendly perceptual variant |
| Perceptual | `perceptualanimeloss` | Anime-tuned perceptual variant |
| Perceptual | `distsloss` | DISTS as loss (set `as_loss: true`) |
| Perceptual | `adistsloss` | Adaptive DISTS |
| Adversarial | `ganloss` | GAN loss (vanilla/lsgan/wgan-gp/etc via `gan_type`) |
| Adversarial | `multiscaleganloss` | GAN loss across multi-scale discriminator outputs |
| Frequency / texture | `fftloss` | FFT-domain loss |
| Frequency / texture | `ffloss` | Focal Frequency loss |
| Regularization | `gradientvarianceloss` | Gradient variance regularizer |
| Regularization | `fliploss` | FLIP perceptual error used as a loss |
| Color | `colorloss` | Color consistency loss (criterion selectable) |
| Color | `hsluvloss` | HSLuv-space perceptual-ish color loss |
| Color | `lumaloss` | Luma-only loss wrapper |
| Geometry / alignment | `nccloss` | Normalized cross-correlation |
| Similarity | `cosimloss` | Cosine similarity-based term |
| Contextual / contrastive | `contextualloss` | Contextual (CX) loss w/ VGG backend |
| Downscale consistency | `bicubicloss` | Downscale-and-compare loss (criterion selectable) |
| Quality proxy | `psnrloss` | PSNR-derived loss |
| Distillation / edges | `linedistillerloss` | LineDistiller feature/edge guidance |
| Distribution / classification | `ldlloss` | Label Distribution Learning loss |
| Segmentation | `bcewithlogitsdiceloss` | BCE + Dice combo (useful for masks) |
| Composite | `averageloss` | Averages multiple criteria |
| Composite | `aesoploss` | AESOP composite; criterion selectable |


### Validation metrics

> Validation metrics are typically enabled via `val.metrics_enabled`.  
> Use paired validation datasets (`PairedImageDataset` / `PairedVideoDataset`) when metrics are enabled.

| Metric | Option | Notes |
|---|---|---|
| PSNR | `calculate_psnr`, `calculate_psnr_pt` | Distortion metric (higher is better). |
| SSIM | `calculate_ssim`, `calculate_ssim_pt` | Distortion/structure metric (higher is better). |
| LPIPS | `calculate_lpips` | Perceptual similarity (lower is better). |
| DISTS | `calculate_dists` | Perceptual similarity (lower is better). |
| TOPIQ | `calculate_topiq`, `calculate_topiq_nr` | Full-reference and no-reference quality estimates (higher is better). |


### Augmentations and degradations

| Where | Config keys | What it does |
|---|---|---|
| Dataset (paired datasets) | `use_hflip`, `use_rot` | Basic geometric augmentation during random crop. |
| Mixture of Augmentations (MoA) | `use_moa`, `moa_augs`, `moa_probs`, `moa_debug`, `moa_debug_limit` | Randomly applies one augmentation per iteration from a user-defined list to improve robustness. |
| Real-ESRGAN style OTF degradations | `blur_prob`, `resize_prob`, `gaussian_noise_prob`, `jpeg_prob` (+ `*_range`/`*_list`/`*_prob` controls) | On-the-fly synthetic degradations (blur/resize/noise/JPEG) for real-world SR training. |
| Second-stage & high-order degradations | `blur_prob2`, `resize_prob2`, `gaussian_noise_prob2`, `jpeg_prob2`, `high_order_degradations` / debug keys | Two-stage degradation pipeline + optional high-order effects to match harsher input distributions. |
| Sharpening (LQ USM) | `lq_usm`, `lq_usm_radius_range` | Optional unsharp masking on the LQ image. |
| OTF queue / performance | `queue_size` | Controls the OTF processing queue (must be multiple of batch size). |


### Datasets / dataloaders

| Dataset / loader | Use | Notes |
|---|---|---|
| PairedImageDataset | Standard paired SR/restoration training (LQ/GT pairs). | Uses random crop of `lq_size`; supports `use_hflip`/`use_rot`. |
| PairedVideoDataset | Video restoration / VSR training. | Use `clip_size` to set frames/clip (must match temporal networks such as `tscunet`). |
| RealESRGANDataset | OTF degraded SR training (unpaired GT only). | Synthesizes LQ from GT on the fly using degradation pipeline options. |
| RealESRGANPairedDataset | Hybrid paired + OTF degradation workflows. | Used in some templates when both LQ and GT are available. |
| SingleImageDataset / SingleVideoDataset | Inference / testing scripts. | Run `test.py` configs; saves outputs to `results/`. |


---

## Spandrel architecture coverage

traiNNer-redux imports many architectures through **Spandrel** (chaiNNer-org). Spandrel itself supports additional *loadable/inferable* architectures beyond what is currently wired into traiNNer-redux training configs.

Spandrel’s README organizes supported models by task, including (examples):
- **SISR:** RRDBNet/ESRGAN, SRVGGNet (Real-ESRGAN compact), SwinIR/Swin2SR, HAT, OmniSR, SRFormer, DAT, GRL, DITN, SPAN, Real-CUGAN, SAFMN, DRCT, PLKSR/RealPLKSR, etc.
- **Face restoration:** GFPGAN, RestoreFormer, CodeFormer
- **Inpainting:** LaMa, MAT
- **Denoising/restoration:** SCUNet, Uformer, KBNet, NAFNet, Restormer, FFTformer, MPRNet, MIRNet2, DnCNN/DRUNet, IPT
- **DeJPEG:** FBCNN
- **Colorization / Dehazing / Low-light:** DDColor, MixDehazeNet, RetinexFormer, HVI-CIDNet

If you need an architecture that exists in Spandrel but not in traiNNer-redux’s `network_g` registry yet, it may still be runnable in other tooling (e.g. chaiNNer) or can be requested/added.

---

## Resources

- **OpenModelDB**: pretrain model repository
- **chaiNNer**: run/export many trained models and assist with dataset prep
- **WTP Dataset Destroyer / Video Destroyer**: create degraded datasets
- **vs_align / ImgAlign**: align paired data

---

## Contributing

- Docs: https://trainner-redux.readthedocs.io/en/stable/contributing.html
- Please open PRs with clear motivation, reproducible config changes, and/or benchmark evidence when adding new architectures or training recipes.
