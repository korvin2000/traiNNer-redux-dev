# traiNNer-redux
![redux3](./docs/assets/redux_logo.png)

## Overview
[traiNNer-redux](https://trainner-redux.readthedocs.io/en/latest/index.html) is a deep learning training framework for image super resolution and restoration which allows you to train PyTorch models for upscaling and restoring images and videos. NVIDIA graphics card is recommended, but AMD works on Linux machines with ROCm.

## Usage Instructions
Please see the [getting started](https://trainner-redux.readthedocs.io/en/latest/getting_started.html) page for instructions on how to use traiNNer-redux.

## ✨ Feature Matrix (quick reference)

Tables below mirror the project registries so you can quickly map option strings to concrete implementations. The weight column follows the resource taxonomy in [`architecture_categories.json`](architecture_categories.json), and dates reflect when the implementation landed in this repository (first commit date) unless a historical paper/release date is well-established. See also the [architecture reference](https://trainner-redux.readthedocs.io/en/latest/arch_reference.html) and [loss reference](https://trainner-redux.readthedocs.io/en/latest/loss_reference.html) for per-model parameter blocks.

### Supported architectures

| arch | option | date | weight |
|------|--------|------|--------|
| ATD | `atd`, `atd_light` | 2025-11-29 | Heavy |
| ArtCNN | `artcnn_r16f96`, `artcnn_r3f24`, `artcnn_r5f48`, `artcnn_r8f24`, `artcnn_r8f48`, `artcnn_r8f64` | 2025-11-29 | Ultra light → Light |
| AutoEncoder | `autoencoder` | 2025-11-29 | — |
| CRAFT | `craft` | 2025-11-29 | Medium |
| DAT | `dat`, `dat_2`, `dat_light`, `dat_s` | 2025-11-29 | Heavy |
| DCTLSA | `dctlsa` | 2025-11-29 | Medium heavy |
| DIS | `dis_balanced`, `dis_fast` | 2025-12-02 | — |
| DITN_Real | `ditn_real` | 2025-11-29 | Medium |
| DRCT | `drct`, `drct_l`, `drct_xl` | 2025-11-29 | Heavy → Ultra heavy |
| DWT | `dwt`, `dwt_s` | 2025-11-29 | Heavy → Ultra heavy |
| EIMN | `eimn_a`, `eimn_l` | 2025-11-29 | Medium |
| ELAN | `elan`, `elan_light` | 2025-11-29 | Medium heavy |
| EMT | `emt` | 2025-11-29 | Medium heavy |
| ESCRealM | `escrealm`, `escrealm_xl` | 2025-11-29 | — |
| FDAT | `fdat`, `fdat_large`, `fdat_light`, `fdat_medium`, `fdat_tiny`, `fdat_xl` | 2025-11-29 | — |
| FlexNet | `flexnet`, `metaflexnet` | 2025-11-29 | Medium heavy |
| GateRV3 | `gaterv3`, `gaterv3_r`, `gaterv3_s` | 2025-11-29 | — |
| GRL | `grl_t`, `grl_s`, `grl_b` | 2025-11-29 | Ultra heavy |
| HAT | `hat_l`, `hat_m`, `hat_s` | 2025-11-29 | Ultra heavy |
| HiT | `hit_sir`, `hit_sng`, `hit_srf` | 2025-11-29 | Medium heavy |
| LKFMixer | `lkfmixer_t`, `lkfmixer_b`, `lkfmixer_l` | 2025-11-27 | — |
| LMLT | `lmlt_tiny`, `lmlt_base`, `lmlt_large` | 2025-11-29 | Light → Medium |
| MAN | `man_tiny`, `man_light`, `man` | 2025-11-29 | Medium → Medium heavy |
| MetaGAN3 | `metagan3` | 2025-11-29 | — |
| MoESR2 | `moesr2` | 2025-11-29 | Medium heavy |
| MoSR | `mosr`, `mosr_t` | 2025-11-29 | Light → Medium |
| MoSRv2 | `mosrv2` | 2025-11-29 | — |
| OmniSR | `omnisr` | 2025-11-29 | Medium heavy |
| PLKSR | `plksr`, `plksr_tiny` | 2025-11-29 | Light |
| RCAN | `rcan`, `rcan_l`, `rcan_unshuffle` | 2025-11-29 | Medium |
| RGT | `rgt`, `rgt_s` | 2025-11-29 | — |
| RRDBNet / ESRGAN | `esrgan`, `esrgan_lite` | 2025-11-29 | Medium → Medium heavy |
| RTMoSR | `rtmosr`, `rtmosr_l`, `rtmosr_ul` | 2025-11-29 | Ultra light → Light |
| RealCUGAN | `realcugan` | 2025-11-29 | Medium heavy |
| RealPLKSR | `realplksr`, `realplksr_large`, `realplksr_tiny` | 2025-11-29 | Light → Medium |
| SAFMN | `safmn`, `safmn_l` | 2025-11-29 | Light → Medium |
| SCUNet_aaf6aa | `scunet_aaf6aa` | 2025-11-29 | Medium heavy |
| SPAN | `span`, `span_f32`, `span_f64`, `span_f96`, `span_s` | 2025-11-29 | Medium |
| SRFormer | `srformer`, `srformer_light` | 2025-11-29 | Medium heavy → Heavy |
| SRVGGNetCompact | `compact`, `ultracompact`, `superultracompact` | 2025-11-29 | Ultra light |
| Sebica | `sebica`, `sebica_mini` | 2025-11-29 | Ultra light |
| SeemoRe | `seemore_t` | 2025-11-29 | Light |
| SpanPlus | `spanplus`, `spanplus_s`, `spanplus_st`, `spanplus_sts` | 2025-11-29 | Ultra light → Medium |
| Swin2SR | `swin2sr_l`, `swin2sr_m`, `swin2sr_s` | 2025-11-29 | Medium heavy → Heavy |
| SwinIR | `swinir_l`, `swinir_m`, `swinir_s` | 2025-11-29 | Medium heavy → Heavy |
| TSCUNet | `tscunet` | 2025-11-29 | — |
| TemporalSPAN | `temporalspan` | 2025-11-29 | — |
| TSPANv2 | `temporalspanv2` | 2025-11-18 | — |
| Unet (segmentation) | `unetsegmentation` | 2025-11-29 | — |

> [!NOTE]
> Additional registry entries (e.g., Paragon Diffusion/SR families, LKFMixer, ElysiumSR, HyperionSR, TSPAN/TemporalSpan, etc.) are also available; consult `docs/source/arch_reference.md` and the architecture registry for the full surface area.

### Supported discriminators

| net | option | date |
|-----|--------|------|
| VGGStyleDiscriminator (spectral norm) | `vggstylediscriminator` | 2025-11-29 |
| UNetDiscriminatorSN | `unetdiscriminatorsn` | 2025-11-29 |
| PatchGAN (single-scale spectral norm) | `patchgandiscriminatorsn` | 2025-11-29 |
| PatchGAN (multiscale spectral norm) | `multiscalepatchgandiscriminatorsn` | 2025-11-29 |
| DUnet (spectral norm U-Net) | `dunet` | 2025-11-29 |
| MetaGAN v2 discriminator | `metagan2` | 2025-11-29 |

### Supported optimizers

| optimizer | option |
|-----------|--------|
| Adan + fused/schedule-free variants | `adan`, `adanschedulefree` | 2025-11-29 |
| AdamW schedule free wrapper | `adamwschedulefree` | 2025-11-29 |
| PyTorch & pytorch-optimizer fallbacks | `adam`, `adamw`, `nadam`, `stableadamw`, `soap`, `adopt`, etc. | 2025-11-29 |
| LR schedulers (via `train.scheduler.type`) | Any torch scheduler name with kwargs | — |

### Supported losses

| loss | option |
|------|--------|
| Pixel / distortion | `l1loss`, `mseloss`, `charbonnierloss` |
| Structural | `ssimloss`, `mssimloss`, `msssiml1loss` |
| Perceptual | `perceptualloss`, `perceptualfp16loss`, `perceptualanimeloss`, `distsloss`, `adistsloss` |
| GAN and adversarial variants | `ganloss`, `multiscaleganloss` |
| Frequency / texture | `fftloss`, `ffloss` |
| Regularizers | `gradientvarianceloss`, `fliploss` |
| Color | `colorloss`, `hsluvloss`, `lumaloss` |
| Geometry / alignment | `nccloss` |
| Similarity | `cosimloss` |
| Contextual / contrastive | `contextualloss` |
| Downscale consistency | `bicubicloss` |
| Quality proxy | `psnrloss` |
| Distillation / edges | `linedistillerloss` |
| Distribution / classification | `ldlloss` |
| Segmentation | `bcewithlogitsdiceloss` |
| Composite | `averageloss`, `aesoploss` |

### Supported metrics

| metric | option |
|--------|--------|
| PSNR (NumPy/PyTorch implementations) | `calculate_psnr`, `calculate_psnr_pt` |
| SSIM (NumPy/PyTorch implementations) | `calculate_ssim`, `calculate_ssim_pt` |
| Learned perceptual similarity | `calculate_lpips` |
| DISTS | `calculate_dists` |
| TOPIQ (full-reference and no-reference) | `calculate_topiq`, `calculate_topiq_nr` |

### Supported augmentations

| augmentation | option |
|--------------|--------|
| Random horizontal/vertical flip | `use_hflip` / `use_vflip` |
| 90° rotations | `use_rot` |
| Mixture of Augmentations (MoA) | `use_moa`, `moa_augs`, `moa_probs`, `moa_debug`, `moa_debug_limit` |
| Real-ESRGAN style on-the-fly degradations | `blur_prob`, `resize_prob`, `gaussian_noise_prob`, `jpeg_prob` (+ `*_range`/`*_list`/`*_prob` controls) |
| Second-stage & high-order degradations | `blur_prob2`, `resize_prob2`, `gaussian_noise_prob2`, `jpeg_prob2`, `high_order_degradations` |
| Sharpening (LQ USM) | `lq_usm`, `lq_usm_radius_range` |
| OTF queue sizing | `queue_size` |

### Supported dataloaders

| loader | option |
|--------|--------|
| Paired image/video datasets | `PairedImageDataset`, `PairedVideoDataset` |
| Single image/video datasets (inference) | `SingleImageDataset`, `SingleVideoDataset` |
| Real-ESRGAN degradations | `RealESRGANDataset`, `RealESRGANPairedDataset` |

## Contributing
Please see the [contributing](https://trainner-redux.readthedocs.io/en/latest/contributing.html) page for more info on how to contribute.

## Resources
- [OpenModelDB](https://openmodeldb.info/): Repository of AI upscaling models, which can be used as pretrain models to train new models. Models trained with this repo can be submitted to OMDB.
- [chaiNNer](https://github.com/chaiNNer-org/chaiNNer): General purpose tool for AI upscaling and image processing, models trained with this repo can be run on chaiNNer. chaiNNer can also assist with dataset preparation.
- [WTP Dataset Destroyer](https://github.com/umzi2/wtp_dataset_destroyer): Tool to degrade high quality images, which can be used to prepare the low quality images for the training dataset.
- [Video Destroyer](https://github.com/Kim2091/video-destroyer): A tool designed to create datasets from a single high quality input video for AI training purposes.
- [helpful-scripts](https://github.com/Kim2091/helpful-scripts): Collection of scripts written to improve experience training AI models.
- [Dataset_Preprocessing](https://github.com/umzi2/Dataset_Preprocessing): A small collection of scripts for initial dataset processing.
- [Enhance Everything! Discord Server](https://discord.gg/cpAUpDK): Get help training a model, share upscaling results, submit your trained models, and more.
- [vs_align](https://github.com/pifroggi/vs_align): Video Alignment and Synchonization for Vapoursynth, tool to align LR and HR datasets.
- [ImgAlign](https://github.com/sonic41592/ImgAlign): Tool for auto aligning, cropping, and scaling HR and LR images for training image based neural networks.
- [Deep Learning Tuning Playbook](https://developers.google.com/machine-learning/guides/deep-learning-tuning-playbook): Document to help you train deep learning models more effectively.

## License and Acknowledgement

traiNNer-redux is released under the [Apache License 2.0](LICENSE.txt). See [LICENSE](LICENSE/README.md) for individual licenses and acknowledgements.

- This repository is a fork of [joeyballentine/traiNNer-redux](https://github.com/joeyballentine/traiNNer-redux) which itself is a fork of [BasicSR](https://github.com/XPixelGroup/BasicSR).
- Network architectures are imported from [Spandrel](https://github.com/chaiNNer-org/spandrel).
- Several architectures are developed by [umzi2](https://github.com/umzi2): [ArtCNN-PyTorch](https://github.com/umzi2/ArtCNN-PyTorch), [DUnet](https://github.com/umzi2/DUnet), [FlexNet](https://github.com/umzi2/FlexNet), [GaterV3](https://github.com/umzi2/GaterV3), [MetaGan](https://github.com/umzi2/MetaGan), [MoESR](https://github.com/umzi2/MoESR), [MoSR](https://github.com/umzi2/MoSR), [RTMoSR](https://github.com/rewaifu/RTMoSR), [SPANPlus](https://github.com/umzi2/SPANPlus)
- The [ArtCNN](https://github.com/Artoriuz/ArtCNN) architecture is originally developed by [Artoriuz](https://github.com/Artoriuz).
- The TSCUNet architecture is from [aaf6aa/SCUNet](https://github.com/aaf6aa/SCUNet) which is a modification of [SCUNet](https://github.com/cszn/SCUNet), and parts of the training code for TSCUNet are adapted from [TSCUNet_Trainer](https://github.com/Demetter/TSCUNet_Trainer).
- The [FDAT](https://github.com/stinkybread/FDAT) architecture is created by [stinkybread (sharekhan)](https://github.com/stinkybread).
- The [TSPAN](https://github.com/Kim2091/TSPAN) and [TSPANv2](https://github.com/Kim2091/TSPANv2) architectures are temporally consistent variants of the SPAN architecture created by [Kim2091](https://github.com/Kim2091). TSPANv2 is an enhanced version of TSPAN with major improvements to temporal stability, Use [Vapourkit](https://github.com/Kim2091/vapourkit) to upscale videos with TSPAN and TSPANv2 models.
- The [DIS](https://github.com/Kim2091/DIS) architecture is an ultra lightweight arch created by [Kim2091](https://github.com/Kim2091).
- Several enhancements reference implementations from [Corpsecreate/neosr](https://github.com/Corpsecreate/neosr) and its original repo [neosr](https://github.com/muslll/neosr).
- Members of the Enhance Everything Discord server: [Corpsecreate](https://github.com/Corpsecreate), [joeyballentine](https://github.com/joeyballentine), [Kim2091](https://github.com/Kim2091).
