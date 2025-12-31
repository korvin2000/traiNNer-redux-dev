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
| [ATD](https://arxiv.org/abs/2401.08209) | `atd`, `atd_light` | 2024-01 | Heavy |
| [ArtCNN](https://github.com/Artoriuz/ArtCNN) | `artcnn_r16f96`, `artcnn_r3f24`, `artcnn_r5f48`, `ar...n_r8f24`, `artcnn_r8f48`, `artcnn_r8f64` | 2024-11 | Ultra light → Light |
| [AutoEncoder](https://en.wikipedia.org/wiki/Autoencoder) | `autoencoder` | — | — |
| [CRAFT](https://arxiv.org/abs/2308.05022) | `craft` | 2023-08 | Medium |
| [DAT](https://arxiv.org/abs/2308.03364) | `dat`, `dat_2`, `dat_light`, `dat_s` | 2023-08 | Heavy |
| [DCTLSA](https://github.com/zengkun301/DCTLSA) | `dctlsa` | 2023 | Medium heavy |
| [DIS](https://github.com/Kim2091/DIS) | `dis_balanced`, `dis_fast` | 2025-11 | lightweight |
| [DITN_Real](https://arxiv.org/abs/2308.02794) | `ditn_real` | 2023-08 | Medium |
| [DRCT](https://arxiv.org/abs/2404.00722) | `drct`, `drct_l`, `drct_xl` | 2024-03 | Heavy → Ultra heavy |
| [DWT](https://github.com/soobin419/DWT) | `dwt`, `dwt_s` | 2023 | Heavy → Ultra heavy |
| [EIMN](https://github.com/liux520/EIMN) | `eimn_a`, `eimn_l` | 2023 | Medium |
| [ELAN](https://arxiv.org/abs/2203.06697) | `elan`, `elan_light` | 2022-03 | Light → Medium |
| [EMT](https://arxiv.org/abs/2305.11403) | `emt`, `emt_l`, `emt_t` | 2023-05 | Light → Medium |
| [ESC](https://github.com/dslisleedh/ESC) | `escrealm` | 2025 | Medium |
| [ESRGAN](https://arxiv.org/abs/1809.00219) | `esrgan`, `esrgan_lite` | 2018-09 | Medium |
| [FDAT](https://github.com/stinkybread/FDAT) | `fdat` | 2025-06 | Medium |
| [FlexNet](https://github.com/umzi2/FlexNet) | `flexnet` | 2024-10 | Light |
| [GateRV3](https://github.com/umzi2/GaterV3) | `gaterv3`, `gaterv3_light`, `gaterv3_ultra` | 2025-06 | Light → Medium |
| [GRL](https://arxiv.org/abs/2303.00748) | `grl`, `grl_b`, `grl_s`, `grl_t` | 2023-03 | Medium heavy |
| [HAT](https://arxiv.org/abs/2205.04437) | `hat`, `hat_s`, `hat_xl` | 2022-05 | Ultra heavy |
| [HiT](https://arxiv.org/abs/2407.05878) | `hit_sng`, `hit_srf` (`srf_medium`, `srf_large`) | 2024-07 | Light → Heavy |
| [LKFMixer](https://arxiv.org/abs/2508.11391) | `lkfmixer` | 2025-08 | Medium |
| [LMLT](https://arxiv.org/abs/2409.03516) | `lmlt_base`, `lmlt_large` | 2024-09 | Medium heavy |
| [MAN](https://arxiv.org/abs/2209.14145) | `man_light`, `man_tiny` | 2022-09 | Medium |
| [MetaGAN3](https://github.com/umzi2/MetaGan) | `metagan3`, `metagan3_jpeg`, `metagan3_jpeg_ca`, `metagan3_pe` | 2025 | Medium heavy |
| [MoESR2](https://github.com/umzi2/MoESR) | `moesr2` | 2025 | Heavy |
| [MoSR](https://github.com/umzi2/MoSR) | `mosr`, `mosr_t` | 2024-08 | Medium |
| [MoSRv2](https://github.com/umzi2/MoSRV2) | `mosrv2`, `mosrv2_t`, `mosrv2_tu` | 2025-03 | Medium |
| [OmniSR](https://arxiv.org/abs/2304.10244) | `omnisr` | 2023-04 | Light |
| [PLKSR](https://arxiv.org/abs/2404.11848) | `plksr`, `plksr_tiny` | 2024-04 | Medium heavy |
| [RCAN](https://arxiv.org/abs/1807.02758) | `rcan` | 2018-07 | Medium heavy |
| [RGT](https://arxiv.org/abs/2303.06373) | `rgt`, `rgt_s` | 2023-03 | Medium heavy |
| [RRDBNet](https://arxiv.org/abs/1809.00219) | `rrdbnet`, `rrdbnet_bicubic` | 2018-09 | Medium |
| [RTMoSR](https://github.com/rewaifu/RTMoSR) | `rtmosr`, `rtmosr_l`, `rtmosr_ul` | 2025-01 | Ultra light → Light |
| [RealCUGAN](https://github.com/bilibili/ailab/tree/main/Real-CUGAN) | `realcugan` | 2022-01 | Medium heavy |
| [RealPLKSR](https://github.com/dslisleedh/PLKSR) | `realplksr`, `realplksr_large`, `realplksr_tiny` | 2024-04 | Light → Medium |
| [SAFMN](https://arxiv.org/abs/2302.13800) | `safmn`, `safmn_l` | 2023-02 | Light → Medium |
| [SCUNet](https://arxiv.org/abs/2203.13278) | `scunet` (`aaf6aa`, `wot`, `clamp`, `esrgan`, `snr`, `tiny`) | 2022-03 | Medium |
| [Sebica](https://arxiv.org/abs/2410.20546) | `sebica`, `sebica_mini` | 2024-10 | Light |
| [SeemoRe](https://arxiv.org/abs/2402.03412) | `seemore_b`, `seemore_t` | 2024-02 | Medium |
| [SPAN](https://arxiv.org/abs/2311.12770) | `span` | 2023-11 | Ultra light |
| [SpanPlus](https://github.com/umzi2/SPANPlus) | `spanplus` (`s`, `sts`, `srep`, `st`, `srep_st`, `sts_st`) | 2024-08 | Ultra light |
| [SRFormer](https://arxiv.org/abs/2303.09735) | `srformer_light`, `srformer_medium` | 2023-03 | Light → Medium |
| [SRVGGNetCompact](https://github.com/xinntao/Real-ESRGAN/blob/master/realesrgan/archs/srvgg_arch.py) | `srvggnet_compact` | — | Light |
| [Swin2SR](https://arxiv.org/abs/2209.11345) | `swin2sr`, `swin2sr_l` | 2022-09 | Heavy |
| [SwinIR](https://arxiv.org/abs/2108.10257) | `swinir`, `swinir_l` | 2021-08 | Heavy |
| [TSCUNet](https://github.com/Demetter/TSCUNet_Trainer) | `tscunet`, `tscunet_l` | 2024-04 | Medium |
| [TSPAN](https://github.com/Kim2091/TSPAN) | `tspan`, `tspan_nx`, `tspanplus` | 2025-09 | Ultra light |
| [TSPANv2](https://github.com/Kim2091/TSPANv2) | `tspan2` | 2025-11 | Ultra light |
| [UNet](https://arxiv.org/abs/1505.04597) | `unet` | 2015-05 | Medium |
| [A2FPN](https://github.com/lironui/A2-FPN) | `a2fpn` | 2022-01 | Light |


> [!NOTE]
> Additional registry entries (e.g., Paragon Diffusion/SR families, LKFMixer, ElysiumSR, HyperionSR, TSPAN/TemporalSpan, etc.) are also available; consult `docs/source/arch_reference.md` and the architecture registry for the full surface area.

### Supported discriminators

| net | option | date |
|-----|--------|------|
| VGGStyleDiscriminator (spectral norm) | `vggstylediscriminator` | 2016 |
| UNetDiscriminatorSN | `unetdiscriminatorsn` | 2020 |
| PatchGAN (single-scale spectral norm) | `patchgandiscriminatorsn` | 2017-2018 |
| PatchGAN (multiscale spectral norm) | `multiscalepatchgandiscriminatorsn` | 2023 |
| DUnet (spectral norm U-Net) | `dunet` | 2024 |
| MetaGAN v2 discriminator | `metagan2` | 2024 |

### Supported optimizers

| optimizer | option |
|-----------|--------|
| Adan + fused/schedule-free variants | `adan`, `adanschedulefree` | 2022 |
| AdamW schedule free wrapper | `adamwschedulefree` | 2017 (2020) |
| PyTorch & pytorch-optimizer fallbacks | `adam`, `adamw`, `nadam`, `stableadamw`, `soap`, `adopt`, etc. | — |
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

### Losses by category

| loss | option |
|------|--------|
| Pixel losses (L1/L2/Huber/Charbonnier) | `L1Loss`, `MSELoss`, `HuberLoss`, `CharbonnierLoss` |
| GAN and adversarial variants | `GANLoss`, `MultiScaleGANLoss`, `R3GANLoss`, `MultiScaleR3GANLoss` |
| Perceptual and LPIPS-family | `PerceptualLoss`, `PerceptualFP16Loss`, `PerceptualAnimeLoss`, `DINOPerceptualLoss`, `DISTSLoss`, `ADISTSLoss`, `ConvNeXtPerceptualLoss` |
| Regularizers and TV | `TVLoss`, `AdaptiveBlockTVLoss`, `FFTLoss`, `GradientVarianceLoss`, `AdaptiveBlockTVLoss` |
| Contrastive / contextual | `ContextualLoss`, `ContrastiveLoss`, `ConsistencyLoss`, `LineDistillerLoss` |
| Miscellaneous specialty losses | `CheckerboardLoss`, `ColorLoss`, `FFLoss`, `LaplacianPyramidLoss`, `LDLLoss`, `HSLuvLoss`, `PSNRLoss`, `BicubicLoss`, `AverageLoss` |

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
