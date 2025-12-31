# traiNNer-redux
![redux3](https://github.com/user-attachments/assets/d107b2fc-6b68-4d3e-b08d-82c8231796cb)

## Overview
[traiNNer-redux](https://trainner-redux.readthedocs.io/en/latest/index.html) is a deep learning training framework for image super resolution and restoration which allows you to train PyTorch models for upscaling and restoring images and videos. NVIDIA graphics card is recommended, but AMD works on Linux machines with ROCm.

## Usage Instructions
Please see the [getting started](https://trainner-redux.readthedocs.io/en/latest/getting_started.html) page for instructions on how to use traiNNer-redux.

## ✨ Feature Matrix (quick reference)

Tables below mirror the project registries so you can quickly map option strings to concrete implementations. The weight column follows the resource taxonomy in [`architecture_categories.json`](architecture_categories.json), and dates reflect when the implementation landed in this repository (first commit date) unless a historical paper/release date is well-established.

### Supported architectures

| arch | option | date | weight |
|------|--------|------|--------|
| HAT-L/HAT-M/HAT-S | `hat_l`, `hat_m`, `hat_s` | 2024-12-28 | Ultra heavy |
| DRCT family | `drct_xl`, `drct_l`, `drct` | 2024-12-28 | Ultra heavy / Ultra heavy / Heavy |
| DAT family | `dat`, `dat_2`, `dat_s`, `dat_light` | 2024-12-28 | Heavy / Heavy / Heavy / Heavy |
| ATD variants | `atd`, `atd_light` | 2024-12-28 | Heavy / Heavy |
| SwinIR | `swinir_l`, `swinir_m`, `swinir_s` | 2024-12-28 | Heavy / Medium heavy / Medium heavy |
| Swin2SR | `swin2sr_l`, `swin2sr_m`, `swin2sr_s` | 2024-12-28 | Heavy / Heavy / Medium heavy |
| SRFormer | `srformer`, `srformer_light` | 2024-12-28 | Heavy / Medium heavy |
| OmniSR | `omnisr` | 2024-12-28 | Medium heavy |
| Real-ESRGAN / ESRGAN | `realesrgan`, `esrgan`, `esrgan_lite` | 2024-12-28 | Medium heavy / Medium heavy / Medium |
| MoESR2 | `moesr2` | 2024-12-28 | Medium heavy |
| FlexNet / MetaFlexNet | `flexnet`, `metaflexnet` | 2024-12-28 | Medium heavy |
| SCUNet variants | `scunet`, `scunet_aaf6aa` | 2024-12-28 | Medium heavy |
| RTMoSR variants | `rtmosr`, `rtmosr_l`, `rtmosr_ul` | 2025-01-01 | Light / Ultra light / Ultra light |
| ArtCNN presets | `artcnn_r16f96`, `artcnn_r8f64`, `artcnn_r8f48` | 2024-12-28 | Light / Light / Ultra light |
| PLKSR | `plksr`, `plksr_tiny` | 2024-12-28 | Light / Light |
| RealPLKSR | `realplksr`, `realplksr_tiny` | 2024-12-28 | Medium / Light |
| SAFMN | `safmn`, `safmn_l` | 2025-01-05 | Light / Medium |
| Span / Span+ | `span`, `span_s`, `spanplus`, `spanplus_s`, `spanplus_st`, `spanplus_sts` | 2024-12-28 | Medium / Medium / Medium / Medium / Light / Ultra light |
| LMLT | `lmlt_tiny`, `lmlt_base`, `lmlt_large` | 2024-12-28 | Light / Medium / Medium |
| MAN | `man_tiny`, `man_light`, `man` | 2024-12-28 | Medium / Medium / Medium heavy |
| EIMN | `eimn_a`, `eimn_l` | 2024-12-28 | Medium / Medium |
| Sebica | `sebica_mini`, `sebica` | 2025-01-05 / unknown | Ultra light / Ultra light |
| DWT | `dwt_s`, `dwt` | 2025-01-05 | Heavy / Ultra heavy |
| RGT | `rgt_s`, `rgt` | 2024-12-28 | Medium heavy |

> [!NOTE]
> Additional registry entries (e.g., Paragon Diffusion/SR families, LKFMixer, ElysiumSR, HyperionSR, TSPAN/TemporalSpan, etc.) are also available; consult `docs/source/arch_reference.md` and the architecture registry for the full surface area.

### Supported discriminators

| net | option | date |
|-----|--------|------|
| VGGStyleDiscriminator (spectral norm) | `discriminator` | 2024-12-28 |
| UNetDiscriminatorSN | `dis` | 2025-12-02 |
| PatchGAN (single-scale spectral norm) | `patchgan` | 2025-06-29 |
| PatchGAN (multiscale spectral norm) | `multiscalepatchgandiscriminatorsn` | 2025-06-29 |
| DUnet (spectral norm U-Net) | `dunet` | 2024-12-28 |
| MetaGAN v2 discriminator | `metagan2` | 2024-12-28 |

### Supported optimizers

| optimizer | option |
|-----------|--------|
| Adan (and fused schedule-free variant) | `Adan`, `adan` | 2025-02-27 |
| Adan schedule free wrapper | `AdanScheduleFree` | 2025-02-27 |
| AdamW schedule free wrapper | `AdamWScheduleFree` | 2025-02-19 |
| PyTorch & pytorch-optimizer fallbacks | `Adam`, `AdamW`, `NAdam`, `StableAdamW`, `SOAP`, `ADOPT`, etc. | 2025-02-19 |

### Supported losses

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
