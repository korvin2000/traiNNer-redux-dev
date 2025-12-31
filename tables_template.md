
## ✨ features

### [supported archs](https://github.com/neosr-project/neosr/wiki/Arch%E2%80%90specific-options):

| arch                                                                                              | option                                                                           | date  | weight                                   |
|---------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------|-------|------------------------------------------|
| [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN)                                             | `esrgan`                                                                          | 2021  | Medium heavy                            |
| [SRVGGNetCompact](https://github.com/XPixelGroup/BasicSR/blob/master/basicsr/archs/srvgg_arch.py) | `compact`                                                                         | 2021  | Ultra light                              |
| [SwinIR](https://github.com/JingyunLiang/SwinIR)                                                  | `swinir_small`, `swinir_medium`, `swinir_large`                                  | 2021  | Medium heavy / Medium heavy / Heavy      |
| [HAT](https://github.com/XPixelGroup/HAT)                                                         | `hat_s`, `hat_m`, `hat_l`                                                         | 2022  | Ultra heavy / Ultra heavy / Ultra heavy  |
| [OmniSR](https://github.com/Francis0625/Omni-SR)                                                  | `omnisr`                                                                          | 2023  | Medium heavy                            |

> [!NOTE]
> Weights reflect approximate resource demands (compute + VRAM) using the project's `architecture_categories.json` plus best-effort estimates for newer models.



### [supported discriminators](https://github.com/neosr-project/neosr/wiki/Arch%E2%80%90specific-options#discriminators):

| net                                                                           | option                        | date  |
|-------------------------------------------------------------------------------|-------------------------------|-------|
| U-Net w/ SN                                                                   | `unet`                        | 2015  |
| [PatchGAN](https://github.com/NVIDIA/pix2pixHD) w/ SN                         | `patchgan`                    | 2018  |
| EA2FPN (bespoke, based on [A2-FPN](https://github.com/lironui/A2-FPN))        | `ea2fpn`                      | 2024  |
| [DUnet](https://github.com/umzi2/DUnet)                                       | `dunet`                       | 2024  |
| [MetaGan](https://github.com/umzi2/MetaGan)                                   | `metagan`                     | 2024  |
| [ResGAN](https://github.com/neosr-project/neosr/blob/master/neosr/archs/resgan_arch.py) | `resgan`               | 2023  |
### [supported optimizers](https://github.com/neosr-project/neosr/wiki/Optimizer-Options):

| optimizer                                                                 | option             	 |
|---------------------------------------------------------------------------|----------------------------|
| [Adam](https://pytorch.org/docs/stable/generated/torch.optim.Adam.html)   | `Adam` or `adam`   	 |
| [AdamW](https://pytorch.org/docs/stable/generated/torch.optim.AdamW.html) | `AdamW` or `adamw` 	 |
| [NAdam](https://pytorch.org/docs/stable/generated/torch.optim.NAdam.html) | `NAdam` or `nadam` 	 |
| [Adan](https://github.com/sail-sg/Adan)                                   | `Adan` or `adan`   	 |


### [supported losses](https://github.com/neosr-project/neosr/wiki/Losses):

| loss                                                                                                                         | option                                      |
|-------------------------------------------------------------------------------------------------------------------------------|--------------------------------------------|
| L1 Loss                                                                                                                      | `L1Loss`, `l1_loss`                         |
| L2 Loss                                                                                                                      | `MSELoss`, `mse_loss`                       |
| Huber Loss                                                                                                                   | `HuberLoss`, `huber_loss`                   |
| CHC (Clipped Huber with Cosine Similarity Loss)                                                                              | `chc_loss`                                  |



### [supported metrics](https://github.com/neosr-project/neosr/wiki/Configuration-Walkthrough#validation)

| metric                                            | option             |
|---------------------------------------------------|--------------------|
| PSNR                                              | `calculate_psnr`   |
| SSIM                                              | `calculate_ssim`   |
| [DISTS](https://github.com/dingkeyan93/DISTS)     | `calculate_dists`  |
| [TOPIQ](https://github.com/chaofengc/IQA-PyTorch) | `calculate_topiq`  |
| ILNIQE                                            | `calculate_ilniqe` |
### [supported augmentations](https://github.com/neosr-project/neosr/wiki/Configuration-Walkthrough#augmentations-aug_prob):

| augmentation						| option	|
|-------------------------------------------------------|---------------|
| Rotation						| `use_rot`	|
| Flip							| `use_hflip`	|
| [MixUp](https://arxiv.org/abs/1710.09412)		| `mixup`	|
| [CutMix](https://arxiv.org/abs/1905.04899)		| `cutmix`	|
| [ResizeMix](https://arxiv.org/abs/2012.11101)		| `resizemix`	|
| [CutBlur](https://github.com/clovaai/cutblur/)	| `cutblur`	|

### [supported models](https://github.com/neosr-project/neosr/wiki/Configuration-Walkthrough#model_type):

| model 	| description                                                            | option    |
|---------------|------------------------------------------------------------------------|-----------|
| Image		| Base model for SISR, supports both Generator and Discriminator         | `image`   |
| OTF     	| Builds on top of `image`, adding Real-ESRGAN on-the-fly degradations	 | `otf`     |

### [supported dataloaders](https://github.com/neosr-project/neosr/wiki/Configuration-Walkthrough#dataset-type):

| loader                                          | option   |
|-------------------------------------------------|----------|
| Paired datasets                                 | `paired` |
| Single datasets (for inference, no GT required) | `single` |
| Real-ESRGAN on-the-fly degradation              | `otf`    |

