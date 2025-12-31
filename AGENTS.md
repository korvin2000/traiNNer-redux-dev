# Agent Guidelines

- Use the `GPT-5.2-Codex-XMas` or `GPT-5.2-Codex` model for reasoning and code + documentation generation when applicable.
- Act as a senior PyTorch engineer and experienced python developer.
- Default to engineer mode with modern reasoning patterns (CoT, PoT, ToT, CoD, CoR, Self-Refine, CoV, priority hierarchy, intermediate summarization) and state clearly when assumptions are made.
- Favor clear, readable, and compact implementations that follow clean code principles.
- Prefer well structured output like tables if writing documentation.
- If unsure about a detail, make a best-effort assumption and clearly state it.

## Project map (traiNNer-redux)
- Entry points: `train.py` for training, `test.py` for inference/validation flows, and `convert_to_onnx.py` for export; options live under `options/` with JSON schemas in `schemas/` to validate configs.
- Core packages: architectures in `traiNNer/archs/`, data loaders under `traiNNer/data/`, models in `traiNNer/models/`, losses in `traiNNer/losses/`, metrics in `traiNNer/metrics/`, optimizers in `traiNNer/optimizers/`, and schedulers in `traiNNer/schedulers/`.
- Documentation sources: `docs/source/` includes the architecture reference with per-model YAML parameter templates.
- Performance taxonomy: `architecture_categories.json` groups generators by speed/VRAM to guide lightweight vs. heavyweight choices.

## Model zoo quick reference
- Generator registry auto-loads every `*_arch.py` in `traiNNer/archs/`. Current generator options include (alphabetical): `artcnn_arch`, `atd_arch`, `autoencoder_arch`, `cgnet_arch`, `craft_arch`, `dat_arch`, `dctlsa_arch`, `detailrefinernet_arch`, `dis_arch`, `ditn_arch`, `drct_arch`, `dwt_arch`, `eimn_arch`, `elan_arch`, `elysiumsr_arch`, `emt_arch`, `escreal_arch`, `fdat_arch`, `flexnet_arch`, `gaterv3_arch`, `gfisrv2_arch`, `grl_arch`, `hat_arch`, `hit_sir_arch`, `hit_sng_arch`, `hit_srf_arch`, `hyperionsr_arch`, `lawfft_arch`, `lkfmixer_arch`, `lmlt_arch`, `lpips_arch`, `man_arch`, `metagan3_arch`, `moesr_arch`, `mosr_arch`, `mosrv2_arch`, `munet_arch`, `omnisr_arch`, `paragondiffusion_arch`, `paragonsr2_arch`, `paragonsr_arch`, `plksr_arch`, `rcan_arch`, `realcugan_arch`, `realplksr_arch`, `rgt_arch`, `rrdbnet_arch`, `rtmosr_arch`, `safmn_arch`, `scunet_aaf6aa_arch`, `sebica_arch`, `seemore_arch`, `span_arch`, `spanplus_arch`, `spanpp_arch`, `srformer_arch`, `srvgg_arch`, `swin2sr_arch`, `swinir_arch`, `temporal_span_arch`, `temporal_span_v2_arch`, `topiq_arch`, `tscunet_arch`, `unetsegmentation_arch`, `vgg_arch`, and discriminator-related modules (`discriminator_arch`, `patchgan_arch`, `dunet_arch`, `metagan2_arch`).
- Discriminators: `VGGStyleDiscriminator` and `UNetDiscriminatorSN` (`discriminator_arch.py`), `PatchGANDiscriminatorSN` and `MultiscalePatchGANDiscriminatorSN` (`patchgan_arch.py`), spectral-norm U-Net variant `DUnet` (`dunet_arch.py`), and spectral MetaGan discriminator `MetaGan2` (`metagan2_arch.py`).
- Optimizers (registry + built-ins): custom `Adan`, `AdanScheduleFree`, `AdamWScheduleFree`, plus PyTorch `Adam`, `AdamW`, `NAdam` and pytorch-optimizer `StableAdamW`, `SOAP`, `ADOPT`.
- Loss suite (`traiNNer/losses`): adaptive_block_tv_loss, adists_loss, aesop_loss, basic_loss, bcewithlogitsdice_loss, chc_loss, checkerboard_loss, consistency_loss, contextual_loss, contrastive_loss, convnext_perceptual_loss, cosim_loss, dino_perceptual_loss, dists_loss, feature_matching_loss, flip_loss, focal_frequency_loss, gan_loss (with R3GAN handling), gradient_variance_loss, hfen_loss, laplacian_loss, ldl_loss, line_distiller_loss, ms_ssim_l1_loss, mssim_loss, ncc_loss, perceptual_anime_loss, perceptual_fp16_loss, perceptual_loss, r3gan_loss, tv_loss, plus scheduling wrappers (`iterative_loss_wrapper`, `dynamic_loss_scheduling`).
- Metrics: PSNR/SSIM (`psnr_ssim`), LPIPS (`lpips`), DISTS (`dists`), TOPIQ (`topiq`), with helper utilities in `metric_util`.

## Collaboration tips
- Keep edits configuration-first: prefer touching `options/` YAMLs and schemas for new experiments, then adjust `traiNNer/` modules.
- Document architectural changes in `docs/source/arch_reference.md` with succinct YAML snippets when adding parameters.
- When refactoring, preserve registry hooks (decorators in `*_arch.py`, `*_loss.py`, `*_optim.py`) to keep dynamic loading intact.
