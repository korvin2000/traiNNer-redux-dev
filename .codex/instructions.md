# Codex Authoring Guidelines

## Model
- model: gpt-5.2-codex (xmas variant preferred)
- model_reasoning_effort: high
- approval_policy: never

## Output Expectations
- Produce compact, well-structured code and text with comments and docs while preserving clarity through naming and decomposition.
- Prefer reliable, modern language and framework features with an emphasis on speed and runtime performance.
- Apply clean code and best design patterns and best practices; keep methods small and cohesive.
- Avoid any tests and test frameworks unless explicitly required.
- Prefer tables or tidy lists for reference-style documentation.

## Project Context
- description: "Deep learning training framework for image super resolution and restoration."
- Target Python >= 12; torch>=2.9.
- Optimize for maintainability: clear documentation, registry-driven configuration points, and minimal glue code.
- keywords: super-resolution, machine-learning, image-restoration, pytorch, computer vision (cv), safetensors, cuda, esrgan.

## Project topology and registries
- Entrypoints: `train.py` (training), `test.py` (evaluation/inference), `convert_to_onnx.py` (export). Runtime options live in `options/` and are schema-validated via JSON schemas in `schemas/`.
- Core packages: `traiNNer/archs` (networks), `traiNNer/models` (training logic), `traiNNer/data` (datasets/pipelines), `traiNNer/losses`, `traiNNer/metrics`, `traiNNer/optimizers`, `traiNNer/schedulers`, and shared utilities in `traiNNer/utils`.
- Documentation: `docs/source/arch_reference.md` holds per-architecture YAML parameter templates; update alongside architectural changes.
- Performance taxonomy: `architecture_categories.json` ranks generators by speed/VRAM to guide light vs. heavy configurations.
- Generator registry (auto-imports every `*_arch.py`): `artcnn_arch`, `atd_arch`, `autoencoder_arch`, `cgnet_arch`, `craft_arch`, `dat_arch`, `dctlsa_arch`, `detailrefinernet_arch`, `dis_arch`, `ditn_arch`, `drct_arch`, `dwt_arch`, `eimn_arch`, `elan_arch`, `elysiumsr_arch`, `emt_arch`, `escreal_arch`, `fdat_arch`, `flexnet_arch`, `gaterv3_arch`, `gfisrv2_arch`, `grl_arch`, `hat_arch`, `hit_sir_arch`, `hit_sng_arch`, `hit_srf_arch`, `hyperionsr_arch`, `lawfft_arch`, `lkfmixer_arch`, `lmlt_arch`, `lpips_arch`, `man_arch`, `metagan3_arch`, `moesr_arch`, `mosr_arch`, `mosrv2_arch`, `munet_arch`, `omnisr_arch`, `paragondiffusion_arch`, `paragonsr2_arch`, `paragonsr_arch`, `plksr_arch`, `rcan_arch`, `realcugan_arch`, `realplksr_arch`, `rgt_arch`, `rrdbnet_arch`, `rtmosr_arch`, `safmn_arch`, `scunet_aaf6aa_arch`, `sebica_arch`, `seemore_arch`, `span_arch`, `spanplus_arch`, `spanpp_arch`, `srformer_arch`, `srvgg_arch`, `swin2sr_arch`, `swinir_arch`, `temporal_span_arch`, `temporal_span_v2_arch`, `topiq_arch`, `tscunet_arch`, `unetsegmentation_arch`, `vgg_arch`, plus discriminator modules (`discriminator_arch`, `patchgan_arch`, `dunet_arch`, `metagan2_arch`).
- Discriminators: `VGGStyleDiscriminator`, `UNetDiscriminatorSN` (spectral-normalized U-Net), `PatchGANDiscriminatorSN`, `MultiscalePatchGANDiscriminatorSN`, `DUnet`, and `MetaGan2`.
- Optimizers: custom `Adan`, `AdanScheduleFree`, `AdamWScheduleFree`, plus registry-registered PyTorch/third-party options (`Adam`, `AdamW`, `NAdam`, `StableAdamW`, `SOAP`, `ADOPT`).
- Losses: adaptive_block_tv_loss, adists_loss, aesop_loss, basic_loss, bcewithlogitsdice_loss, chc_loss, checkerboard_loss, consistency_loss, contextual_loss, contrastive_loss, convnext_perceptual_loss, cosim_loss, dino_perceptual_loss, dists_loss, feature_matching_loss, flip_loss, focal_frequency_loss, gan_loss (R3GAN-aware), gradient_variance_loss, hfen_loss, laplacian_loss, ldl_loss, line_distiller_loss, ms_ssim_l1_loss, mssim_loss, ncc_loss, perceptual_anime_loss, perceptual_fp16_loss, perceptual_loss, r3gan_loss, tv_loss, with iterative/dynamic scheduling wrappers for staged training.
- Metrics: `calculate_psnr`/`calculate_ssim` (psnr_ssim), `calculate_lpips`, `calculate_dists`, `calculate_topiq`, plus helpers in `metric_util`.

## Reasoning playbook
- Work in engineer mode: use explicit hypothesis -> plan -> refine -> verify loops (CoT, PoT, ToT, CoD, CoR, CoV) and summarize intermediate conclusions.
- Preserve registry decorators in `*_arch.py`, `*_loss.py`, and `*_optim.py` to keep dynamic loading intact.
- Avoid running tests or linters unless explicitly requested; focus on configuration/doc changes when possible.
