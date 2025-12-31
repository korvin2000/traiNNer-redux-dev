# Mixed Precision & Throughput Static Analysis Report

## SECTION 1: CRITICAL EXECUTIVE SUMMARY
1. **The Flaw (sr_model.py:L489-L492):** `gt_for_loss.gt.clone()` dereferences the Tensor method rather than the tensor data, raising an attribute error before the first backward pass. **Consequence:** Training aborts on the first minibatch; no gradients are produced. **Fix:**
   ```python
   # before accumulating losses
- real_images_unaug = gt_for_loss.gt.clone()
+ real_images_unaug = gt_for_loss.clone()
   ```
2. **The Flaw (sr_model.py:L774-L810):** Discriminator gradients are unscaled twice (`unscale_` on the same optimizer at L777 and L785) while still inside autocast, which keeps gradients in fp16 and doubles the unscale kernel launch. **Consequence:** Extra GPU launches and fp16 gradient norms during clipping/monitoring increase overhead and make clipping thresholds unreliable under amp. **Fix:**
   ```python
   # inside discriminator update
   self.scaler_d.scale(l_d_total / self.accum_iters).backward()
   if apply_gradient:
-      self.scaler_d.unscale_(self.optimizer_d)
-      ...
-      self.scaler_d.unscale_(self.optimizer_d)
+      self.scaler_d.unscale_(self.optimizer_d)
+      ...  # monitor / clip using unscaled fp32 grads
   ```
3. **The Flaw (utils/color_util.py:L288-L307,L310-L319):** YCbCr conversion builds weight/bias tensors on every call and immediately casts them to the autocast dtype (fp16/bf16) via `to(img)`. **Consequence:** Repeated host→device copies per batch hurt throughput, and fp16 conversion weights/biases magnify rounding error in the ±128 offsets, causing chroma drift in mixed precision. **Fix:** Register the weights/biases as persistent float32 buffers (or module-level cached tensors) and cast the input to float32 for the conversion path before re‑casting the output.

## SECTION 2: DETAILED ISSUE LOG (Minimum 20 Items)
1. **[Critical] Misreferenced GT tensor:** Loss prep uses `gt_for_loss.gt.clone()` instead of cloning the tensor, causing an exception before loss computation. *Context:* Blocks all training runs.【F:traiNNer/models/sr_model.py†L488-L492】
2. **[Critical] Double unscale of D grads:** `self.scaler_d.unscale_` called twice on the same optimizer within one step, and monitoring happens before casting grads to fp32. *Context:* Adds redundant GPU work and keeps grad stats in half precision, weakening clipping/adaptive control under amp.【F:traiNNer/models/sr_model.py†L774-L808】
3. **[High] Autocast math for YCbCr weights:** Conversion weights/biases are recreated per call and cast to fp16/bf16 inside autocast. *Context:* Host→device tensor churn and fp16 rounding on ±128 offsets degrade color accuracy and throughput.【F:traiNNer/utils/color_util.py†L288-L307】【F:traiNNer/utils/color_util.py†L310-L319】
4. **[High] Distributed loss reduction drops floats:** `reduce_loss_dict` only keeps Tensor entries, discarding scalar floats from `loss_dict`. *Context:* Logs and schedulers relying on float metrics go missing on multi-GPU, desynchronizing training decisions.【F:traiNNer/models/base_model.py†L848-L863】
5. **[High] Loss accumulation in autocast:** `l_g_total` aggregation and dynamic loss weights run entirely inside autocast without promoting to fp32. *Context:* Small-weight losses can underflow in fp16/bf16, biasing total loss and scheduler signals.【F:traiNNer/models/sr_model.py†L483-L668】
6. **[High] GAN real/fake conversions cloned twice:** `fake_images_unaug = self.output.clone()` and augmentation clones occur before the discriminator step even when GAN is disabled. *Context:* Unnecessary GPU memory traffic for non-GAN runs, reducing throughput.【F:traiNNer/models/sr_model.py†L488-L499】
7. **[High] Grad norm computation stacks per-parameter norms:** Building a `torch.stack` over every parameter norm each step allocates a large temporary tensor on GPU. *Context:* Inflates VRAM and kernel launches, especially for large backbones, slowing training.【F:traiNNer/models/sr_model.py†L696-L707】
8. **[High] D grad monitoring before fp32 cast:** Gradient monitoring and clipping happen right after unscale but still in autocast context. *Context:* Stats and clipping use fp16/bf16 values, making thresholds noisy and potentially skipping necessary clipping.【F:traiNNer/models/sr_model.py†L774-L807】
9. **[Medium] `current_accum_iter` unused:** The parameter is never referenced inside `optimize_parameters`. *Context:* Accumulation-aware logic (e.g., logging/EMA cadence) cannot react to micro-steps, leading to misaligned metrics or EMA updates.【F:traiNNer/models/sr_model.py†L452-L724】
10. **[Medium] EMA update gated only by optimizer skip:** EMA is skipped when the generator step underflows, but not when `apply_gradient` is False during accumulation. *Context:* EMA lags behind real updates during gradient accumulation, hurting eval quality.【F:traiNNer/models/sr_model.py†L832-L834】
11. **[Medium] RGB↔YCbCr constants rebuilt per call:** Weight/bias tensors are reallocated for every conversion call. *Context:* Adds CPU/GPU overhead in both training and validation loops; better cached as buffers.【F:traiNNer/utils/color_util.py†L288-L305】【F:traiNNer/utils/color_util.py†L310-L319】
12. **[Medium] No epsilon in YCbCr division:** Conversions divide by 255 without epsilon and operate in half precision under autocast. *Context:* Rounding to zero for dark pixels in fp16 can introduce banding; safer to upcast to fp32 with epsilon.【F:traiNNer/utils/color_util.py†L306-L307】
13. **[Medium] Validation tiling weight map in fp32 only:** `infer_tiled` builds `weight_map` in default fp32 while the rest may run in amp. *Context:* Implicit dtype conversions per tile and extra memory; should align with amp dtype or preallocate once.【F:traiNNer/models/sr_model.py†L857-L866】
14. **[Medium] Missing detach on EMA targets for LDL:** LDL loss uses EMA output without detaching before `loss` call. *Context:* If a loss accidentally retains graph references, EMA buffer could track gradients, wasting memory.【F:traiNNer/models/sr_model.py†L530-L557】
15. **[Medium] Gradient clipping threshold reuse:** `clip_grad_norm_` uses dynamic threshold but value is not logged for discriminator, and duplicates unscale. *Context:* Harder to diagnose exploding D grads and wastes work.【F:traiNNer/models/sr_model.py†L804-L813】
16. **[Medium] Loss weights use `abs(loss_weight)`:** Negative weights trigger LQ targets but sign is discarded when accumulating. *Context:* Cannot express intentional negative contributions (e.g., adversarial balancing), potentially mis-training custom losses.【F:traiNNer/models/sr_model.py†L639-L663】
17. **[Medium] Missing overflow guard on `weight_map` linspace:** `torch.linspace` in amp can produce subnormal weights for large tiles. *Context:* Can underflow in fp16, leading to uneven blending seams during tiled inference.【F:traiNNer/models/sr_model.py†L861-L866】
18. **[Medium] Discriminator step uses generator output computed in prior autocast without fresh autocast guard:** `self.output` generated in a different autocast context is reused. *Context:* dtype may be fp32 when amp disabled mid-epoch, causing unexpected upcasting in D forward.【F:traiNNer/models/sr_model.py†L725-L775】
19. **[Medium] Training state resume asserts scalers even when amp disabled:** `resume_training` asserts scaler existence if keys exist, but checkpoints may include scalers while `use_amp` is False. *Context:* Resuming non-amp runs from amp checkpoints will crash, preventing recovery.【F:traiNNer/models/base_model.py†L812-L820】
20. **[Medium] No guard against zero `loss_samples`:** `get_current_log` divides by `self.loss_samples` without checking zero after validation-only epochs. *Context:* Potential division by zero when logs are requested before any training batch.【F:traiNNer/models/base_model.py†L126-L137】

## SECTION 3: FP16 & NUMERICAL STABILITY CHECK
- **Epsilon on denominators:** None found in YCbCr conversions; divisions by 255 occur in the autocast path without stabilization.【F:traiNNer/utils/color_util.py†L306-L307】
- **GradScaler usage:** Generator and discriminator share GradScaler but discriminator unscales twice and monitors gradients in mixed precision, risking unstable clipping.【F:traiNNer/models/sr_model.py†L774-L808】
- **Large reductions:** Loss aggregation and distributed reductions run under autocast (fp16/bf16) without explicit fp32 casting; dynamic-loss weights are multiplied in half precision before backward, inviting underflow.【F:traiNNer/models/sr_model.py†L483-L668】【F:traiNNer/models/base_model.py†L848-L863】
