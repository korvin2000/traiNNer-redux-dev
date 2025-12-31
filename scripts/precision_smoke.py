"""Synthetic precision smoke test.

Runs a few forward/backward/step iterations under fp32, fp16, and bf16
(if supported) to spot AMP regressions without requiring datasets.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Iterable

import torch
from torch import nn
from torch.cuda.amp import GradScaler, autocast


@dataclass
class PrecisionResult:
    mode: str
    loss: float
    overflow: bool


class TinyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 8, 3, padding=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(8, 3, 3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
        return self.net(x)


def run_steps(dtype: torch.dtype, device: torch.device, steps: int) -> PrecisionResult:
    model = TinyModel().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    scaler = GradScaler(enabled=dtype != torch.float32, device=device.type)
    loss_val: float = 0.0
    overflow = False

    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        with autocast(device_type=device.type, dtype=dtype, enabled=dtype != torch.float32):
            inputs = torch.randn(2, 3, 16, 16, device=device)
            targets = torch.randn_like(inputs)
            outputs = model(inputs)
            loss = torch.nn.functional.mse_loss(outputs, targets)

        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scale_before = scaler.get_scale()
        scaler.step(opt)
        scaler.update()
        overflow = overflow or scaler.get_scale() < scale_before
        loss_val += loss.detach().float().item()

    model.eval()
    with torch.no_grad():
        with autocast(device_type=device.type, dtype=dtype, enabled=dtype != torch.float32):
            val_out = model(torch.randn(1, 3, 16, 16, device=device))
            _ = val_out.mean().item()

    return PrecisionResult(mode=str(dtype), loss=loss_val / steps, overflow=overflow)


def main(modes: Iterable[str], steps: int) -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for AMP smoke test")

    device = torch.device("cuda")
    dtype_map = {
        "fp32": torch.float32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }

    for mode in modes:
        dtype = dtype_map[mode]
        if mode == "bf16" and not torch.cuda.is_bf16_supported():
            print("Skipping bf16: not supported on this GPU")
            continue
        result = run_steps(dtype, device, steps)
        print(f"{mode}: loss={result.loss:.4f} overflow={result.overflow}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--modes",
        nargs="+",
        default=["fp32", "fp16", "bf16"],
        choices=["fp32", "fp16", "bf16"],
        help="Precision modes to run",
    )
    parser.add_argument("--steps", type=int, default=3, help="Number of train steps")
    args = parser.parse_args()
    main(args.modes, args.steps)
