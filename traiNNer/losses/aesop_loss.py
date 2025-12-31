from typing import Literal

import torch
from safetensors.torch import load_file
from torch import Tensor, nn

from traiNNer.archs.autoencoder_arch import AutoEncoder
from traiNNer.losses.basic_loss import L1Loss, charbonnier_loss
from traiNNer.losses.ms_ssim_l1_loss import MSSSIML1Loss
from traiNNer.utils.registry import LOSS_REGISTRY


@LOSS_REGISTRY.register()
class AESOPLoss(nn.Module):
    def __init__(
        self,
        loss_weight: float,
        scale: int,
        pretrain_network_ae: str,
        criterion: Literal["l1", "charbonnier", "msssiml1"] = "charbonnier",
    ) -> None:
        super().__init__()
        self.loss_weight = loss_weight
        self.ae = AutoEncoder(freeze_encoder=True, freeze_decoder=True, scale=scale)

        if pretrain_network_ae.endswith(".safetensors"):
            state_dict = load_file(pretrain_network_ae)
        else:
            state_dict = torch.load(
                pretrain_network_ae, map_location="cpu", weights_only=True
            )

        self.ae.load_state_dict(state_dict)

        if criterion == "l1":
            self.criterion = L1Loss(1.0)
        elif criterion == "charbonnier":
            self.criterion = charbonnier_loss
        else:
            self.criterion = MSSSIML1Loss(1.0)

    @staticmethod
    def _load_state(path: str) -> dict:
        if path.endswith(".safetensors"):
            return load_file(path)

        ckpt = torch.load(path, map_location="cpu", weights_only=True)
        if isinstance(ckpt, dict) and "params" in ckpt:
            return ckpt["params"]
        return ckpt

    def forward(self, sr: Tensor, hr: Tensor) -> Tensor:
        ae_sr = self.ae(sr)
        ae_hr = self.ae(hr.detach())
        return self.criterion(ae_sr, ae_hr)
