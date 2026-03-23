from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from pytorch_forecasting.metrics import QuantileLoss


@dataclass
class LossSpec:
    name: str
    quantiles: Sequence[float]
    lambda_strategy: str = "vix_linear"
    lambda_min: float = 1.0
    lambda_max: float = 3.0
    lambda_alpha: float = 0.8
    signal_clip: float = 3.0
    quantile_downside_gamma: float = 0.5
    downside_temperature: float = 0.02


class VIXWeightedAdaptiveQuantileLoss(QuantileLoss):
    """
    Adaptive quantile loss for TFT-VIX.

    Uses train-normalized decoder features for VIX only:

        lambda_t = clip(1 + alpha * relu(z_vix_t), lambda_min, lambda_max)
        downside_gate_t = sigmoid(-y_t / tau)
        quantile_weight_q = 1 + gamma * relu((0.5 - q) / 0.4)
        total_weight_tq = 1 + downside_gate_t * (lambda_t - 1) * quantile_weight_q

    The final weighted loss is:

        L_tq = total_weight_tq * 2 * max((q - 1) * e_tq, q * e_tq)
        e_tq = y_t - y_hat_tq
    """

    def __init__(
        self,
        quantiles: Sequence[float],
        lambda_strategy: str = "vix_linear",
        lambda_min: float = 1.0,
        lambda_max: float = 3.0,
        lambda_alpha: float = 0.8,
        signal_clip: float = 3.0,
        quantile_downside_gamma: float = 0.5,
        downside_temperature: float = 0.02,
        **kwargs,
    ):
        super().__init__(quantiles=list(quantiles), **kwargs)
        self.lambda_strategy = lambda_strategy
        self.lambda_min = float(lambda_min)
        self.lambda_max = float(lambda_max)
        self.lambda_alpha = float(lambda_alpha)
        self.signal_clip = float(signal_clip)
        self.quantile_downside_gamma = float(quantile_downside_gamma)
        self.downside_temperature = float(downside_temperature)

    def compute_quantile_weights(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        quantiles = torch.tensor(self.quantiles, device=device, dtype=dtype)
        downside_focus = torch.relu((0.5 - quantiles) / 0.4)
        return 1.0 + self.quantile_downside_gamma * downside_focus

    def _sanitize_signal(self, signal: torch.Tensor) -> torch.Tensor:
        return torch.clamp(
            torch.nan_to_num(signal.detach(), nan=0.0, posinf=self.signal_clip, neginf=-self.signal_clip),
            min=-self.signal_clip,
            max=self.signal_clip,
        )

    def compute_lambda_t(self, vix_signal: torch.Tensor) -> torch.Tensor:
        vix_signal = self._sanitize_signal(vix_signal)

        if self.lambda_strategy == "vix_linear":
            raw = 1.0 + self.lambda_alpha * torch.relu(vix_signal)
        elif self.lambda_strategy == "vix_sigmoid":
            raw = self.lambda_min + (self.lambda_max - self.lambda_min) * torch.sigmoid(self.lambda_alpha * vix_signal)
            return raw.unsqueeze(-1)
        elif self.lambda_strategy == "vix_threshold":
            raw = 1.0 + self.lambda_alpha * (vix_signal > 0).to(vix_signal.dtype)
            raw = torch.clamp(raw, min=self.lambda_min, max=self.lambda_max)
            return raw.unsqueeze(-1)
        else:
            raise ValueError(f"Unsupported lambda strategy: {self.lambda_strategy}")

        raw = torch.clamp(raw, min=self.lambda_min, max=self.lambda_max)
        return raw.unsqueeze(-1)

    def compute_weighted_loss(
        self,
        y_pred: torch.Tensor,
        target: torch.Tensor,
        vix_signal: torch.Tensor,
    ) -> torch.Tensor:
        base_losses = self.loss(y_pred, target)
        lambda_t = self.compute_lambda_t(vix_signal=vix_signal)
        quantile_weights = self.compute_quantile_weights(device=target.device, dtype=target.dtype).view(1, 1, -1)
        temperature = max(self.downside_temperature, 1e-4)
        downside_gate = torch.sigmoid(-torch.nan_to_num(target.detach(), nan=0.0).unsqueeze(-1) / temperature)
        total_weight = 1.0 + downside_gate * (lambda_t - 1.0) * quantile_weights
        return base_losses * total_weight


def build_loss(spec: LossSpec):
    if spec.name == "quantile":
        return QuantileLoss(quantiles=list(spec.quantiles))
    if spec.name == "adaptive_quantile":
        return VIXWeightedAdaptiveQuantileLoss(
            quantiles=list(spec.quantiles),
            lambda_strategy=spec.lambda_strategy,
            lambda_min=spec.lambda_min,
            lambda_max=spec.lambda_max,
            lambda_alpha=spec.lambda_alpha,
            signal_clip=spec.signal_clip,
            quantile_downside_gamma=spec.quantile_downside_gamma,
            downside_temperature=spec.downside_temperature,
        )
    raise ValueError(f"Unsupported loss name: {spec.name}")
