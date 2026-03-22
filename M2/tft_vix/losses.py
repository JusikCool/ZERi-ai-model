from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from pytorch_forecasting.metrics import QuantileLoss


@dataclass
class LossSpec:
    name: str
    quantiles: Sequence[float]


class AdaptiveLossBuilder:
    """Reserved extension point for the future VIX-weighted adaptive loss."""

    def build(self):
        raise NotImplementedError(
            "Adaptive loss is not enabled in this stage. "
            "Implement a pytorch_forecasting-compatible metric here later."
        )


def build_loss(spec: LossSpec):
    if spec.name == "quantile":
        return QuantileLoss(quantiles=list(spec.quantiles))
    if spec.name == "adaptive_quantile":
        return AdaptiveLossBuilder().build()
    raise ValueError(f"Unsupported loss name: {spec.name}")
