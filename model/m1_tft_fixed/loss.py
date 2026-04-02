from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from pytorch_forecasting.metrics import QuantileLoss


@dataclass
class LossSpec:
    quantiles: Sequence[float]


def build_loss(spec: LossSpec) -> QuantileLoss:
    return QuantileLoss(quantiles=list(spec.quantiles))
