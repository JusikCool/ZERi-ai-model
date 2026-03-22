from __future__ import annotations

from pytorch_forecasting import TemporalFusionTransformer

from .config import TFTVIXConfig
from .losses import LossSpec, build_loss


def build_tft_model(train_dataset, config: TFTVIXConfig) -> TemporalFusionTransformer:
    loss = build_loss(LossSpec(name=config.loss_name, quantiles=config.quantiles))
    return TemporalFusionTransformer.from_dataset(
        train_dataset,
        learning_rate=config.learning_rate,
        hidden_size=config.hidden_size,
        attention_head_size=config.attention_head_size,
        dropout=config.dropout,
        hidden_continuous_size=config.hidden_continuous_size,
        output_size=len(config.quantiles),
        loss=loss,
        log_interval=10,
        reduce_on_plateau_patience=3,
    )
