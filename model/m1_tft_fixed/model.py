from __future__ import annotations

from pytorch_forecasting import TemporalFusionTransformer

try:
    from .config import TFTFixedConfig
    from .loss import LossSpec, build_loss
except ImportError:  # pragma: no cover - direct script execution
    from config import TFTFixedConfig
    from loss import LossSpec, build_loss


def build_tft_model(train_dataset, config: TFTFixedConfig) -> TemporalFusionTransformer:
    loss = build_loss(LossSpec(quantiles=config.quantiles))
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


def load_tft_model_from_checkpoint(
    checkpoint_path: str, config: TFTFixedConfig
) -> TemporalFusionTransformer:
    del config
    return TemporalFusionTransformer.load_from_checkpoint(
        checkpoint_path, weights_only=False
    )
