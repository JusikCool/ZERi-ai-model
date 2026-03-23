from __future__ import annotations

from pytorch_forecasting import TemporalFusionTransformer

from .config import TFTVIXConfig
from .losses import LossSpec, VIXWeightedAdaptiveQuantileLoss, build_loss


class AdaptiveLossTemporalFusionTransformer(TemporalFusionTransformer):
    def _compute_adaptive_loss(self, prediction, y, x):
        if isinstance(y, (tuple, list)):
            target = y[0]
            weight = y[1]
        else:
            target = y
            weight = None

        vix_idx = self.hparams.x_reals.index("VIX_Close")
        vix_signal = x["decoder_cont"][..., vix_idx]

        losses = self.loss.compute_weighted_loss(
            y_pred=prediction,
            target=target,
            vix_signal=vix_signal,
        )

        if weight is not None:
            losses = losses * weight.unsqueeze(-1)

        losses = self.loss.mask_losses(losses, x["decoder_lengths"])
        return self.loss.reduce_loss(losses, lengths=x["decoder_lengths"])

    def step(self, x, y, batch_idx: int, **kwargs):
        out = self(x, **kwargs)
        prediction = out["prediction"]

        if self.predicting:
            loss = None
        elif isinstance(self.loss, VIXWeightedAdaptiveQuantileLoss):
            loss = self._compute_adaptive_loss(prediction=prediction, y=y, x=x)
        else:
            loss = self.loss(prediction, y)

        self.log(
            f"{self.current_stage}_loss",
            loss,
            on_step=self.training,
            on_epoch=True,
            prog_bar=True,
            batch_size=len(x["decoder_target"]),
        )
        log = {"loss": loss, "n_samples": x["decoder_lengths"].size(0)}
        return log, out


def build_tft_model(train_dataset, config: TFTVIXConfig) -> TemporalFusionTransformer:
    loss = build_loss(
        LossSpec(
            name=config.loss_name,
            quantiles=config.quantiles,
            lambda_strategy=config.lambda_strategy,
            lambda_min=config.lambda_min,
            lambda_max=config.lambda_max,
            lambda_alpha=config.lambda_alpha,
            signal_clip=config.signal_clip,
            quantile_downside_gamma=config.quantile_downside_gamma,
            downside_temperature=config.downside_temperature,
        )
    )
    model_cls = AdaptiveLossTemporalFusionTransformer if config.loss_name == "adaptive_quantile" else TemporalFusionTransformer
    return model_cls.from_dataset(
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
