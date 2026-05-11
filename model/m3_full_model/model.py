import torch
import pytorch_lightning as pl
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.metrics import QuantileLoss

from .loss import AdaptivePinballLoss


class M3FullModel(pl.LightningModule):
    def __init__(
        self,
        tft: TemporalFusionTransformer,
        adaptive_loss: AdaptivePinballLoss,
        vix_encoder_idx: int,
        sigma_encoder_idx: int,
        learning_rate: float = 0.001,
        vix_mean: float = 0.0,
        vix_std: float = 1.0,
    ):
        super().__init__()
        self.tft = tft
        self.adaptive_loss = adaptive_loss
        self.vix_encoder_idx = vix_encoder_idx
        self.sigma_encoder_idx = sigma_encoder_idx
        self.learning_rate = learning_rate
        self.vix_mean = vix_mean
        self.vix_std = vix_std

    @classmethod
    def from_dataset(
        cls,
        dataset: TimeSeriesDataSet,
        learning_rate: float = 0.001,
        hidden_size: int = 64,
        attention_head_size: int = 4,
        dropout: float = 0.1,
        hidden_continuous_size: int = 32,
        quantiles: list[float] = None,
        vix_threshold: float = 25.0,
        vix_mean: float = 0.0,
        vix_std: float = 1.0,
        sigma_scale: float = 1.0,
        alpha_down: float = 1.0,
        beta_down: float = 1.0,
        alpha_up: float = 1.0,
        beta_up: float = 1.0,
        crossing_weight: float = 0.1,
    ) -> "M3FullModel":
        quantiles = quantiles or [0.05,0.10,0.15,0.20,0.25,0.30,
             0.35,0.40,0.45,0.50,0.55,
             0.60,0.65,0.70,0.75,0.80,
             0.85,0.90,0.95]

        reals: list[str] = dataset.reals
        vix_idx = reals.index("VIX_Close")
        sigma_idx = reals.index("Realized_Vol_20d")

        tft = TemporalFusionTransformer.from_dataset(
            dataset,
            learning_rate=learning_rate,
            hidden_size=hidden_size,
            attention_head_size=attention_head_size,
            dropout=dropout,
            hidden_continuous_size=hidden_continuous_size,
            output_size=len(quantiles),
            loss=QuantileLoss(quantiles=quantiles),
            log_interval=10,
            reduce_on_plateau_patience=4,
        )

        adaptive_loss = AdaptivePinballLoss(
            quantiles=quantiles,
            vix_threshold=vix_threshold,
            vix_scale=vix_std,
            sigma_scale=sigma_scale,
            alpha_down=alpha_down,
            beta_down=beta_down,
            alpha_up=alpha_up,
            beta_up=beta_up,
            crossing_weight=crossing_weight,
        )

        return cls(
            tft=tft,
            adaptive_loss=adaptive_loss,
            vix_encoder_idx=vix_idx,
            sigma_encoder_idx=sigma_idx,
            learning_rate=learning_rate,
            vix_mean=vix_mean,
            vix_std=vix_std,
        )

    def forward(self, x: dict) -> dict:
        return self.tft(x)

    def _extract_vix_sigma(
        self, x: dict, pred_len: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        enc = x["encoder_cont"]
        vix_norm = enc[:, -1, self.vix_encoder_idx]
        vix_raw = vix_norm * self.vix_std + self.vix_mean
        vix = vix_raw.unsqueeze(1).expand(-1, pred_len)
        sigma = enc[:, -1, self.sigma_encoder_idx].unsqueeze(1).expand(-1, pred_len)
        return vix, sigma

    def training_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
        x, y = batch
        y_true = y[0]
        pred_len = y_true.shape[1]

        out = self.tft(x)
        y_pred = out["prediction"]

        vix, sigma = self._extract_vix_sigma(x, pred_len)
        loss = self.adaptive_loss(y_pred, y_true, vix, sigma)

        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
        x, y = batch
        y_true = y[0]
        pred_len = y_true.shape[1]

        out = self.tft(x)
        y_pred = out["prediction"]

        vix, sigma = self._extract_vix_sigma(x, pred_len)
        loss = self.adaptive_loss(y_pred, y_true, vix, sigma)

        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def predict(self, x: dict) -> torch.Tensor:
        self.eval()
        with torch.no_grad():
            out = self.tft(x)
        return out["prediction"]

    def configure_optimizers(self) -> dict:
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", patience=4, factor=0.5
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "monitor": "val_loss"},
        }
