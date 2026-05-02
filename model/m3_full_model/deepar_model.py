import torch
import pytorch_lightning as pl
from pytorch_forecasting import DeepAR, TimeSeriesDataSet
from pytorch_forecasting.metrics import NormalDistributionLoss

from .loss import AdaptivePinballLoss


class M3FullModel(pl.LightningModule):
    """
    DeepAR + AdaptivePinballLoss 조합.

    DeepAR은 distribution 파라미터(loc, scale)를 출력하고, 학습 시에는 일반적으로
    NormalDistributionLoss(NLL)로 학습된다. 본 클래스는 다른 모델들과 동일한
    AdaptivePinballLoss로 학습하기 위해, 각 step마다 distribution loss의
    to_quantiles() 메서드로 quantile (B, T, Q)을 analytic하게 계산한 뒤
    AdaptivePinballLoss에 전달한다.
    """

    def __init__(
        self,
        deepar: DeepAR,
        adaptive_loss: AdaptivePinballLoss,
        distribution_loss: NormalDistributionLoss,
        vix_encoder_idx: int,
        sigma_encoder_idx: int,
        quantiles: list[float],
        learning_rate: float = 0.001,
        vix_mean: float = 0.0,
        vix_std: float = 1.0,
    ):
        super().__init__()
        self.deepar = deepar
        self.adaptive_loss = adaptive_loss
        # NormalDistributionLoss 인스턴스를 quantile 변환용으로 보관
        self.distribution_loss = distribution_loss
        self.vix_encoder_idx = vix_encoder_idx
        self.sigma_encoder_idx = sigma_encoder_idx
        self.quantiles = quantiles
        self.learning_rate = learning_rate
        self.vix_mean = vix_mean
        self.vix_std = vix_std

    @classmethod
    def from_dataset(
        cls,
        dataset: TimeSeriesDataSet,
        learning_rate: float = 0.001,
        hidden_size: int = 30,
        rnn_layers: int = 2,
        cell_type: str = "LSTM",
        dropout: float = 0.1,
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
        quantiles = quantiles or [0.1, 0.5, 0.9]

        reals: list[str] = dataset.reals
        vix_idx = reals.index("VIX_Close")
        sigma_idx = reals.index("Realized_Vol_20d")

        # DeepAR이 학습 중에 사용하는 NLL loss. 모델 내부에서 distribution 파라미터를 학습.
        # quantile 변환에도 동일한 인스턴스를 재활용.
        distribution_loss = NormalDistributionLoss(quantiles=quantiles)

        deepar = DeepAR.from_dataset(
            dataset,
            learning_rate=learning_rate,
            hidden_size=hidden_size,
            rnn_layers=rnn_layers,
            cell_type=cell_type,
            dropout=dropout,
            loss=distribution_loss,
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
            deepar=deepar,
            adaptive_loss=adaptive_loss,
            distribution_loss=distribution_loss,
            vix_encoder_idx=vix_idx,
            sigma_encoder_idx=sigma_idx,
            quantiles=quantiles,
            learning_rate=learning_rate,
            vix_mean=vix_mean,
            vix_std=vix_std,
        )

    def forward(self, x: dict) -> dict:
        return self.deepar(x)

    def _extract_vix_sigma(
        self, x: dict, pred_len: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        enc = x["encoder_cont"]
        vix_norm = enc[:, -1, self.vix_encoder_idx]
        vix_raw = vix_norm * self.vix_std + self.vix_mean
        vix = vix_raw.unsqueeze(1).expand(-1, pred_len)
        sigma = enc[:, -1, self.sigma_encoder_idx].unsqueeze(1).expand(-1, pred_len)
        return vix, sigma

    def _to_quantile_tensor(self, prediction: torch.Tensor) -> torch.Tensor:
        """
        DeepAR forward의 prediction (distribution parameters)을
        AdaptivePinballLoss가 기대하는 (B, T, Q) shape으로 변환.
        NormalDistributionLoss.to_quantiles()는 analytic하게 quantile을 계산하므로
        gradient가 잘 흐른다 (sampling보다 안정적).
        """
        return self.distribution_loss.to_quantiles(prediction, quantiles=self.quantiles)

    def training_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
        x, y = batch
        y_true = y[0]
        pred_len = y_true.shape[1]

        out = self.deepar(x)
        # DeepAR은 'prediction' key에 distribution parameters를 담아서 반환
        y_pred = self._to_quantile_tensor(out["prediction"])

        vix, sigma = self._extract_vix_sigma(x, pred_len)
        loss = self.adaptive_loss(y_pred, y_true, vix, sigma)

        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
        x, y = batch
        y_true = y[0]
        pred_len = y_true.shape[1]

        out = self.deepar(x)
        y_pred = self._to_quantile_tensor(out["prediction"])

        vix, sigma = self._extract_vix_sigma(x, pred_len)
        loss = self.adaptive_loss(y_pred, y_true, vix, sigma)

        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def predict(self, x: dict) -> torch.Tensor:
        """
        다른 모델들과 동일하게 (B, T, Q) shape의 quantile tensor를 반환.
        백테스트 코드가 model.predict(x)로 quantile 직접 받아 처리하기 위함.
        """
        self.eval()
        with torch.no_grad():
            out = self.deepar(x)
            y_pred = self._to_quantile_tensor(out["prediction"])
        return y_pred

    def configure_optimizers(self) -> dict:
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", patience=4, factor=0.5
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "monitor": "val_loss"},
        }