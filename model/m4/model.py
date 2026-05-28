import torch
import pytorch_lightning as pl
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.metrics import QuantileLoss

from .loss import AdaptivePinballLoss


class M4FullModel(pl.LightningModule):
    """
    M4 확장 모델.

    [M3 호환 모드] use_garch_sigma=False, vol_group_map=None
      → 기존 M3 와 동일 (Realized_Vol_20d, scalar α/β)

    [M4-GARCH only] use_garch_sigma=True, vol_group_map=None
      → σ 자리에 GARCH_Variance, scalar α/β (M3 의 단순 확장)

    [M4-Combined] use_garch_sigma=True, vol_group_map={'AAPL': 'mid_vol', ...}
                   + alpha_down_by_group={'low_vol': 0.3, 'mid_vol': 0.5, 'high_vol': 0.8} 등
      → σ 자리 GARCH_Variance + vol_group 별 α/β 적용
    """

    def __init__(
        self,
        tft: TemporalFusionTransformer,
        adaptive_loss: AdaptivePinballLoss,
        vix_encoder_idx: int,
        sigma_encoder_idx: int,
        learning_rate: float = 0.001,
        vix_mean: float = 0.0,
        vix_std: float = 1.0,
        ticker_to_vol_group_idx: torch.Tensor = None,
    ):
        super().__init__()
        self.tft = tft
        self.adaptive_loss = adaptive_loss
        self.vix_encoder_idx = vix_encoder_idx
        self.sigma_encoder_idx = sigma_encoder_idx
        self.learning_rate = learning_rate
        self.vix_mean = vix_mean
        self.vix_std = vix_std

        # ===== M4: ticker idx → vol_group idx 매핑 buffer =====
        # None 이면 empty tensor 등록 → forward 에서 None 으로 인식하여 M3 호환 동작
        if ticker_to_vol_group_idx is None:
            ticker_to_vol_group_idx = torch.tensor([], dtype=torch.long)
        self.register_buffer(
            "ticker_to_vol_group_idx", ticker_to_vol_group_idx
        )

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
        # ===== M4 추가 인자 =====
        use_garch_sigma: bool = True,
        ticker_to_vol_group_idx: torch.Tensor = None,
        group_labels: list[str] = None,
        alpha_down_by_group: dict = None,
        beta_down_by_group: dict = None,
        alpha_up_by_group: dict = None,
        beta_up_by_group: dict = None,
    ) -> "M4FullModel":
        quantiles = quantiles or [0.1, 0.5, 0.9]
        group_labels = group_labels or ["low_vol", "mid_vol", "high_vol"]

        reals: list[str] = dataset.reals
        vix_idx = reals.index("VIX_Close")

        # ===== M4: σ 자리 선택 =====
        if use_garch_sigma and "GARCH_Variance" in reals:
            sigma_idx = reals.index("GARCH_Variance")
            print(f"[M4FullModel] σ feature: GARCH_Variance (idx={sigma_idx})  [M4 mode]")
        else:
            sigma_idx = reals.index("Realized_Vol_20d")
            print(f"[M4FullModel] σ feature: Realized_Vol_20d (idx={sigma_idx})  [M3 mode]")

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

        # ===== M4: ticker idx → vol_group idx 매핑 등록 =====
        # 외부 (run_full_pipeline) 에서 이미 만든 tensor 를 받음 → categorical_encoders API 의존성 없음
        if ticker_to_vol_group_idx is not None:
            if not isinstance(ticker_to_vol_group_idx, torch.Tensor):
                ticker_to_vol_group_idx = torch.tensor(
                    ticker_to_vol_group_idx, dtype=torch.long
                )
            else:
                ticker_to_vol_group_idx = ticker_to_vol_group_idx.long()
            print(
                f"[M4FullModel] ticker→vol_group 매핑 등록: "
                f"{len(ticker_to_vol_group_idx)} tickers, groups={group_labels}"
            )
        else:
            print(f"[M4FullModel] ticker_to_vol_group_idx=None → M4-GARCH only mode")

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
            group_labels=group_labels,
            alpha_down_by_group=alpha_down_by_group,
            beta_down_by_group=beta_down_by_group,
            alpha_up_by_group=alpha_up_by_group,
            beta_up_by_group=beta_up_by_group,
        )

        return cls(
            tft=tft,
            adaptive_loss=adaptive_loss,
            vix_encoder_idx=vix_idx,
            sigma_encoder_idx=sigma_idx,
            learning_rate=learning_rate,
            vix_mean=vix_mean,
            vix_std=vix_std,
            ticker_to_vol_group_idx=ticker_to_vol_group_idx,
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

    def _extract_vol_group_idx(self, x: dict) -> torch.Tensor:
        """
        Sample 별 vol_group_idx 추출.
        매핑 등록 안 됐으면 (M4-GARCH only) None 반환 → loss 에서 group 0 default.
        """
        if self.ticker_to_vol_group_idx.numel() == 0:
            return None
        # x["groups"]: (batch, n_group_ids) — 우리는 group_ids=[GROUP_ID] 라 (batch, 1)
        ticker_idx = x["groups"][:, 0]  # (batch,)
        return self.ticker_to_vol_group_idx[ticker_idx]

    def training_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
        x, y = batch
        y_true = y[0]
        pred_len = y_true.shape[1]

        out = self.tft(x)
        y_pred = out["prediction"]

        vix, sigma = self._extract_vix_sigma(x, pred_len)
        vol_group_idx = self._extract_vol_group_idx(x)
        loss = self.adaptive_loss(y_pred, y_true, vix, sigma, vol_group_idx)

        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
        x, y = batch
        y_true = y[0]
        pred_len = y_true.shape[1]

        out = self.tft(x)
        y_pred = out["prediction"]

        vix, sigma = self._extract_vix_sigma(x, pred_len)
        vol_group_idx = self._extract_vol_group_idx(x)
        loss = self.adaptive_loss(y_pred, y_true, vix, sigma, vol_group_idx)

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
