from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List


@dataclass
class TFTVIXConfig:
    data_path: Path = Path(r"C:\Users\user\Desktop\JERi\ZERi-ai-model\M2\artifacts\tft_vix_panel_ready.csv")
    output_dir: Path = Path(r"C:\Users\user\Desktop\JERi\ZERi-ai-model\M2\runs\tft_vix")
    target_column: str = "Target_Return_5d"
    group_ids: List[str] = field(default_factory=lambda: ["group_id"])
    static_categoricals: List[str] = field(default_factory=lambda: ["group_id"])
    time_varying_known_categoricals: List[str] = field(default_factory=lambda: ["Month", "Day_of_Week"])
    time_varying_known_reals: List[str] = field(default_factory=lambda: ["time_idx"])
    time_varying_unknown_reals: List[str] = field(
        default_factory=lambda: [
            "Open",
            "High",
            "Low",
            "Close",
            "Volume",
            "Dividends",
            "Stock Splits",
            "NASDAQ_Close",
            "VIX_Close",
            "FEDFUNDS",
            "UNRATE",
            "DTWEXBGS",
            "CPIAUCSL",
            "PCEPI",
            "GDP",
            "M2SL",
            "GS10",
            "T10Y2Y",
            "PAYEMS",
            "CSUSHPISA",
            "INDPRO",
            "RSI_14",
            "ATR_14",
            "SMA_20",
            "Returns",
            "Realized_Vol_20d",
        ]
    )
    max_encoder_length: int = 60
    max_prediction_length: int = 1
    quantiles: List[float] = field(default_factory=lambda: [0.10, 0.25, 0.50])
    loss_name: str = "quantile"
    batch_size: int = 64
    num_workers: int = 0
    learning_rate: float = 1e-3
    hidden_size: int = 32
    attention_head_size: int = 4
    hidden_continuous_size: int = 16
    dropout: float = 0.1
    max_epochs: int = 10
    accelerator: str = "cpu"
    devices: int = 1
    gradient_clip_val: float = 0.1
    limit_train_batches: float = 1.0
    limit_val_batches: float = 1.0
    log_every_n_steps: int = 1
    smoke_test_epochs: int = 2
    seed: int = 42
