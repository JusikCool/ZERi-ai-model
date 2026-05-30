from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List


BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent.parent


@dataclass
class TFTFixedConfig:
    data_path: Path = PROJECT_ROOT / "data" / "raw" / "tft_processed_panel_v3.csv"
    output_dir: Path = BASE_DIR / "runs" / "tft_fixed"
    target_column: str = "Target_Return_5d"
    excluded_model_columns: List[str] = field(
        default_factory=lambda: ["VIX_Close", "Realized_Vol_20d"]
    )
    group_ids: List[str] = field(default_factory=lambda: ["group_id"])
    static_categoricals: List[str] = field(default_factory=lambda: ["group_id"])
    time_varying_known_categoricals: List[str] = field(
        default_factory=lambda: ["Month", "Day_of_Week"]
    )
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
        ]
    )
    max_encoder_length: int = 60
    max_prediction_length: int = 10
    val_ratio: float = 0.2
    quantiles: List[float] = field(default_factory=lambda: [0.10, 0.50, 0.90])
    batch_size: int = 64
    num_workers: int = 0
    learning_rate: float = 3e-4
    hidden_size: int = 64
    attention_head_size: int = 4
    hidden_continuous_size: int = 16
    dropout: float = 0.1
    max_epochs: int = 50
    accelerator: str = "gpu"
    devices: int = 1
    gradient_clip_val: float = 0.1
    limit_train_batches: float = 1.0
    limit_val_batches: float = 1.0
    log_every_n_steps: int = 1
    smoke_test_epochs: int = 2
    seed: int = 42
    default_plot_ticker: str = "AAPL"
