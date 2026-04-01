import numpy as np
import pandas as pd
import torch
from scipy.stats import chi2

from pytorch_forecasting import TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.models import TemporalFusionTransformer
from pytorch_forecasting.metrics import MultiHorizonMetric
from pytorch_forecasting.metrics import QuantileLoss

class AdaptivePinballLoss(QuantileLoss):
    def __init__(
        self,
        quantiles: list = [0.1, 0.5, 0.9],
        alpha_down: float = 1.0,
        alpha_up: float = 2.0,
        vix_threshold: float = 20.0,
        crossing_weight: float = 0.1,
    ):
        super().__init__(quantiles=quantiles)
        self.alpha_down = alpha_down
        self.alpha_up = alpha_up
        self.vix_threshold = vix_threshold
        self.crossing_weight = crossing_weight

    def loss(self, y_pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        losses = []

        # --- per-quantile pinball loss ---
        for i, q in enumerate(self.quantiles):
            errors = target - y_pred[..., i]
            pinball = torch.where(errors >= 0, q * errors, (q - 1) * errors)
            losses.append(pinball)

        losses = torch.stack(losses, dim=-1)  # (batch, time, n_q)

        if len(self.quantiles) >= 2:
            pred_spread = (y_pred[..., -1] - y_pred[..., 0]).detach()  # (batch, time)

            spread_normalized = torch.sigmoid(pred_spread * 50)
            weight = (
                spread_normalized * self.alpha_up
                + (1 - spread_normalized) * self.alpha_down
            ).unsqueeze(-1)  # (batch, time, 1)
        else:
            weight = self.alpha_up

        weighted_loss = (weight * losses).mean()

        # --- quantile crossing penalty ---
        crossing_loss = 0.0
        for i in range(len(self.quantiles) - 1):
            crossing = torch.relu(y_pred[..., i] - y_pred[..., i + 1])
            crossing_loss += crossing.mean()
        crossing_loss *= self.crossing_weight

        return weighted_loss + crossing_loss

def kupiec_pof_test(
    y_true: np.ndarray,
    q_pred: np.ndarray,
    quantile: float = 0.1,
) -> dict:
    n = len(y_true)
    violations = np.sum(y_true < q_pred)
    t0 = n - violations
    t1 = violations
    p = t1 / n

    if t1 == 0 or t1 == n:
        return {"violations": int(t1), "n": n, "violation_rate": p, "lr_stat": np.nan, "p_value": np.nan}

    lr = -2 * (
        t1 * np.log(quantile / p) + t0 * np.log((1 - quantile) / (1 - p))
    )
    p_value = 1 - chi2.cdf(lr, df=1)

    return {
        "violations": int(t1),
        "n": n,
        "violation_rate": round(p, 6),
        "lr_stat": round(lr, 6),
        "p_value": round(p_value, 6),
    }

def run_validation_by_group(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    groups: np.ndarray,
    quantiles: list[float] = None,
    vr_threshold: float = 0.05,
    pvalue_threshold: float = 0.05,
) -> pd.DataFrame:
    quantiles = quantiles or [0.1, 0.5, 0.9]
    rows = []

    for group in np.unique(groups):
        mask = groups == group
        for i, q in enumerate(quantiles):
            result = kupiec_pof_test(y_true[mask], y_pred[mask, i], quantile=q)
            result["group_id"] = group
            result["quantile"] = q
            result["vr_pass"] = abs(result["violation_rate"] - q) <= vr_threshold
            result["kupiec_pass"] = (
                result["p_value"] >= pvalue_threshold
                if not np.isnan(result["p_value"])
                else False
            )
            rows.append(result)

    df = pd.DataFrame(rows)[
        ["group_id", "quantile", "violations", "n", "violation_rate", "lr_stat", "p_value", "vr_pass", "kupiec_pass"]
    ]
    return df

MAX_ENCODER_LENGTH = 60
MAX_PREDICTION_LENGTH = 10
DATA_CSV_PATH = "./tft_processed_panel_v1.csv"
MODEL_PATH = "./model/m2/trial_23/m2_epoch=17_val_loss=0.0001.ckpt"

df = pd.read_csv(DATA_CSV_PATH, parse_dates=["Date"])
df = df.sort_values(["group_id", "time_idx"]).reset_index(drop=True)
df["group_id"] = df["group_id"].astype(str)
df["Month"] = df["Month"].astype(str)
df["Day_of_Week"] = df["Day_of_Week"].astype(str)
df = df.dropna(subset=["Target_Return_5d"]).reset_index(drop=True)
df["time_idx"] = df.groupby("group_id").cumcount()

training_cutoff = int(df["time_idx"].max() * 0.8)

train_dataset = TimeSeriesDataSet(
    df[df["time_idx"] <= training_cutoff],
    time_idx="time_idx",
    target="Target_Return_5d",
    group_ids=["group_id"],
    max_encoder_length=MAX_ENCODER_LENGTH,
    max_prediction_length=MAX_PREDICTION_LENGTH,
    static_categoricals=["group_id"],
    time_varying_known_categoricals=["Month", "Day_of_Week"],
    time_varying_unknown_reals=[
        "Open", "High", "Low", "Close", "Volume",
        "NASDAQ_Close", "VIX_Close", "FEDFUNDS", "UNRATE", "DTWEXBGS",
        "CPIAUCSL", "PCEPI", "GDP", "M2SL", "GS10", "T10Y2Y",
        "PAYEMS", "CSUSHPISA", "INDPRO",
        "RSI_14", "ATR_14", "SMA_20", "Returns", "Realized_Vol_20d",
    ],
    target_normalizer=GroupNormalizer(groups=["group_id"], transformation="softplus"),
    add_relative_time_idx=True,
    add_target_scales=True,
    add_encoder_length=True,
    allow_missing_timesteps=True,
)

val_dataset = TimeSeriesDataSet.from_dataset(
    train_dataset,
    df[df["time_idx"] > training_cutoff - 60],
    predict=True,
    stop_randomization=True,
)

loss = AdaptivePinballLoss(
    quantiles=[0.1, 0.5, 0.9],
    vix_threshold=20.0,
    alpha_down=2.435607553765352,
    alpha_up=0.5318672138943901,
    crossing_weight=0.2194809592983885,
)

model = TemporalFusionTransformer.from_dataset(
    dataset=train_dataset,
    learning_rate=0.00010548090781374072,
    hidden_size=32,
    attention_head_size=4,
    dropout=0.15555217120326112,
    hidden_continuous_size=8,
    loss=loss,
    output_size=3
)
ckpt = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
model.load_state_dict(ckpt["state_dict"])
model.eval()

group_mapping = {i: g for i, g in enumerate(sorted(df["group_id"].unique()))}

max_time = df["time_idx"].max()
min_time = df["time_idx"].min()
fold_size = (max_time - min_time) // (5 + 1)

all_y_true = []
all_y_pred = []
all_groups = []

for i in range(5):
    train_end = min_time + fold_size * (i + 1)
    val_end = train_end + fold_size

    fold_df = df[
        (df["time_idx"] > train_end - 60)
        & (df["time_idx"] <= val_end)
    ].copy()

    val_ds = TimeSeriesDataSet.from_dataset(
        train_dataset,
        fold_df,
        predict=False,
        stop_randomization=True,
    )
    val_loader = val_ds.to_dataloader(
        train=False,
        batch_size=64 * 2,
        num_workers=0,
    )

    with torch.no_grad():
        for batch in val_loader:
            x, y = batch
            y_true = y[0].numpy()[:, 0]        # horizon 1
            out = model(x)
            y_pred = out["prediction"].detach().numpy()[:, 0, :]  # horizon 1

            group_ints = x["groups"][:, 0].numpy()
            group_names = np.array([group_mapping[g] for g in group_ints])

            all_y_true.append(y_true)
            all_y_pred.append(y_pred)
            all_groups.append(group_names)

y_true_all = np.concatenate(all_y_true)
y_pred_all = np.concatenate(all_y_pred, axis=0)
groups_all = np.concatenate(all_groups)

report = run_validation_by_group(
    y_true_all,
    y_pred_all,
    groups_all,
    quantiles=[0.1, 0.5, 0.9],
    vr_threshold=0.05,
    pvalue_threshold=0.05,
)

print(report)