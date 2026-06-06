import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from pytorch_forecasting import TimeSeriesDataSet

from model.m4.dataset import (
    build_dataset,
    build_dataset_sector_aware,
    get_vix_stats,
    GROUP_ID,
    TIME_IDX,
)
from model.m4.model import M4FullModel
from model.sector_models.sectors import SECTOR_IDS, SECTOR_MAP
from validation.kupiec.kupiec import run_validation_by_group
from validation.kupiec.coverage_tests import (
    run_validation_by_group_extended,
    print_extended_summary,
)

CHECKPOINT_DIR = Path("model/saved")
COVID_START = "2020-01-01"
COVID_END = "2021-01-01"
UKRAINE_INFLATION_START = "2022-02-01"
UKRAINE_INFLATION_END = "2022-07-01"
TRUMP_TARIFF_START = "2025-04-01"
TRUMP_TARIFF_END = "2025-05-31"

VOL_GROUP_MAP_PATH = Path("data/raw/vol_group_map.json")
GROUP_LABELS = ["low_vol", "mid_vol", "high_vol"]
SECTOR_LABELS = list(SECTOR_IDS)


def _build_dataset_for_mode(mode, df, max_encoder_length, max_prediction_length):
    if mode == "m4_sector":
        return build_dataset_sector_aware(
            df,
            max_encoder_length=max_encoder_length,
            max_prediction_length=max_prediction_length,
        )
    return build_dataset(
        df,
        max_encoder_length=max_encoder_length,
        max_prediction_length=max_prediction_length,
    )


def _build_ticker_to_sector_idx(panel_df):
    panel_tickers_sorted = sorted(panel_df["group_id"].unique())
    label_to_idx = {l: i for i, l in enumerate(SECTOR_LABELS)}
    sec_list = []
    for ticker in panel_tickers_sorted:
        sec = SECTOR_MAP.get(ticker)
        if sec is None or sec not in label_to_idx:
            raise ValueError(
                f"m4_sector 백테스트: {ticker} 가 SECTOR_MAP 에 없음. "
                f"load_data_sector_aware(drop_unmapped=True) 로 df 를 만들어 전달하세요."
            )
        sec_list.append(label_to_idx[sec])
    return torch.tensor(sec_list, dtype=torch.long)


def _best_params_path(mode: str = None) -> Path:
    if mode is None:
        return CHECKPOINT_DIR / "best_params.json"
    return CHECKPOINT_DIR / f"best_params_{mode}.json"


def _load_best_params(mode: str = None) -> dict:
    path = _best_params_path(mode)
    if path.exists():
        with open(path, encoding="utf-8") as f:
            best = json.load(f)
        print(f"[backtest] best_params 로드: {path}")
        return best
    print(f"[backtest] ⚠️ {path} 없음 → config default 사용")
    return {}


def _load_vol_group_map() -> dict:
    if VOL_GROUP_MAP_PATH.exists():
        with open(VOL_GROUP_MAP_PATH, encoding="utf-8") as f:
            return json.load(f)
    return None


def _build_ticker_to_vol_group_idx(
    panel_df: pd.DataFrame, vol_group_map: dict
) -> torch.Tensor:
    if vol_group_map is None:
        return None
    panel_tickers_sorted = sorted(panel_df["group_id"].unique())
    label_to_idx = {l: i for i, l in enumerate(GROUP_LABELS)}
    default_label = GROUP_LABELS[len(GROUP_LABELS) // 2]

    vg_list = []
    n_missing = 0
    for ticker in panel_tickers_sorted:
        if ticker not in vol_group_map:
            n_missing += 1
            vg_str = default_label
        else:
            vg_str = vol_group_map[ticker]
            if vg_str not in label_to_idx:
                vg_str = default_label
        vg_list.append(label_to_idx[vg_str])
    if n_missing > 0:
        print(
            f"[backtest] ⚠️ vol_group_map 에 없는 ticker {n_missing}개 "
            f"→ '{default_label}' fallback"
        )
    return torch.tensor(vg_list, dtype=torch.long)


def _find_checkpoint(mode: str = None) -> str:
    if mode is not None:
        pattern = str(CHECKPOINT_DIR / f"{mode}_best_*.ckpt")
        ckpts = glob.glob(pattern)
        if ckpts:
            return sorted(ckpts)[-1]
        print(f"[backtest] ⚠️ '{pattern}' 매치 없음 → 전체 검색 fallback")
    ckpts = glob.glob(str(CHECKPOINT_DIR / "*.ckpt"))
    if not ckpts:
        raise FileNotFoundError(f"{CHECKPOINT_DIR} 에 체크포인트 없음")
    return sorted(ckpts)[-1]


def _infer_arch_from_ckpt(ckpt_path: str) -> dict:
    """
    체크포인트의 state_dict 텐서 shape 에서 architecture 파라미터를 직접 추정.

    hyper_parameters 보다 더 신뢰 — tensor shape 는 절대 거짓말 안 함.

    추정:
        hidden_size            : LSTM weight_ih shape = (4*hidden_size, input_size)
        attention_head_size    : multihead_attn.q_layers.{i}.weight 개수
        hidden_continuous_size : prescalers.<var>.weight shape = (hcs, 1)

    그 외 (dropout, learning_rate, α/β) 는 hyper_parameters 에서 보충.
    """
    try:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    except TypeError:
        ckpt = torch.load(ckpt_path, map_location="cpu")

    state_dict = ckpt.get("state_dict", {})
    arch = {}

    # 1) hidden_size from LSTM weight_ih
    for k in ("tft.lstm_encoder.weight_ih_l0", "tft.lstm_decoder.weight_ih_l0"):
        if k in state_dict:
            arch["hidden_size"] = state_dict[k].shape[0] // 4
            break

    # 2) attention_head_size from q_layers count
    n_heads = 0
    while f"tft.multihead_attn.q_layers.{n_heads}.weight" in state_dict:
        n_heads += 1
    if n_heads > 0:
        arch["attention_head_size"] = n_heads

    # 3) hidden_continuous_size from prescalers
    for k in sorted(state_dict.keys()):
        if k.startswith("tft.prescalers.") and k.endswith(".weight"):
            arch["hidden_continuous_size"] = state_dict[k].shape[0]
            break

    # 4) hyper_parameters 보충
    hparams = ckpt.get("hyper_parameters", {}) or {}
    for k in ("dropout", "learning_rate", "crossing_weight"):
        if k in hparams:
            arch[k] = hparams[k]
    for k, v in hparams.items():
        if k.startswith("alpha_") or k.startswith("beta_"):
            arch[k] = v

    return arch


def _build_model_kwargs(
    model_cfg: dict,
    vix_mean: float,
    vix_std: float,
    mode: str = None,
    df: pd.DataFrame = None,
    vol_group_map: dict = None,
    ckpt_path: str = None,
) -> dict:
    """
    M4FullModel.from_dataset 의 kwargs 생성.
    우선순위: 체크포인트 state_dict 추정 > best_params > config["model"]
    """
    best = _load_best_params(mode)
    ckpt_arch = _infer_arch_from_ckpt(ckpt_path) if ckpt_path else {}
    if ckpt_arch:
        arch_keys = {k: v for k, v in ckpt_arch.items()
                     if k in ("hidden_size", "attention_head_size",
                              "hidden_continuous_size", "dropout")}
        print(f"[backtest] ckpt arch (inferred from state_dict): {arch_keys}")

    def pick(key, default=None):
        if key in ckpt_arch and ckpt_arch[key] is not None:
            return ckpt_arch[key]
        if key in best:
            return best[key]
        return model_cfg.get(key, default)

    kwargs = dict(
        learning_rate=pick("learning_rate", model_cfg.get("learning_rate", 1e-3)),
        hidden_size=pick("hidden_size", model_cfg["hidden_size"]),
        attention_head_size=pick(
            "attention_head_size", model_cfg["attention_head_size"]
        ),
        dropout=pick("dropout", model_cfg["dropout"]),
        quantiles=model_cfg["quantiles"],
        vix_threshold=model_cfg["vix_threshold"],
        vix_mean=vix_mean,
        vix_std=vix_std,
        use_garch_sigma=True,
        group_labels=GROUP_LABELS,
    )

    hcs = pick("hidden_continuous_size", None)
    if hcs is not None:
        kwargs["hidden_continuous_size"] = hcs

    cw = pick("crossing_weight", None)
    if cw is not None:
        kwargs["crossing_weight"] = cw

    if mode == "m4_combined":
        kwargs["alpha_down_by_group"] = {g: pick(f"alpha_down_{g}", 1.0) for g in GROUP_LABELS}
        kwargs["beta_down_by_group"] = {g: pick(f"beta_down_{g}", 1.0) for g in GROUP_LABELS}
        kwargs["alpha_up_by_group"] = {g: pick(f"alpha_up_{g}", 1.0) for g in GROUP_LABELS}
        kwargs["beta_up_by_group"] = {g: pick(f"beta_up_{g}", 1.0) for g in GROUP_LABELS}
        kwargs["ticker_to_vol_group_idx"] = _build_ticker_to_vol_group_idx(
            df, vol_group_map
        )
    elif mode == "m4_sector":
        kwargs["alpha_down_by_group"] = {s: pick(f"alpha_down_{s}", 1.0) for s in SECTOR_LABELS}
        kwargs["beta_down_by_group"] = {s: pick(f"beta_down_{s}", 1.0) for s in SECTOR_LABELS}
        kwargs["alpha_up_by_group"] = {s: pick(f"alpha_up_{s}", 1.0) for s in SECTOR_LABELS}
        kwargs["beta_up_by_group"] = {s: pick(f"beta_up_{s}", 1.0) for s in SECTOR_LABELS}
        kwargs["ticker_to_vol_group_idx"] = _build_ticker_to_sector_idx(df)
        kwargs["group_labels"] = SECTOR_LABELS
    else:
        for k in ("alpha_down", "beta_down", "alpha_up", "beta_up"):
            v = pick(k, None)
            if v is not None:
                kwargs[k] = v
        kwargs["ticker_to_vol_group_idx"] = None

    return kwargs


def _build_validation_reports(
    y_true_all, y_pred_all, groups_all, model_cfg, val_cfg, title=""
):
    basic_report = run_validation_by_group(
        y_true_all, y_pred_all, groups_all,
        quantiles=model_cfg["quantiles"],
        vr_threshold=val_cfg["violation_rate_threshold"],
        pvalue_threshold=val_cfg["kupiec_pvalue_threshold"],
    )
    extended_report = run_validation_by_group_extended(
        y_true_all, y_pred_all, groups_all,
        quantiles=model_cfg["quantiles"],
        vr_threshold=val_cfg["violation_rate_threshold"],
        pvalue_threshold=val_cfg["kupiec_pvalue_threshold"],
        dq_n_lags=val_cfg.get("dq_n_lags", 4),
    )
    if title:
        print_extended_summary(extended_report, title=title)
    return {"basic": basic_report, "extended": extended_report}


def rolling_window_backtest(
    df, config, n_splits=5, return_extended=False, mode=None,
):
    data_cfg = config["data"]
    model_cfg = config["model"]

    print(f"\n[backtest] mode={mode}")
    ckpt_path = _find_checkpoint(mode)
    print(f"[backtest] checkpoint: {ckpt_path}")

    vol_group_map = _load_vol_group_map() if mode == "m4_combined" else None
    if mode == "m4_combined":
        if vol_group_map is None:
            raise ValueError(f"mode='m4_combined' 인데 {VOL_GROUP_MAP_PATH} 없음")
        print(f"[backtest] vol_group_map 로드: {len(vol_group_map)} tickers")

    vix_mean, vix_std = get_vix_stats(df)
    train_ds, _ = _build_dataset_for_mode(
        mode, df,
        max_encoder_length=data_cfg["window_size"],
        max_prediction_length=data_cfg["horizon"],
    )

    model_kwargs = _build_model_kwargs(
        model_cfg, vix_mean, vix_std,
        mode=mode, df=df, vol_group_map=vol_group_map,
        ckpt_path=ckpt_path,
    )
    model = M4FullModel.from_dataset(dataset=train_ds, **model_kwargs)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    group_mapping = {i: g for i, g in enumerate(sorted(df[GROUP_ID].unique()))}
    max_time = df[TIME_IDX].max()
    min_time = df[TIME_IDX].min()
    fold_size = (max_time - min_time) // (n_splits + 1)

    all_y_true, all_y_pred, all_groups = [], [], []

    for i in range(n_splits):
        train_end = min_time + fold_size * (i + 1)
        val_end = train_end + fold_size
        fold_df = df[
            (df[TIME_IDX] > train_end - data_cfg["window_size"])
            & (df[TIME_IDX] <= val_end)
        ].copy()

        val_ds = TimeSeriesDataSet.from_dataset(
            train_ds, fold_df, predict=False, stop_randomization=True,
        )
        val_loader = val_ds.to_dataloader(
            train=False, batch_size=model_cfg["batch_size"] * 2, num_workers=0,
        )

        with torch.no_grad():
            for batch in val_loader:
                x, y = batch
                y_true = y[0].numpy()
                y_pred = model.predict(x).numpy()
                group_ints = x["groups"][:, 0].numpy()
                group_names = np.array([group_mapping[g] for g in group_ints])
                all_y_true.append(y_true[:, -1])
                all_y_pred.append(y_pred[:, -1, :])
                all_groups.append(group_names)

    y_true_all = np.concatenate(all_y_true)
    y_pred_all = np.concatenate(all_y_pred, axis=0)
    groups_all = np.concatenate(all_groups)

    val_cfg = config["validation"]
    reports = _build_validation_reports(
        y_true_all, y_pred_all, groups_all,
        model_cfg, val_cfg,
        title=f"Rolling-window Backtest (mode={mode}, {n_splits} splits)",
    )
    return reports if return_extended else reports["basic"]


def period_backtest(
    df, config, start, end, return_extended=False, mode=None,
):
    data_cfg = config["data"]
    model_cfg = config["model"]

    print(f"\n[backtest] mode={mode}, period={start}~{end}")
    ckpt_path = _find_checkpoint(mode)
    print(f"[backtest] checkpoint: {ckpt_path}")

    vol_group_map = _load_vol_group_map() if mode == "m4_combined" else None
    if mode == "m4_combined":
        if vol_group_map is None:
            raise ValueError(f"mode='m4_combined' 인데 {VOL_GROUP_MAP_PATH} 없음")
        print(f"[backtest] vol_group_map 로드: {len(vol_group_map)} tickers")

    vix_mean, vix_std = get_vix_stats(df)
    train_ds, _ = _build_dataset_for_mode(
        mode, df,
        max_encoder_length=data_cfg["window_size"],
        max_prediction_length=data_cfg["horizon"],
    )

    model_kwargs = _build_model_kwargs(
        model_cfg, vix_mean, vix_std,
        mode=mode, df=df, vol_group_map=vol_group_map,
        ckpt_path=ckpt_path,
    )
    model = M4FullModel.from_dataset(dataset=train_ds, **model_kwargs)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    group_mapping = {i: g for i, g in enumerate(sorted(df[GROUP_ID].unique()))}
    period_time_idx = df[
        (df["Date"] >= start) & (df["Date"] < end)
    ][TIME_IDX].unique()
    if len(period_time_idx) == 0:
        raise ValueError(f"구간({start}~{end}) 데이터 없음")

    t_min = int(period_time_idx.min())
    t_max = int(period_time_idx.max())
    fold_df = df[
        (df[TIME_IDX] >= t_min - data_cfg["window_size"])
        & (df[TIME_IDX] <= t_max)
    ].copy()

    val_ds = TimeSeriesDataSet.from_dataset(
        train_ds, fold_df, predict=False, stop_randomization=True,
    )
    val_loader = val_ds.to_dataloader(
        train=False, batch_size=model_cfg["batch_size"] * 2, num_workers=0,
    )

    all_y_true, all_y_pred, all_groups = [], [], []
    with torch.no_grad():
        for batch in val_loader:
            x, y = batch
            y_true = y[0].numpy()
            y_pred = model.predict(x).numpy()
            group_ints = x["groups"][:, 0].numpy()
            group_names = np.array([group_mapping[g] for g in group_ints])
            all_y_true.append(y_true[:, -1])
            all_y_pred.append(y_pred[:, -1, :])
            all_groups.append(group_names)

    y_true_all = np.concatenate(all_y_true)
    y_pred_all = np.concatenate(all_y_pred, axis=0)
    groups_all = np.concatenate(all_groups)

    val_cfg = config["validation"]
    reports = _build_validation_reports(
        y_true_all, y_pred_all, groups_all,
        model_cfg, val_cfg,
        title=f"Period Backtest (mode={mode}, {start} ~ {end})",
    )
    reports["basic"]["period"] = f"{start} ~ {end}"
    reports["extended"]["period"] = f"{start} ~ {end}"
    return reports if return_extended else reports["basic"]


def covid_backtest(df, config, start=COVID_START, end=COVID_END, return_extended=False, mode=None):
    return period_backtest(df, config, start, end, return_extended=return_extended, mode=mode)


def ukraine_inflation_backtest(df, config, start=UKRAINE_INFLATION_START, end=UKRAINE_INFLATION_END, return_extended=False, mode=None):
    return period_backtest(df, config, start, end, return_extended=return_extended, mode=mode)


def trump_tariff_backtest(df, config, start=TRUMP_TARIFF_START, end=TRUMP_TARIFF_END, return_extended=False, mode=None):
    return period_backtest(df, config, start, end, return_extended=return_extended, mode=mode)