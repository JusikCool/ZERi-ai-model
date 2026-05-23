import json
from pathlib import Path

import pandas as pd

from model.m3_full_model.dataset import GROUP_ID, load_data

from .dataset import SECTOR_DATA_PATH
from .sectors import get_tickers

GLOBAL_SCALES_PATH = Path("model/saved/global_scales.json")
SECTOR_SCALES_PATH = Path("model/saved/sector_scales.json")


def compute_global_scales(df: pd.DataFrame) -> dict:
    vix = df["VIX_Close"].dropna()
    sigma = df["Realized_Vol_20d"].dropna()
    return {
        "vix_mean": float(vix.mean()),
        "vix_std": float(vix.std()),
        "sigma_mean": float(sigma.mean()),
        "sigma_std": float(sigma.std()),
        "n_rows": int(len(df)),
        "n_groups": int(df["group_id"].nunique()),
    }


def compute_and_save_global_scales(data_path: Path = SECTOR_DATA_PATH) -> dict:
    df = load_data(data_path)
    scales = compute_global_scales(df)
    GLOBAL_SCALES_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(GLOBAL_SCALES_PATH, "w", encoding="utf-8") as f:
        json.dump(scales, f, indent=2, ensure_ascii=False)
    return scales


def load_global_scales(auto_compute: bool = True) -> dict:
    if not GLOBAL_SCALES_PATH.exists():
        if not auto_compute:
            raise FileNotFoundError(
                f"{GLOBAL_SCALES_PATH} 가 존재하지 않습니다. "
                "compute_and_save_global_scales() 를 먼저 실행하세요."
            )
        return compute_and_save_global_scales()
    with open(GLOBAL_SCALES_PATH, encoding="utf-8") as f:
        return json.load(f)


def compute_sector_sigma_scales(
    sector_ids: list[str], data_path: Path = SECTOR_DATA_PATH
) -> dict:
    df = load_data(data_path)
    out: dict[str, dict] = {}
    for sector_id in sector_ids:
        tickers = get_tickers(sector_id)
        sub = df[df[GROUP_ID].isin(tickers)]
        sigma = sub["Realized_Vol_20d"].dropna()
        out[sector_id] = {
            "sigma_mean": float(sigma.mean()),
            "sigma_std": float(sigma.std()),
            "n_rows": int(len(sub)),
            "tickers": tickers,
        }
    return out


def compute_and_save_sector_scales(
    sector_ids: list[str], data_path: Path = SECTOR_DATA_PATH
) -> dict:
    scales = compute_sector_sigma_scales(sector_ids, data_path=data_path)
    SECTOR_SCALES_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(SECTOR_SCALES_PATH, "w", encoding="utf-8") as f:
        json.dump(scales, f, indent=2, ensure_ascii=False)
    return scales


def load_sector_scales(auto_compute: bool = True) -> dict:
    if not SECTOR_SCALES_PATH.exists():
        if not auto_compute:
            raise FileNotFoundError(
                f"{SECTOR_SCALES_PATH} 가 존재하지 않습니다. "
                "compute_and_save_sector_scales() 를 먼저 실행하세요."
            )
        from .sectors import SECTOR_IDS

        return compute_and_save_sector_scales(SECTOR_IDS)
    with open(SECTOR_SCALES_PATH, encoding="utf-8") as f:
        return json.load(f)


def get_scales_for_sector(sector_id: str) -> dict:
    g = load_global_scales()
    s = load_sector_scales()
    if sector_id not in s:
        from .sectors import SECTOR_IDS

        s = compute_and_save_sector_scales(SECTOR_IDS)
    sec = s[sector_id]
    return {
        "vix_mean": g["vix_mean"],
        "vix_std": g["vix_std"],
        "sigma_mean": sec["sigma_mean"],
        "sigma_std": sec["sigma_std"],
    }


if __name__ == "__main__":
    from .sectors import SECTOR_IDS

    g = compute_and_save_global_scales()
    print(f"전역 스케일 저장: {GLOBAL_SCALES_PATH}")
    for k, v in g.items():
        print(f"  {k}: {v}")
    s = compute_and_save_sector_scales(SECTOR_IDS)
    print(f"\n섹터별 σ 스케일 저장: {SECTOR_SCALES_PATH}")
    for sec, vals in s.items():
        print(f"  [{sec}] sigma_std={vals['sigma_std']:.4f}, sigma_mean={vals['sigma_mean']:.4f}")
