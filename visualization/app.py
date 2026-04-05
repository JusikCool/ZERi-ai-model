import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import glob

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import torch
import yaml
from plotly.subplots import make_subplots
from pytorch_forecasting import TimeSeriesDataSet

from model.m3_full_model.dataset import GROUP_ID, TIME_IDX, build_dataset, load_data
from model.m3_full_model.model import M3FullModel

CONFIG_PATH = Path("configs/config.yaml")
CHECKPOINT_DIR = Path("model/saved")


@st.cache_resource(show_spinner="모델 및 데이터 로딩 중...")
def load_everything():
    with open(CONFIG_PATH, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    df = load_data()
    data_cfg = config["data"]
    model_cfg = config["model"]

    train_ds, _ = build_dataset(
        df,
        max_encoder_length=data_cfg["window_size"],
        max_prediction_length=data_cfg["horizon"],
    )

    ckpts = sorted(glob.glob(str(CHECKPOINT_DIR / "*.ckpt")))
    ckpt_path = ckpts[-1]

    model = M3FullModel.from_dataset(
        dataset=train_ds,
        learning_rate=model_cfg["learning_rate"],
        hidden_size=model_cfg["hidden_size"],
        attention_head_size=model_cfg["attention_head_size"],
        dropout=model_cfg["dropout"],
        quantiles=model_cfg["quantiles"],
        vix_threshold=model_cfg["vix_threshold"],
    )
    state = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state["state_dict"])
    model.eval()

    return config, df, train_ds, model


@st.cache_data
def run_val_predictions(_model, _train_ds, df, config):
    data_cfg = config["data"]
    cutoff = int(df[TIME_IDX].max() * 0.8)
    val_df = df[df[TIME_IDX] > cutoff - data_cfg["window_size"]].copy()

    val_ds = TimeSeriesDataSet.from_dataset(
        _train_ds, val_df, predict=False, stop_randomization=True
    )
    loader = val_ds.to_dataloader(train=False, batch_size=128, num_workers=0)

    group_list = sorted(df[GROUP_ID].unique())
    g2name = {i: g for i, g in enumerate(group_list)}

    raw: dict = {t: {} for t in group_list}
    with torch.no_grad():
        for batch in loader:
            x, y = batch
            y_pred = _model.predict(x).numpy()
            gints = x["groups"][:, 0].numpy()
            dtidx = x["decoder_time_idx"].numpy()
            for b in range(len(gints)):
                tick = g2name[int(gints[b])]
                for h in range(dtidx.shape[1]):
                    t = int(dtidx[b, h])
                    raw[tick][t] = y_pred[b, h].tolist()

    results = {}
    for tick in group_list:
        t_df = (
            df[df[GROUP_ID] == tick][[TIME_IDX, "Date", "Close", "Target_Return_5d"]]
            .drop_duplicates(TIME_IDX)
            .set_index(TIME_IDX)
        )
        preds = raw[tick]
        times = sorted([t for t in preds if t in t_df.index])
        if not times:
            continue

        horizon = config["data"]["horizon"]
        future_times = [t + horizon for t in times]
        valid = [(t, ft) for t, ft in zip(times, future_times) if ft in t_df.index]
        if not valid:
            continue
        times, future_times = zip(*valid)

        future_dates = [pd.Timestamp(t_df.loc[ft, "Date"]).strftime("%Y-%m-%d") for ft in future_times]
        closes = np.array([float(t_df.loc[t, "Close"]) for t in times])

        q10 = np.array([preds[t][0] for t in times])
        q50 = np.array([preds[t][1] for t in times])
        q90 = np.array([preds[t][2] for t in times])

        results[tick] = dict(
            dates=future_dates,
            q10_price=closes * (1 + q10),
            q50_price=closes * (1 + q50),
            q90_price=closes * (1 + q90),
        )
    return results


def run_future_prediction(model, train_ds, df, config, ticker):
    from pandas.tseries.offsets import BDay

    data_cfg = config["data"]
    t_df = df[df[GROUP_ID] == ticker].sort_values(TIME_IDX)
    last = t_df.iloc[-1].copy()
    last_date = pd.Timestamp(last["Date"])
    last_tidx = int(last[TIME_IDX])
    last_close = float(last["Close"])
    horizon = data_cfg["horizon"]

    future_dates = pd.date_range(last_date + BDay(1), periods=horizon, freq="B")
    future_rows = []
    for i, d in enumerate(future_dates):
        r = last.copy()
        r["Date"] = d
        r[TIME_IDX] = last_tidx + i + 1
        r["Month"] = str(d.month)
        r["Day_of_Week"] = str(d.dayofweek)
        r["Target_Return_5d"] = 0.0
        future_rows.append(r)

    ticker_full = pd.concat(
        [t_df.tail(data_cfg["window_size"]).copy(), pd.DataFrame(future_rows)],
        ignore_index=True,
    )
    full_df = pd.concat(
        [df[df[GROUP_ID] != ticker].copy(), ticker_full], ignore_index=True
    )

    try:
        pred_ds = TimeSeriesDataSet.from_dataset(
            train_ds, full_df, predict=True, stop_randomization=True
        )
        loader = pred_ds.to_dataloader(train=False, batch_size=128, num_workers=0)

        group_list = sorted(df[GROUP_ID].unique())
        g2name = {i: g for i, g in enumerate(group_list)}

        best = None
        with torch.no_grad():
            for batch in loader:
                x, _ = batch
                y_pred = model.predict(x).numpy()
                gints = x["groups"][:, 0].numpy()
                for b in range(len(gints)):
                    if g2name[int(gints[b])] == ticker:
                        best = y_pred[b]

        if best is not None:
            q10 = best[:, 0]
            q50 = best[:, 1]
            q90 = best[:, 2]
            return dict(
                dates=[d.strftime("%Y-%m-%d") for d in future_dates],
                last_close=last_close,
                last_date=last_date,
                q10_price=last_close * (1 + q10),
                q50_price=last_close * (1 + q50),
                q90_price=last_close * (1 + q90),
            )
    except Exception as e:
        st.warning(f"{ticker} 미래 예측 실패: {e}")
    return None


def _make_band_fig(ticker, val_preds, future_pred, history_days, df):
    vp = val_preds.get(ticker)
    if vp is None:
        return go.Figure()

    val_dates = np.array(vp["dates"])
    q10 = vp["q10_price"]
    q50 = vp["q50_price"]
    q90 = vp["q90_price"]

    t_df = df[df[GROUP_ID] == ticker].sort_values(TIME_IDX)
    all_dates = pd.to_datetime(t_df["Date"].values).strftime("%Y-%m-%d")
    all_close = t_df["Close"].values.astype(float)

    if history_days > 0:
        cutoff_dt = val_dates.astype("datetime64[D]").max() - np.timedelta64(history_days, "D")
        val_mask = val_dates.astype("datetime64[D]") >= cutoff_dt
        val_dates, q10, q50, q90 = (
            val_dates[val_mask], q10[val_mask], q50[val_mask], q90[val_mask],
        )

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=all_dates, y=all_close,
        line=dict(color="#FFD700", width=1.5),
        name="실제 종가",
    ))

    fig.add_trace(go.Scatter(
        x=np.concatenate([val_dates, val_dates[::-1]]),
        y=np.concatenate([q90, q10[::-1]]),
        fill="toself", fillcolor="rgba(255,100,100,0.12)",
        line=dict(color="rgba(0,0,0,0)"),
        name="Q0.1 ~ Q0.9 구간",
    ))
    fig.add_trace(go.Scatter(
        x=val_dates, y=q10,
        line=dict(color="rgba(255,80,80,0.55)", width=1, dash="dot"),
        name="Q0.1 (하방 위험)",
    ))
    fig.add_trace(go.Scatter(
        x=val_dates, y=q90,
        line=dict(color="rgba(80,150,255,0.55)", width=1, dash="dot"),
        name="Q0.9 (상방 한계)",
    ))
    fig.add_trace(go.Scatter(
        x=val_dates, y=q50,
        line=dict(color="rgba(255,165,50,0.9)", width=1.5, dash="dash"),
        name="Q0.5 (중앙값 예측)",
    ))

    if future_pred:
        fd = future_pred["dates"]
        anchor_d = [pd.Timestamp(future_pred["last_date"]).strftime("%Y-%m-%d")]
        lc = future_pred["last_close"]

        fd_ext = anchor_d + list(fd)
        fq10 = np.concatenate([[lc], future_pred["q10_price"]])
        fq50 = np.concatenate([[lc], future_pred["q50_price"]])
        fq90 = np.concatenate([[lc], future_pred["q90_price"]])

        fig.add_trace(go.Scatter(
            x=list(fd_ext) + list(fd_ext[::-1]),
            y=np.concatenate([fq90, fq10[::-1]]),
            fill="toself", fillcolor="rgba(80,220,120,0.15)",
            line=dict(color="rgba(0,0,0,0)"),
            name="미래 예측 구간 (10 거래일)",
        ))
        fig.add_trace(go.Scatter(
            x=fd_ext, y=fq50,
            line=dict(color="rgba(80,220,120,0.95)", width=2, dash="dash"),
            name="미래 Q0.5",
        ))
        fig.add_trace(go.Scatter(
            x=fd_ext, y=fq10,
            line=dict(color="rgba(80,220,120,0.5)", width=1, dash="dot"),
            name="미래 Q0.1",
        ))
        fig.add_trace(go.Scatter(
            x=fd_ext, y=fq90,
            line=dict(color="rgba(80,220,120,0.5)", width=1, dash="dot"),
            name="미래 Q0.9",
        ))
        vline_x = str(future_pred["last_date"].date())
        fig.add_shape(
            type="line",
            x0=vline_x, x1=vline_x,
            y0=0, y1=1, yref="paper",
            line=dict(dash="dash", color="rgba(200,200,200,0.35)"),
        )
        fig.add_annotation(
            x=vline_x, y=1, yref="paper",
            text="▶ 예측 시작",
            showarrow=False,
            font=dict(color="rgba(200,200,200,0.6)", size=11),
            xanchor="left",
        )

    fig.update_layout(
        title=f"{ticker} — 예측 구간 (Quantile Prediction Band)",
        xaxis_title="날짜", yaxis_title="주가 (USD)",
        template="plotly_dark", height=480,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified",
    )
    return fig


def _make_lambda_fig(ticker, df, model):
    t_df = df[df[GROUP_ID] == ticker].sort_values(TIME_IDX).copy()
    dates = pd.to_datetime(t_df["Date"].values).strftime("%Y-%m-%d")
    vix_raw = t_df["VIX_Close"].values.astype(float)
    sigma_raw = t_df["Realized_Vol_20d"].values.astype(float)

    sigma_z = (sigma_raw - sigma_raw.mean()) / (sigma_raw.std() + 1e-8)

    al = model.adaptive_loss
    vix_ex = np.maximum(vix_raw - al.vix_threshold, 0.0) / al.vix_scale
    sig_t = np.maximum(sigma_z, 0.0) / al.sigma_scale

    lam_down = 1.0 + al.alpha_down * vix_ex + al.beta_down * sig_t
    lam_up = 1.0 + al.alpha_up * vix_ex + al.beta_up * sig_t

    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=(
            "VIX & 실현변동성(σ) 원시값",
            "λt 동적 가중치 — Q0.1(하방) / Q0.9(상방)",
        ),
        shared_xaxes=True,
        vertical_spacing=0.12,
    )

    fig.add_trace(go.Scatter(
        x=dates, y=vix_raw, name="VIX",
        line=dict(color="rgba(255,160,50,0.85)", width=1.2),
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=dates, y=sigma_raw * 100, name="σ × 100 (%)",
        line=dict(color="rgba(100,180,255,0.85)", width=1.2),
    ), row=1, col=1)
    fig.add_hline(
        y=25, line_dash="dot", line_color="rgba(255,100,100,0.45)",
        annotation_text="VIX = 25", annotation_position="top right",
        row=1, col=1,
    )

    fig.add_trace(go.Scatter(
        x=dates, y=lam_down, name="λt Q0.1 (하방)",
        line=dict(color="rgba(255,80,80,0.85)", width=1.5),
    ), row=2, col=1)
    fig.add_trace(go.Scatter(
        x=dates, y=lam_up, name="λt Q0.9 (상방)",
        line=dict(color="rgba(80,150,255,0.85)", width=1.5, dash="dash"),
    ), row=2, col=1)
    fig.add_hline(
        y=1.0, line_dash="dash", line_color="rgba(200,200,200,0.25)",
        row=2, col=1,
    )

    fig.update_yaxes(title_text="VIX / σ×100", row=1, col=1)
    fig.update_yaxes(title_text="λt", row=2, col=1)
    fig.update_layout(
        title=f"{ticker} — 적응형 가중치 λt 동향",
        template="plotly_dark", height=540,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified",
    )
    return fig


# ── UI ────────────────────────────────────────────────────────

st.set_page_config(
    page_title="하방 리스크 예측 대시보드",
    layout="wide",
    page_icon="📉",
)
st.title("📉 Adaptive AI 기반 하방 리스크 예측 대시보드")
st.caption(
    "M3 Full Model  ·  TFT + VIX·σ Adaptive Pinball Loss  ·  "
    "Multi-Quantile (Q0.1 / Q0.5 / Q0.9)"
)

config, df, train_ds, model = load_everything()

with st.sidebar:
    st.header("⚙️ 설정")
    selected = st.multiselect(
        "종목 선택",
        options=sorted(df[GROUP_ID].unique()),
        default=sorted(df[GROUP_ID].unique()),
    )
    history_days = st.selectbox(
        "이력 표시 기간",
        options=[90, 180, 365, 0],
        index=1,
        format_func=lambda x: f"최근 {x}일" if x > 0 else "전체 기간",
    )
    show_future = st.checkbox("미래 10 거래일 예측 표시", value=True)
    st.markdown("---")
    date_min = pd.Timestamp(df["Date"].min()).strftime("%Y-%m-%d")
    date_max = pd.Timestamp(df["Date"].max()).strftime("%Y-%m-%d")
    st.caption(f"데이터 기간: {date_min} ~ {date_max}")
    st.caption("모델: M3 Full Model (v3, epoch=08)")

with st.spinner("검증 데이터 예측 계산 중..."):
    val_preds = run_val_predictions(model, train_ds, df, config)

tab1, tab2 = st.tabs(["📈 예측 구간 (Quantile Band)", "⚡ λt 동향 (Adaptive Weight)"])

with tab1:
    if not selected:
        st.info("사이드바에서 종목을 선택해주세요.")
    for ticker in selected:
        future_pred = None
        if show_future:
            with st.spinner(f"{ticker} 미래 10 거래일 예측 중..."):
                future_pred = run_future_prediction(model, train_ds, df, config, ticker)
        st.plotly_chart(
            _make_band_fig(ticker, val_preds, future_pred, history_days, df),
            use_container_width=True,
        )

with tab2:
    if not selected:
        st.info("사이드바에서 종목을 선택해주세요.")
    for ticker in selected:
        st.plotly_chart(
            _make_lambda_fig(ticker, df, model),
            use_container_width=True,
        )
