"""
kronos_forecast.py
Loads a pre-trained Kronos model and generates OHLCV forecasts for a stock.

Requirements:
  - Kronos repo cloned to KRONOS_REPO_PATH (see below)
  - pip install yfinance torch transformers

Usage:
    python kronos_forecast.py --ticker AAPL --pred_len 24 --interval 1h
    python kronos_forecast.py --ticker BTC-USD --pred_len 48 --interval 1h --model kronos-mini
"""

import matplotlib
matplotlib.use('Agg')
import sys
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
from datetime import timedelta

# ─── CONFIGURE THIS ───────────────────────────────────────────────────────────
# Point this to wherever you cloned https://github.com/shiyu-coder/Kronos
KRONOS_REPO_PATH = r"C:\AI-Source\Kronos"
# ──────────────────────────────────────────────────────────────────────────────

# Add Kronos to path so we can import its modules
sys.path.insert(0, str(Path(KRONOS_REPO_PATH).resolve()))

from kronos_data_fetcher import fetch_ohlcv

MODEL_CONFIGS = {
    "kronos-mini": {
        "model_id":     "NeoQuasar/Kronos-mini",
        "tokenizer_id": "NeoQuasar/Kronos-Tokenizer-2k",
        "max_context":  2048,
    },
    "kronos-small": {
        "model_id":     "NeoQuasar/Kronos-small",
        "tokenizer_id": "NeoQuasar/Kronos-Tokenizer-base",
        "max_context":  512,
    },
    "kronos-base": {
        "model_id":     "NeoQuasar/Kronos-base",
        "tokenizer_id": "NeoQuasar/Kronos-Tokenizer-base",
        "max_context":  512,
    },
}


def infer_future_timestamps(last_ts: pd.Timestamp, interval: str, n: int) -> pd.Series:
    """Generate n future timestamps based on the candle interval."""
    freq_map = {
        "1m":  timedelta(minutes=1),
        "5m":  timedelta(minutes=5),
        "15m": timedelta(minutes=15),
        "30m": timedelta(minutes=30),
        "1h":  timedelta(hours=1),
        "1d":  timedelta(days=1),
    }
    delta = freq_map.get(interval, timedelta(hours=1))
    return pd.Series([last_ts + delta * (i + 1) for i in range(n)])


_PREDICTOR_CACHE: dict = {}


def load_predictor(model_key: str = "kronos-small", device: str = "auto"):
    """Load (and cache) the Kronos predictor. One load per process, not per call."""
    from model import Kronos, KronosTokenizer, KronosPredictor  # from Kronos repo

    cfg = MODEL_CONFIGS[model_key]

    if device == "auto":
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"

    key = (model_key, device)
    if key not in _PREDICTOR_CACHE:
        print(f"\nDevice: {device}  |  Model: {model_key}")
        print(f"Loading tokenizer ({cfg['tokenizer_id']})...")
        tokenizer = KronosTokenizer.from_pretrained(cfg["tokenizer_id"])
        print(f"Loading model ({cfg['model_id']})...")
        model = Kronos.from_pretrained(cfg["model_id"])
        _PREDICTOR_CACHE[key] = KronosPredictor(model, tokenizer, max_context=cfg["max_context"])
    return _PREDICTOR_CACHE[key]


def run_forecast(
    ticker:    str,
    interval:  str  = "1h",
    lookback:  int  = 400,
    pred_len:  int  = 24,
    model_key: str  = "kronos-small",
    device:    str  = "auto",
    sample_count: int = 3,
    temperature:  float = 1.0,
    top_p:        float = 0.9,
    out_dir:   str  = "output",
    num_paths: int  = 1,
    asof=None,
    prefetched_df: pd.DataFrame | None = None,
    make_plot: bool = True,
    reuse_within_hours: float = 0,
):
    """
    Generate a Kronos forecast.

    num_paths > 1: runs num_paths independent sampled paths (sample_count=1 each)
      and returns their mean as the forecast. Raw close paths are attached in
      pred_df.attrs["paths"] (list of np arrays) for probabilistic scoring.
      Total inference cost equals one call with sample_count=num_paths.
    asof: forecast as of a past timestamp (backtesting, no look-ahead).
    prefetched_df: skip the yfinance fetch and use this OHLCV frame.
    reuse_within_hours: if a forecast CSV newer than this exists (live mode
      only), load and return it instead of re-running inference.
    """
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    csv_name = out_path / f"{ticker.replace('/', '_')}_{interval}_forecast.csv"

    if reuse_within_hours > 0 and asof is None and csv_name.exists():
        import time
        age_h = (time.time() - csv_name.stat().st_mtime) / 3600
        if age_h <= reuse_within_hours:
            print(f"Reusing forecast ({age_h:.1f}h old) -> {csv_name}")
            cached = pd.read_csv(csv_name, parse_dates=["timestamps"])
            return cached

    predictor = load_predictor(model_key, device)

    if prefetched_df is not None:
        df = prefetched_df
        if asof is not None:
            cutoff = pd.to_datetime(asof)
            ts = df["timestamps"]
            if ts.dt.tz is not None and cutoff.tzinfo is None:
                cutoff = cutoff.tz_localize(ts.dt.tz)
            df = df[ts <= cutoff]
        df = df.tail(lookback).reset_index(drop=True)
        if df.empty:
            raise ValueError(f"No candles for {ticker} at or before {asof}")
    else:
        df = fetch_ohlcv(ticker, interval, lookback, asof=asof)

    x_df        = df[["open", "high", "low", "close", "volume"]].copy()
    x_timestamp = df["timestamps"]
    y_timestamp = infer_future_timestamps(df["timestamps"].iloc[-1], interval, pred_len)

    print(f"\nForecasting {pred_len} candles for {ticker}"
          + (f" (asof {asof})" if asof is not None else "")
          + (f" [{num_paths} paths]" if num_paths > 1 else "") + "...")

    if num_paths > 1:
        paths = []
        for _ in range(num_paths):
            p = predictor.predict(
                df=x_df, x_timestamp=x_timestamp, y_timestamp=y_timestamp,
                pred_len=pred_len, T=temperature, top_p=top_p, sample_count=1,
            )
            paths.append(p)
        pred_df = sum(p for p in paths) / len(paths)  # mean path, element-wise
        pred_df.index.name = "timestamps"
        pred_df = pred_df.reset_index()
        pred_df.attrs["paths"] = [p["close"].to_numpy() for p in paths]
    else:
        pred_df = predictor.predict(
            df=x_df, x_timestamp=x_timestamp, y_timestamp=y_timestamp,
            pred_len=pred_len, T=temperature, top_p=top_p, sample_count=sample_count,
        )
        pred_df.index.name = "timestamps"
        pred_df = pred_df.reset_index()

    if asof is None:
        pred_df.to_csv(csv_name, index=False)
        print(f"Forecast saved -> {csv_name}")
        if make_plot:
            _plot(df, pred_df, ticker, interval, out_path)

    return pred_df


def _plot(hist_df: pd.DataFrame, pred_df: pd.DataFrame, ticker: str, interval: str, out_path: Path):
    display_bars = min(120, len(hist_df))
    hist_tail = hist_df.tail(display_bars)

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(hist_tail["timestamps"], hist_tail["close"], label="Historical Close", color="#4c8bf5", linewidth=1.5)
    ax.plot(pred_df["timestamps"],   pred_df["close"],   label="Kronos Forecast",  color="#f5a623", linewidth=2, linestyle="--")

    # Shade the forecast region
    ax.axvspan(pred_df["timestamps"].iloc[0], pred_df["timestamps"].iloc[-1],
               alpha=0.08, color="#f5a623", label="Forecast window")

    ax.set_title(f"{ticker} — Kronos {interval} Forecast ({len(pred_df)} candles)", fontsize=13)
    ax.set_xlabel("Time")
    ax.set_ylabel("Price")
    ax.legend()
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d %H:%M"))
    fig.autofmt_xdate()
    plt.tight_layout()

    chart_path = out_path / f"{ticker.replace('/', '_')}_{interval}_forecast.png"
    plt.savefig(chart_path, dpi=150)
    plt.close()
    print(f"Chart saved   -> {chart_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Kronos stock forecast")
    parser.add_argument("--ticker",       default="AAPL",         help="Yahoo Finance ticker")
    parser.add_argument("--interval",     default="1h",            help="Candle interval: 1m/5m/15m/30m/1h/1d")
    parser.add_argument("--lookback",     default=400, type=int,   help="Historical candles to feed in (max 512 for small/base)")
    parser.add_argument("--pred_len",     default=24,  type=int,   help="How many candles to forecast")
    parser.add_argument("--model",        default="kronos-small",  choices=list(MODEL_CONFIGS), help="Model variant")
    parser.add_argument("--device",       default="auto",          help="auto | cpu | cuda | mps")
    parser.add_argument("--sample_count", default=3,   type=int,   help="Forecast paths to average (higher = smoother)")
    parser.add_argument("--temperature",  default=1.0, type=float, help="Sampling temperature")
    parser.add_argument("--top_p",        default=0.9, type=float, help="Nucleus sampling p")
    parser.add_argument("--out_dir",      default="output",        help="Directory for CSV + chart output")
    args = parser.parse_args()

    pred = run_forecast(
        ticker       = args.ticker,
        interval     = args.interval,
        lookback     = args.lookback,
        pred_len     = args.pred_len,
        model_key    = args.model,
        device       = args.device,
        sample_count = args.sample_count,
        temperature  = args.temperature,
        top_p        = args.top_p,
        out_dir      = args.out_dir,
    )

    print("\n--- Forecast Preview ---")
    print(pred[["timestamps", "open", "high", "low", "close"]].to_string(index=False))
