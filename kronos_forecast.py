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
import os
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
from datetime import timedelta

# ─── CONFIGURE THIS ───────────────────────────────────────────────────────────
# Point this to wherever you cloned https://github.com/shiyu-coder/Kronos
KRONOS_REPO_PATH = r"C:\Projects\investment-pipeline\Kronos"
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


def _meta_path(csv_name: Path) -> Path:
    return Path(str(csv_name) + ".meta.json")


def _paths_path(csv_name: Path) -> Path:
    return Path(str(csv_name) + ".paths.npz")


def _cache_params(num_paths, sample_count, pred_len, model_key, temperature, top_p) -> dict:
    return {"num_paths": num_paths, "sample_count": sample_count, "pred_len": pred_len,
            "model_key": model_key, "temperature": temperature, "top_p": top_p}


def _load_cached_forecast(csv_name: Path, params: dict, num_paths: int):
    """Return the cached forecast DataFrame only if the sidecar meta matches
    params exactly (parameter-aware cache — was mtime-only, so a changed
    num_paths/model/etc. silently served a stale forecast). A multi-path
    cache additionally requires the .npz sidecar to reconstruct
    .attrs['paths']; missing it forces a miss so callers never get a
    multi-path forecast without its per-path data."""
    meta_path = _meta_path(csv_name)
    if not meta_path.exists():
        return None
    try:
        with open(meta_path) as f:
            saved = json.load(f)
    except (json.JSONDecodeError, OSError):
        return None
    if saved != params:
        return None
    cached = pd.read_csv(csv_name, parse_dates=["timestamps"])
    if num_paths > 1:
        paths_path = _paths_path(csv_name)
        if not paths_path.exists():
            return None
        with np.load(paths_path) as npz:
            keys = sorted(npz.files, key=lambda s: int(s[1:]))
            cached.attrs["paths"] = [npz[k] for k in keys]
    return cached


def _save_forecast_cache(pred_df: pd.DataFrame, csv_name: Path, params: dict, num_paths: int) -> None:
    """Write forecast CSV + sidecar meta/paths atomically (tmp + os.replace) —
    daily and intraday runs can race on the same cache file."""
    tmp = str(csv_name) + ".tmp"
    pred_df.to_csv(tmp, index=False)
    os.replace(tmp, csv_name)

    meta_path = _meta_path(csv_name)
    meta_tmp = str(meta_path) + ".tmp"
    with open(meta_tmp, "w") as f:
        json.dump(params, f, indent=2)
    os.replace(meta_tmp, meta_path)

    if num_paths > 1:
        paths_path = _paths_path(csv_name)
        paths_tmp = str(paths_path) + ".tmp"
        with open(paths_tmp, "wb") as f:
            np.savez(f, **{f"p{i}": arr for i, arr in enumerate(pred_df.attrs["paths"])})
        os.replace(paths_tmp, paths_path)


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
    cache_params = _cache_params(num_paths, sample_count, pred_len, model_key, temperature, top_p)

    if reuse_within_hours > 0 and asof is None and csv_name.exists():
        import time
        age_h = (time.time() - csv_name.stat().st_mtime) / 3600
        if age_h <= reuse_within_hours:
            cached = _load_cached_forecast(csv_name, cache_params, num_paths)
            if cached is not None:
                print(f"Reusing forecast ({age_h:.1f}h old) -> {csv_name}")
                return cached
            print(f"[KRONOS] cache present but parameters changed (or paths missing) "
                  f"— re-running inference")

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
        last_px = float(x_df["close"].iloc[-1])
        terminal = [float(p["close"].iloc[-1]) / last_px - 1.0 for p in paths]
        good = [i for i, p in enumerate(paths) if not p["close"].isna().any()]
        if not good:
            raise RuntimeError("Kronos: all sampled paths contain NaN")
        if len(good) < len(paths):
            print(f"[KRONOS] {len(paths) - len(good)}/{len(paths)} sampled path(s) "
                  f"contained NaN and were excluded")
        order = sorted(good, key=lambda i: terminal[i])
        pred_df = paths[order[len(order) // 2]].copy()   # median REAL path, not a fabricated composite
        pred_df.index.name = "timestamps"
        pred_df = pred_df.reset_index()
        pred_df.attrs["paths"] = [paths[i]["close"].to_numpy() for i in good]
    else:
        pred_df = predictor.predict(
            df=x_df, x_timestamp=x_timestamp, y_timestamp=y_timestamp,
            pred_len=pred_len, T=temperature, top_p=top_p, sample_count=sample_count,
        )
        pred_df.index.name = "timestamps"
        pred_df = pred_df.reset_index()

    if asof is None:
        _save_forecast_cache(pred_df, csv_name, cache_params, num_paths)
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
