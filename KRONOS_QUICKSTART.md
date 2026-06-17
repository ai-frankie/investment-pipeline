# Kronos Stock Forecast — Quick Start

## What This Does
Uses the Kronos foundation model (trained on 45+ global exchanges) to forecast OHLCV candlestick data for any Yahoo Finance ticker. No fine-tuning required — the pretrained `kronos-small` model works out of the box.

---

## Step 1 — Clone the Kronos Repo

```bash
git clone https://github.com/shiyu-coder/Kronos.git
```

Place this folder **next to** the scripts (or update `KRONOS_REPO_PATH` in `kronos_forecast.py`).

---

## Step 2 — Install Dependencies

```bash
# From inside the Kronos folder
pip install -r Kronos/requirements.txt

# Extra deps for our scripts
pip install yfinance matplotlib
```

> **No GPU?** It runs on CPU. `kronos-mini` (4.1M params) is fastest; `kronos-small` (24.7M) is more accurate but slower. Expect ~30–120s on CPU for a 400-candle context.

---

## Step 3 — Run a Forecast

```bash
# Forecast AAPL next 24 hours (hourly candles)
python kronos_forecast.py --ticker AAPL --interval 1h --pred_len 24

# Forecast BTC next 48 hours
python kronos_forecast.py --ticker BTC-USD --interval 1h --pred_len 48

# Forecast SPY daily, next 10 days
python kronos_forecast.py --ticker SPY --interval 1d --pred_len 10

# Use the lighter/faster model
python kronos_forecast.py --ticker AAPL --model kronos-mini --interval 1h --pred_len 24
```

Output goes to `./output/`:
- `AAPL_1h_forecast.csv` — OHLCV forecast table
- `AAPL_1h_forecast.png` — historical + forecast chart

---

## Key Parameters

| Flag | Default | Notes |
|------|---------|-------|
| `--ticker` | `AAPL` | Any Yahoo Finance symbol |
| `--interval` | `1h` | `1m` `5m` `15m` `30m` `1h` `1d` |
| `--lookback` | `400` | Max 512 for kronos-small/base |
| `--pred_len` | `24` | Candles to forecast |
| `--model` | `kronos-small` | `kronos-mini`, `kronos-small`, `kronos-base` |
| `--sample_count` | `3` | More paths = smoother forecast, slower |
| `--out_dir` | `output` | Where to save CSV + chart |

---

## Model Selection Guide

| Model | Params | Speed (CPU) | Best For |
|-------|--------|-------------|----------|
| kronos-mini | 4.1M | Fast (~30s) | Quick scans, many tickers |
| kronos-small | 24.7M | Medium (~60s) | Balanced quality/speed |
| kronos-base | 102M | Slow (~3–5min) | Best pretrained quality |

---

## ⚠️ Important Disclaimer

Kronos forecasts **price patterns**, not fundamental value. Treat output as one signal among many — not a trade recommendation. Always apply your own risk management.

---

## Next Steps (Optional)

- **Batch multiple tickers**: modify `kronos_forecast.py` to loop over a list and call `predictor.predict_batch()`
- **Fine-tune on your own data**: follow the Kronos repo's `finetune/` guide
- **Automate daily forecasts**: ask Claude to set this up as a scheduled task
