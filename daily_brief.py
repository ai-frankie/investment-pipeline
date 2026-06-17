"""
daily_brief.py
Zero-Claude-token daily commentary: Ollama (local llama3.1:8b) reads the
latest proposals + ledger and writes a plain-English brief, then pushes it
to the NotebookLM Brain so it's queryable later.

Runs as the last scheduler step. Ollama down or NotebookLM unauthenticated
-> degrades gracefully, never blocks the pipeline.

Usage:
    python daily_brief.py            # generate + push to Brain
    python daily_brief.py --no-push  # generate only
"""

import argparse
import json
import subprocess
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "llama3.1:8b"
BRAIN_ID = "a47639d0-d122-46db-9ba5-fd1d40372769"
BRIEF_DIR = Path("output/briefs")


def latest_proposals() -> pd.DataFrame | None:
    files = sorted(Path("output").glob("proposals_*.csv"))
    return pd.read_csv(files[-1]) if files else None


def ledger_summary() -> str:
    eq = Path("ledger/equity_curve.csv")
    if not eq.exists():
        return "No paper trades yet."
    df = pd.read_csv(eq)
    last = df.iloc[-1]
    return (f"Paper equity ${last['equity']:,.0f} "
            f"(cash ${last['cash']:,.0f}, positions ${last['market_value']:,.0f})")


def generate_brief() -> str:
    props = latest_proposals()
    if props is None:
        return ""
    table = props[["ticker", "score", "action", "regime_ok", "er_blackout",
                   "macro_event", "news_flag", "target_value"]].to_string(index=False)
    prompt = (
        "You are a buy-side analyst writing a 150-word daily brief for a "
        "disciplined retail quant. Today's pipeline output (score 0-1, "
        "BUY>=0.7, entries blocked by blackouts/regime gates):\n\n"
        f"{table}\n\n{ledger_summary()}\n\n"
        "Write: 1) one-line market posture, 2) top 3 names by score with one "
        "reason each, 3) any blackout/veto in effect and when it lifts. "
        "No hype, no advice disclaimer, plain English."
    )
    resp = requests.post(OLLAMA_URL, json={"model": MODEL, "prompt": prompt,
                                           "stream": False}, timeout=300)
    resp.raise_for_status()
    return resp.json().get("response", "").strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-push", action="store_true")
    args = parser.parse_args()

    try:
        brief = generate_brief()
    except Exception as e:
        print(f"[BRIEF] Ollama unavailable: {e}")
        return
    if not brief:
        print("[BRIEF] no proposals to summarize")
        return

    BRIEF_DIR.mkdir(parents=True, exist_ok=True)
    path = BRIEF_DIR / f"brief_{datetime.now().strftime('%Y%m%d')}.md"
    path.write_text(f"# Daily Brief {datetime.now().strftime('%Y-%m-%d')}\n\n{brief}\n",
                    encoding="utf-8")
    print(f"[BRIEF] saved -> {path}\n\n{brief}")

    if not args.no_push:
        try:
            subprocess.run(["notebooklm", "source", "add", str(path),
                            "--notebook", BRAIN_ID],
                           capture_output=True, timeout=120, check=True)
            print("[BRIEF] pushed to Brain")
        except Exception as e:
            print(f"[BRIEF] Brain push failed (brief still saved): {e}")


if __name__ == "__main__":
    main()
