"""
run_scheduler.py
Daily pipeline scheduler — runs every weekday at 4:30pm ET.

Jobs (all inside pipeline.run_pipeline — contracts, congress, news, Kronos
forecasts, scoring, paper-ledger record/mark):
  1. Scoring pipeline
  2. Windows toast notification on completion

Usage:
    python run_scheduler.py            # starts scheduler, runs daily at 4:30pm ET
    python run_scheduler.py --now      # run immediately (for testing)

Requirements:
    pip install apscheduler plyer
"""

import argparse
import json
import sys
import traceback
from datetime import datetime
from pathlib import Path

from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger

LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

CONFIG_PATH = "config.json"


def load_config():
    with open(CONFIG_PATH) as f:
        return json.load(f)


def notify(title: str, msg: str, duration: int = 6):
    """Windows toast notification via plyer — fails silently if unavailable."""
    try:
        from plyer import notification
        notification.notify(title=title, message=msg, timeout=duration, app_name="Quant Pipeline")
    except Exception:
        print(f"[NOTIFY] {title}: {msg}")


def run_pipeline():
    """Full daily pipeline (everything happens inside pipeline.run_pipeline)."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = LOG_DIR / f"pipeline_{ts}.log"

    # Redirect stdout to log file while keeping console output
    class Tee:
        def __init__(self, *files):
            self.files = files
        def write(self, obj):
            for f in self.files:
                f.write(obj)
                f.flush()
        def flush(self):
            for f in self.files:
                f.flush()

    log_file = open(log_path, "w", encoding="utf-8")
    original_stdout = sys.stdout
    sys.stdout = Tee(original_stdout, log_file)

    try:
        cfg = load_config()
        tickers = cfg.get("tickers", [])

        print(f"\n{'='*60}")
        print(f"DAILY PIPELINE — {datetime.now().strftime('%Y-%m-%d %H:%M:%S ET')}")
        print(f"Tickers: {tickers}")
        print(f"{'='*60}")

        import pipeline as pipeline_mod
        # Kronos off by default: 2026-06-15 walk-forward backtest showed IC -0.13 /
        # 43% hit-rate at 1h/24-candle config (no usable edge). Toggle config.json
        # "use_kronos" back to true once a re-test at a saner horizon proves edge.
        # See notes/hermes_notes/2026-06-15-CARRY-FORWARD.md
        use_kronos = cfg.get("use_kronos", True)
        proposals = pipeline_mod.run_pipeline(tickers=tickers, use_kronos=use_kronos)

        buy_count = len(proposals[proposals["action"] == "BUY"]) if not proposals.empty else 0
        scored = len(proposals)
        risk_flags = len(proposals[proposals["news_flag"] == "NEWS-RISK"]) if not proposals.empty else 0

        print("\n" + "=" * 60)
        print("DAILY SUMMARY")
        print("=" * 60)
        print(f"Tickers scored: {scored}")
        print(f"BUY signals:    {buy_count}")
        print(f"News risk:      {risk_flags}")
        print(f"\nLog saved -> {log_path}")

        notify(
            "Daily Pipeline Complete",
            f"{buy_count} BUY | {scored} scored | {risk_flags} news-risk | {datetime.now().strftime('%H:%M')}",
        )

        # Local-LLM daily brief -> NotebookLM Brain (zero Claude tokens)
        try:
            import daily_brief
            daily_brief.main()
        except Exception as e:
            print(f"[BRIEF] skipped: {e}")

    except Exception:
        err = traceback.format_exc()
        print(f"\nPIPELINE ERROR:\n{err}")
        notify("Pipeline ERROR", f"Check {log_path.name} for details", duration=10)

    finally:
        sys.stdout = original_stdout
        log_file.close()


def main():
    parser = argparse.ArgumentParser(description="Daily quant pipeline scheduler")
    parser.add_argument("--now", action="store_true", help="Run immediately instead of waiting for schedule")
    args = parser.parse_args()

    if args.now:
        print("Running pipeline immediately...")
        run_pipeline()
        return

    scheduler = BlockingScheduler(timezone="US/Eastern")

    trigger = CronTrigger(
        day_of_week="mon-fri",
        hour=16,
        minute=30,
        timezone="US/Eastern",
    )

    scheduler.add_job(
        run_pipeline,
        trigger=trigger,
        id="daily_pipeline",
        name="Daily Kronos + Gov Contract Pipeline",
        replace_existing=True,
    )

    print(f"Scheduler started.")
    try:
        job = scheduler.get_jobs()[0]
        next_run = getattr(job, "next_run_time", None) or getattr(job, "next_fire_time", None)
        if next_run:
            print(f"Next run: {next_run.strftime('%Y-%m-%d %H:%M %Z')}")
    except Exception:
        print("Next run: Mon-Fri 4:30pm ET")
    print(f"Runs: Mon-Fri at 4:30pm ET")
    print(f"Logs: {LOG_DIR.resolve()}")
    print(f"Press Ctrl+C to stop.\n")

    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        print("Scheduler stopped.")


if __name__ == "__main__":
    main()
