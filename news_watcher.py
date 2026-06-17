"""
news_watcher.py
Current-event layer v2: Yahoo Finance headlines (free, via yfinance) scored by
FinBERT (ProsusAI/finbert, local CPU) with recency weighting, plus a hard
event-class veto. Falls back to a keyword lexicon if transformers/model are
unavailable (offline-safe — the pipeline never blocks on this layer).

Two-layer design (the proven retail pattern):
  1. Sentiment score: recency-weighted mean of P(pos)-P(neg), half-life 12h,
     trusted only with 2+ headlines.
  2. Hard veto: headlines matching adverse event classes (fraud, investigation,
     guidance cut, restatement, going concern, bankruptcy, deal break, abrupt
     C-suite exit) block NEW entries regardless of sentiment sign.

Output per ticker: {"sent", "n", "flag", "veto", "veto_reason"}
  flag: NEWS-RISK / NEWS-POS / -
  veto: True blocks BUY -> HOLD in the pipeline (entry gate only).

Usage:
    python news_watcher.py --tickers NVDA META CACI
    python news_watcher.py --tickers NVDA --lexicon     # skip FinBERT
"""

import argparse
import math
from datetime import datetime, timezone

import yfinance as yf

POSITIVE = {
    "beat", "beats", "surge", "surges", "soar", "soars", "jump", "jumps",
    "rally", "rallies", "upgrade", "upgraded", "outperform", "record",
    "strong", "growth", "profit", "profits", "gain", "gains", "bullish",
    "win", "wins", "won", "award", "awarded", "expand", "expansion",
    "raise", "raised", "boost", "boosts", "boosted", "top", "tops",
    "exceed", "exceeds", "approval", "approved", "breakthrough",
    "partnership", "contract", "buyback", "dividend",
}
NEGATIVE = {
    "miss", "misses", "fall", "falls", "drop", "drops", "plunge", "plunges",
    "sink", "sinks", "slump", "slumps", "downgrade", "downgraded",
    "underperform", "weak", "loss", "losses", "bearish", "selloff",
    "lawsuit", "sued", "probe", "investigation", "recall", "fraud",
    "layoff", "layoffs", "cut", "cuts", "warning", "warns", "bankruptcy",
    "default", "decline", "declines", "fine", "fined", "halt", "halted",
    "crash", "fear", "fears", "tariff", "tariffs", "shortfall", "delay",
    "delays", "delayed",
}

# Hard event-class veto: these block new entries even on positive sentiment
VETO_EVENTS = {
    "fraud/accounting": ["fraud", "accounting irregularit", "restatement", "restates"],
    "investigation": ["sec investigation", "sec probe", "doj investigation",
                      "doj probe", "subpoena", "under investigation",
                      "criminal probe", "regulatory investigation"],
    "guidance cut": ["cuts guidance", "lowers guidance", "guidance cut",
                     "cuts outlook", "lowers outlook", "slashes forecast",
                     "cuts forecast", "warns on profit", "profit warning"],
    "c-suite exit": ["cfo resigns", "ceo resigns", "cfo departs", "ceo departs",
                     "cfo steps down", "ceo steps down", "ceo ousted", "cfo ousted"],
    "solvency": ["going concern", "bankruptcy", "chapter 11", "liquidity crisis",
                 "debt default", "misses payment"],
    "deal break": ["terminates merger", "deal collapses", "abandons merger",
                   "merger blocked", "acquisition blocked", "deal falls"],
    "adverse legal": ["adverse verdict", "jury awards", "loses lawsuit",
                      "court rules against"],
}

HALF_LIFE_HOURS = 12
MAX_AGE_HOURS = 48
MIN_HEADLINES = 2

_FINBERT = None


def _load_finbert():
    """Lazy global load — ~10s once per process, then ~ms per headline batch."""
    global _FINBERT
    if _FINBERT is None:
        from transformers import pipeline as hf_pipeline
        _FINBERT = hf_pipeline("text-classification", model="ProsusAI/finbert",
                               top_k=None, device=-1)
    return _FINBERT


def _headlines(ticker: str, max_items: int = 15) -> list[dict]:
    """[{title, age_hours}] — handles both old and new yfinance news schemas."""
    try:
        items = yf.Ticker(ticker).news or []
    except Exception as e:
        print(f"[NEWS] {ticker}: {e}")
        return []
    now = datetime.now(timezone.utc)
    out = []
    for it in items[:max_items]:
        content = it.get("content") if isinstance(it.get("content"), dict) else it
        title = content.get("title")
        if not title:
            continue
        age_h = 0.0
        pub = content.get("pubDate") or it.get("providerPublishTime")
        try:
            if isinstance(pub, (int, float)):
                age_h = (now - datetime.fromtimestamp(pub, tz=timezone.utc)).total_seconds() / 3600
            elif pub:
                age_h = (now - datetime.fromisoformat(str(pub).replace("Z", "+00:00"))).total_seconds() / 3600
        except Exception:
            pass
        if age_h <= MAX_AGE_HOURS:
            out.append({"title": str(title), "age_hours": max(0.0, age_h)})
    return out


def _lexicon_score(title: str) -> float:
    words = {w.strip(".,!?:;()'\"").lower() for w in title.split()}
    pos, neg = len(words & POSITIVE), len(words & NEGATIVE)
    return (pos - neg) / (pos + neg) if pos + neg else 0.0


def _finbert_scores(titles: list[str]) -> list[float]:
    """P(pos) - P(neg) per title, batched."""
    clf = _load_finbert()
    results = clf(titles, truncation=True, max_length=64, batch_size=16)
    scores = []
    for r in results:
        probs = {d["label"].lower(): d["score"] for d in r}
        scores.append(probs.get("positive", 0.0) - probs.get("negative", 0.0))
    return scores


def _check_veto(titles: list[str]) -> tuple[bool, str]:
    joined = [t.lower() for t in titles]
    for category, patterns in VETO_EVENTS.items():
        for t in joined:
            if any(p in t for p in patterns):
                return True, category
    return False, ""


def get_news_signals(tickers, use_finbert: bool = True) -> dict:
    finbert_ok = use_finbert
    out = {}
    for t in tickers:
        heads = _headlines(t)
        if not heads:
            out[t] = {"sent": 0.0, "n": 0, "flag": "-", "veto": False, "veto_reason": ""}
            continue
        titles = [h["title"] for h in heads]

        raw = None
        if finbert_ok:
            try:
                raw = _finbert_scores(titles)
            except Exception as e:
                print(f"[NEWS] FinBERT unavailable ({e}) — lexicon fallback")
                finbert_ok = False
        if raw is None:
            raw = [_lexicon_score(x) for x in titles]

        weights = [math.exp(-math.log(2) * h["age_hours"] / HALF_LIFE_HOURS) for h in heads]
        sent = round(sum(w * s for w, s in zip(weights, raw)) / max(sum(weights), 1e-9), 3)

        veto, reason = _check_veto(titles)
        flag = "-"
        if len(titles) >= MIN_HEADLINES:
            if sent <= -0.3:
                flag = "NEWS-RISK"
            elif sent >= 0.3:
                flag = "NEWS-POS"
        if veto:
            flag = "VETO"
        out[t] = {"sent": sent, "n": len(titles), "flag": flag,
                  "veto": veto, "veto_reason": reason}
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Headline sentiment per ticker (FinBERT + event veto)")
    parser.add_argument("--tickers", nargs="+", required=True)
    parser.add_argument("--lexicon", action="store_true", help="Skip FinBERT, lexicon only")
    args = parser.parse_args()
    for t, s in get_news_signals(args.tickers, use_finbert=not args.lexicon).items():
        extra = f"  VETO({s['veto_reason']})" if s["veto"] else ""
        print(f"{t:6s} sent={s['sent']:+.3f}  headlines={s['n']:2d}  {s['flag']}{extra}")
