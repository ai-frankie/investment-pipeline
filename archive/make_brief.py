import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import json
import os

# Load holdings
df_pos = pd.read_csv('ledger/positions.csv')
held = df_pos['ticker'].tolist()
print("Holdings:", held)

# Load config for universe (optional)
with open('config.json') as f:
    config = json.load(f)
universe = config.get('tickers', [])
print("Universe length:", len(universe))

# Define theme tickers (from universe where possible)
theme_tickers = {
    'Water/Utilities/Water-Tech': ['AWK', 'PHO'],
    'AI-Hardware components (semis, memory, power, cooling)': ['NVDA'],  # add more if in universe: e.g., 'MU', 'AMD', 'INTC' but not in universe
    'AI-Software/Infra': ['MSFT', 'AMZN']  # both in universe
}
# Defense contractors from universe
defense_tickers = [t for t in universe if t in {'PLTR', 'CACI', 'SAIC', 'BAH', 'LDOS', 'RTX', 'LMT', 'NOC', 'GD'}]
print("Defense tickers in universe:", defense_tickers)

# Simple sentiment word lists
POSITIVE_WORDS = {'beat', 'beats', 'surge', 'surges', 'soar', 'soars', 'jump', 'jumps', 'rally', 'rallies', 'upgrade', 'upgraded', 'outperform', 'record', 'strong', 'growth', 'profit', 'profits', 'gain', 'gains', 'bullish', 'win', 'wins', 'won', 'award', 'awarded', 'expand', 'expansion', 'raise', 'raised', 'boost', 'boosts', 'boosted', 'top', 'tops', 'exceed', 'exceeds', 'approval', 'approved', 'breakthrough', 'partnership', 'contract', 'buyback', 'dividend'}
NEGATIVE_WORDS = {'miss', 'misses', 'fall', 'falls', 'drop', 'drops', 'plunge', 'plunges', 'sink', 'sinks', 'slump', 'slumps', 'downgrade', 'downgraded', 'underperform', 'weak', 'loss', 'losses', 'bearish', 'selloff', 'lawsuit', 'sued', 'probe', 'investigation', 'recall', 'fraud', 'layoff', 'layoffs', 'cut', 'cuts', 'warning', 'warns', 'bankruptcy', 'default', 'decline', 'declines', 'fine', 'fined', 'halt', 'halted', 'crash', 'fear', 'fears', 'tariff', 'tariffs', 'shortfall', 'delay', 'delays', 'delayed'}

def get_headlines(ticker, max_items=10):
    try:
        items = yf.Ticker(ticker).news or []
    except Exception as e:
        print(f"[NEWS] {ticker}: {e}")
        return []
    now = datetime.utcnow()
    out = []
    for it in items[:max_items]:
        content = it.get('content') if isinstance(it.get('content'), dict) else it
        title = content.get('title')
        if not title:
            continue
        age_h = 0.0
        pub = content.get('pubDate') or it.get('providerPublishTime')
        try:
            if isinstance(pub, (int, float)):
                age_h = (now - datetime.fromtimestamp(pub)).total_seconds() / 3600
            elif pub:
                age_h = (now - datetime.fromisoformat(str(pub).replace('Z', '+00:00'))).total_seconds() / 3600
        except Exception:
            pass
        if age_h <= 24:  # last 24 hours
            out.append({'title': str(title), 'age_hours': max(0.0, age_h)})
    return out

def simple_sentiment(title):
    words = {w.strip(".,!?:;()'\"").lower() for w in title.split()}
    pos = len(words & POSITIVE_WORDS)
    neg = len(words & NEGATIVE_WORDS)
    if pos + neg == 0:
        return 0.0
    return (pos - neg) / (pos + neg)

def get_sentiment(ticker):
    headlines = get_headlines(ticker)
    if not headlines:
        return 0.0, 0, "No recent news"
    scores = [simple_sentiment(h['title']) for h in headlines]
    # simple average (could weight by recency)
    avg_score = sum(scores) / len(scores)
    return avg_score, len(headlines), headlines[0]['title'] if headlines else ""

# Get data for holdings
price_data = {}
for t in held:
    try:
        df = yf.download(t, period='3d', interval='1d', progress=False)
        if len(df) >= 2:
            close_prev = float(df['Close'].iloc[-2])
            close_last = float(df['Close'].iloc[-1])
            pct = (close_last - close_prev) / close_prev * 100
            price_data[t] = {'prev': close_prev, 'last': close_last, 'pct': pct}
        else:
            price_data[t] = {'prev': None, 'last': None, 'pct': None}
    except Exception as e:
        price_data[t] = {'prev': None, 'last': None, 'pct': None, 'error': str(e)}

news_data = {}
for t in held:
    score, n, headline = get_sentiment(t)
    news_data[t] = {'score': score, 'n': n, 'headline': headline}

# Generate take based on simple thresholds
def get_take(score):
    if score <= -0.2:
        return "BEARISH"
    elif score >= 0.2:
        return "BULLISH"
    else:
        return "NEUTRAL"

# Output markdown
today = datetime.utcnow().strftime('%Y-%m-%d')
out_dir = 'output'
os.makedirs(out_dir, exist_ok=True)
outfile = os.path.join(out_dir, f'nightly_research_{today}.md')
with open(outfile, 'w') as f:
    f.write(f'# Nightly Research {today}\\n\\n')
    f.write('## Held positions\\n')
    for t in held:
        p = price_data.get(t)
        pct = p['pct'] if p and p['pct'] is not None else None
        pct_str = f'{pct:+.2f}%' if pct is not None else 'N/A'
        n = news_data[t]['n']
        headline = news_data[t]['headline']
        if len(headline) > 100:
            headline = headline[:97] + '...'
        take = get_take(news_data[t]['score'])
        f.write(f'- **{t}**: Price change {pct_str}. Headline: {headline}. Take: {take}\\n')
    f.write('\\n## Themes\\n')
    # For each theme, pick a couple of tickers and give a sentence
    for theme, tickers in theme_tickers.items():
        # filter to those we have data for (maybe in universe or held)
        relevant = [t for t in tickers if t in universe or t in held][:2]  # take up to 2
        if not relevant:
            f.write(f'- {theme}: No data available.\\n')
            continue
        # simple aggregate sentiment?
        lines = []
        for t in relevant:
            pct = price_data.get(t, {}).get('pct')
            pct_str = f'{pct:+.2f}%' if pct is not None else 'N/A'
            take = get_take(news_data[t]['score']) if t in news_data else 'NEUTRAL'
            lines.append(f'{t} ({pct_str}, {take})')
        f.write(f'- {theme}: {"; ".join(lines)}\\n')
    f.write('\\n## Gov/Defense + filings\\n')
    # For defense tickers, mention any recent awards or filings? We'll just note if any news.
    defense_news = []
    for t in defense_tickers:
        n = news_data[t]['n'] if t in news_data else 0
        if n > 0:
            headline = news_data[t]['headline']
            if len(headline) > 100:
                headline = headline[:97] + '...'
            defense_news.append(f'{t}: {headline}')
        else:
            defense_news.append(f'{t}: No recent news')
    if defense_news:
        f.write('- ' + '\\n- '.join(defense_news) + '\\n')
    else:
        f.write('- No recent news for defense contractors.\\n')
    f.write('\\n## Unverified\\n')
    f.write('- All news headlines and price changes are sourced from Yahoo Finance via yfinance and should be verified with original sources if needed.\\n')
    f.write('- Sentiment scores are based on a simple keyword lexicon and are approximate.\\n')
print('Written to', outfile)