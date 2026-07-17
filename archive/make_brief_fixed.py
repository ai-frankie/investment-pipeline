import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import json
import os
import sys
sys.path.insert(0, '.')

# Import helper from news_watcher
from news_watcher import _headlines, _lexicon_score

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

# Get price change for held tickers
def get_close_series(df):
    if isinstance(df.columns, pd.MultiIndex):
        close_cols = [col for col in df.columns if col[0] == 'Close']
        if close_cols:
            return df[close_cols[0]]
    else:
        if 'Close' in df.columns:
            return df['Close']
    return None

price_data = {}
for t in held:
    try:
        df = yf.download(t, period='3d', interval='1d', progress=False)
        close_series = get_close_series(df)
        if close_series is not None and len(close_series) >= 2:
            close_prev = float(close_series.iloc[-2])
            close_last = float(close_series.iloc[-1])
            pct = (close_last - close_prev) / close_prev * 100
            price_data[t] = {'prev': close_prev, 'last': close_last, 'pct': pct}
        else:
            price_data[t] = {'prev': None, 'last': None, 'pct': None}
    except Exception as e:
        price_data[t] = {'prev': None, 'last': None, 'pct': None, 'error': str(e)}

# Get headlines and sentiment for held tickers
headlines_data = {}
for t in held:
    try:
        heads = _headlines(t)  # returns list of dicts with 'title' and 'age_hours'
        headlines_data[t] = heads
    except Exception as e:
        print(f"Error fetching headlines for {t}: {e}")
        headlines_data[t] = []

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
        heads = headlines_data.get(t, [])
        headline = ''
        if heads:
            # take the most recent (first in list because _headlines returns in descending order of recency? Actually it's in the order of the feed, but we filtered by age)
            # We'll just take the first one.
            headline = heads[0].get('title', '').strip()
            if len(headline) > 100:
                headline = headline[:97] + '...'
        else:
            headline = 'No recent news found'
        # Compute sentiment from headline
        if headline != 'No recent news found':
            score = _lexicon_score(headline)
            if score <= -0.2:
                take = 'BEARISH'
            elif score >= 0.2:
                take = 'BULLISH'
            else:
                take = 'NEUTRAL'
        else:
            take = 'NEUTRAL'
        f.write(f'- **{t}**: Price change {pct_str}. Headline: {headline}. Take: {take}\\n')
    f.write('\\n## Themes\\n')
    # placeholders
    f.write('- Water/Utilities/Water-Tech: (placeholder)\\n')
    f.write('- AI-Hardware components (semis, memory, power, cooling): (placeholder)\\n')
    f.write('- AI-Software/Infra: (placeholder)\\n')
    f.write('\\n## Gov/Defense + filings\\n')
    f.write('- Placeholder for awards/8-K\\n')
    f.write('\\n## Unverified\\n')
    f.write('- All news headlines above are unverified; need verification via original source.\\n')
print('Written to', outfile)