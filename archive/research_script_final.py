import os
import json
import csv
import time
import urllib.request
import urllib.error
import xml.etree.ElementTree as ET
from datetime import datetime

# Helper function to make HTTP requests with timeout and retry
def fetch_url(url, timeout=15):
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'})
        with urllib.request.urlopen(req, timeout=timeout) as response:
            return response.read().decode('utf-8')
    except Exception as e:
        return None

# Get today's date for filename
today = datetime.now().strftime('%Y-%m-%d')

# Read positions.csv
positions_path = r'C:\Projects\investment-pipeline\ledger\positions.csv'
tickers = []
try:
    with open(positions_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            tickers.append(row['ticker'].strip().upper())
except Exception as e:
    tickers = []

# Read config.json (for reference)
config_path = r'C:\Projects\investment-pipeline\config.json'
config = {}
try:
    with open(config_path, 'r') as f:
        config = json.load(f)
except Exception as e:
    config = {}

# Fetch SEC company_tickers.json to map ticker to CIK
sec_tickers_url = "https://www.sec.gov/files/company_tickers.json"
sec_data = fetch_url(sec_tickers_url)
ticker_to_cik = {}
if sec_data:
    try:
        data = json.loads(sec_data)
        for entry in data.values():
            ticker = entry['ticker'].upper()
            cik = str(entry['cik_str']).zfill(10)
            ticker_to_cik[ticker] = cik
    except Exception as e:
        ticker_to_cik = {}

# Define 24 hours ago in Unix timestamp
now = time.time()
twenty_four_hours_ago = now - 24 * 3600

# Function to check if a timestamp string is within last 24 hours
def is_recent(time_str, source='rss'):
    try:
        formats = [
            '%a, %d %b %Y %H:%M:%S %Z',  # RSS
            '%Y-%m-%dT%H:%M:%SZ',         # Atom
            '%Y-%m-%d %H:%M:%S',
            '%d %b %Y %H:%M:%S'
        ]
        for fmt in formats:
            try:
                parsed = datetime.strptime(time_str.strip(), fmt)
                return (now - parsed.timestamp()) <= 24 * 3600
            except ValueError:
                continue
        return False
    except Exception:
        return False

# Function to get Yahoo Finance quote for price change
def get_quote(ticker):
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}?interval=1d&range=1d"
    data = fetch_url(url)
    if not data:
        return None, None
    try:
        j = json.loads(data)
        result = j['chart']['result'][0]
        meta = result['meta']
        current = meta.get('regularMarketPrice')
        previous = meta.get('previousClose')
        if current is not None and previous is not None:
            change_pct = ((current - previous) / previous) * 100
            return current, change_pct
    except Exception:
        pass
    return None, None

# Function to get Yahoo Finance news RSS
def get_news(ticker):
    url = f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}&region=US&lang=en-US"
    data = fetch_url(url)
    if not data:
        return []
    try:
        root = ET.fromstring(data)
        items = []
        for item in root.findall('.//item'):
            title_elem = item.find('title')
            pubdate_elem = item.find('pubDate')
            if title_elem is not None and pubdate_elem is not None:
                title = title_elem.text
                pubdate = pubdate_elem.text
                if is_recent(pubdate, 'rss'):
                    items.append({'title': title, 'date': pubdate})
        return items
    except Exception:
        return []

# Function to get SEC 8-K filings
def get_sec_8k(ticker):
    cik = ticker_to_cik.get(ticker)
    if not cik:
        return []
    url = f"https://www.sec.gov/cgi-bin/browse-edgar?action=getcurrent&CIK={cik}&type=8-k&dateb=&owner=exclude&output=atom"
    data = fetch_url(url)
    if not data:
        return []
    try:
        root = ET.fromstring(data)
        ns = {'atom': 'http://www.w3.org/2005/Atom'}
        entries = []
        for entry in root.findall('.//atom:entry', ns):
            title_elem = entry.find('atom:title', ns)
            updated_elem = entry.find('atom:updated', ns)
            link_elem = entry.find("atom:link[@rel='alternate']", ns)
            if title_elem is not None and updated_elem is not None:
                title = title_elem.text
                updated = updated_elem.text
                if is_recent(updated, 'atom'):
                    link = link_elem.get('href') if link_elem is not None else ''
                    entries.append({'title': title, 'date': updated, 'link': link})
        return entries
    except Exception:
        return []

# Process each ticker
held_results = []
for ticker in tickers:
    current, change_pct = get_quote(ticker)
    news_items = get_news(ticker)
    filings = get_sec_8k(ticker)
    
    # Format price change
    price_str = f"${current:.2f}" if current is not None else "N/A (unverified)"
    change_str = f"{change_pct:+.2f}%" if change_pct is not None else "N/A (unverified)"
    
    # Get most recent news headline
    news_headline = "No recent news (unverified)"
    if news_items:
        news_items.sort(key=lambda x: x['date'], reverse=True)
        news_headline = news_items[0]['title'][:100]  # Truncate
    
    # Get most recent 8-K filing
    filing_info = "No recent 8-K (unverified)"
    if filings:
        filings.sort(key=lambda x: x['date'], reverse=True)
        filing = filings[0]
        filing_info = f"8-K: {filing['title'][:60]} (unverified)"
    
    # Determine simple takeaway based on news headline (unverified)
    takeaway = "Neutral (unverified)"
    if news_headline != "No recent news (unverified)":
        lower_headline = news_headline.lower()
        if any(word in lower_headline for word in ['beat', 'raise', 'up', 'gain', 'win', 'contract', 'profit', 'upbeat']):
            takeaway = "Bullish (unverified - based on headline)"
        elif any(word in lower_headline for word in ['miss', 'cut', 'down', 'loss', 'lawsuit', 'investigation', 'downbeat']):
            takeaway = "Bearish (unverified - based on headline)"
    
    held_results.append({
        'ticker': ticker,
        'price': price_str,
        'change': change_str,
        'news': news_headline,
        'filing': filing_info,
        'takeaway': takeaway
    })

# Define theme tickers (representative samples)
water_tickers = ['PHO', 'ECL', 'XYY', 'AWK', 'WTR']
ai_hardware_tickers = ['NVDA', 'AMD', 'INTC', 'MU', 'AMAT', 'LRCX', 'KLAC', 'ASML', 'TSM', 'AVGO']
ai_software_tickers = ['MSFT', 'GOOGL', 'AMZN', 'IBM', 'ORCL', 'CRM', 'SNOW', 'NET', 'DDOG']
defense_tickers = ['LDOS', 'LMT', 'NOC', 'GD', 'RTX', 'HII', 'LHX', 'NOC']  # LDOS is already in holdings

# Function to get news for a list of tickers (for themes)
def get_theme_news(ticker_list, theme_name):
    news_items = []
    for ticker in ticker_list[:3]:  # Limit to 3 tickers per theme to avoid too many requests
        items = get_news(ticker)
        for item in items:
            # Simple keyword check for theme relevance (unverified)
            title_lower = item['title'].lower()
            if theme_name == 'water' and any(word in title_lower for word in ['water', 'utility', 'utility']):
                news_items.append(f"{ticker}: {item['title'][:60]} (unverified)")
            elif theme_name == 'ai_hardware' and any(word in title_lower for word in ['chip', 'semiconductor', 'memory', 'power', 'cooling', 'ai', 'data center']):
                news_items.append(f"{ticker}: {item['title'][:60]} (unverified)")
            elif theme_name == 'ai_software' and any(word in title_lower for word in ['cloud', 'software', 'ai', 'ai', 'infrastructure']):
                news_items.append(f"{ticker}: {item['title'][:60]} (unverified)")
            elif theme_name == 'defense' and any(word in title_lower for word in ['defense', 'contract', 'dod', 'pentagon', 'military']):
                news_items.append(f"{ticker}: {item['title'][:60]} (unverified)")
    return news_items[:3]  # Limit to 3 items per theme

# Get theme news (unverified)
water_news = get_theme_news(water_tickers, 'water')
ai_hardware_news = get_theme_news(ai_hardware_tickers, 'ai_hardware')
ai_software_news = get_theme_news(ai_software_tickers, 'ai_software')
defense_news = get_theme_news(defense_tickers, 'defense')

# Build markdown output
lines = []
lines.append(f"# Nightly Research Brief - {today}")
lines.append("")
lines.append("## Held positions")
for res in held_results:
    lines.append(f"- **{res['ticker']}**: {res['news']} | Price: {res['price']} ({res['change']}) | {res['filing']} | Takeaway: {res['takeaway']}")
lines.append("")
lines.append("## Themes")
lines.append("- Water/Utilities/Water-Tech:")
if water_news:
    for item in water_news:
        lines.append(f"  - {item}")
else:
    lines.append("  - No recent theme-specific news found (unverified)")
lines.append("- AI-hardware components (semis, memory, power, cooling):")
if ai_hardware_news:
    for item in ai_hardware_news:
        lines.append(f"  - {item}")
else:
    lines.append("  - No recent theme-specific news found (unverified)")
lines.append("- AI-software/infra:")
if ai_software_news:
    for item in ai_software_news:
        lines.append(f"  - {item}")
else:
    lines.append("  - No recent theme-specific news found (unverified)")
lines.append("")
lines.append("## Gov/Defense + filings")
if defense_news:
    for item in defense_news:
        lines.append(f"- {item}")
else:
    lines.append("- No recent government/defense contract news found (unverified)")
lines.append("")
lines.append("## Unverified")
lines.append("- All data (prices, changes, news, filings, takeaways) is unverified due to potential rate limits or fetch failures during this run.")
lines.append("- Theme and government/defense news are based on limited ticker samples and keyword matching (unverified).")
lines.append("- Price data is from Yahoo Finance and may not reflect real-time or exact last 24h change.")
lines.append("- News and filings are sourced from public RSS feeds; verification of completeness and timeliness is unverified.")

# Write to file
output_dir = r'C:\Projects\investment-pipeline\output'
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, f'nightly_research_{today}.md')
with open(output_path, 'w', encoding='utf-8') as f:
    f.write('\n'.join(lines))

# Output summary for the agent
print(f"Research brief written to: {output_path}")
print(f"Processed {len(tickers)} tickers: {', '.join(tickers)}")