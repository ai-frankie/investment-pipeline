import yfinance as yf
import pandas as pd
ticker = 'META'
df = yf.download(ticker, period='5d', interval='1d')
print('Shape:', df.shape)
print('Columns:', df.columns.tolist())
print('Close:')
print(df['Close'])
print('Last two closes:')
if len(df) >= 2:
    print(df['Close'].iloc[-2:])
    pct = (df['Close'].iloc[-1] - df['Close'].iloc[-2]) / df['Close'].iloc[-2] * 100
    print(f'Change: {pct:.2f}%')
else:
    print('Not enough data')
