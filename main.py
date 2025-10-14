# generar_dataset_financiero_ia.py
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_business_days(start_date, periods):
    dates = pd.bdate_range(start=start_date, periods=periods)  # días hábiles
    return dates

def synthesize_for_ticker(ticker, dates, seed=None):
    rng = np.random.default_rng(seed)
    n = len(dates)
    price = 100 + np.cumsum(rng.normal(0, 1, size=n))  # random walk desde 100
    openp = price + rng.normal(0, 0.5, size=n)
    closep = price + rng.normal(0, 0.5, size=n)
    volume = (rng.integers(1_000_000, 100_000_000, size=n)).astype(int)
    sentiment = rng.normal(0, 0.25, size=n)
    prediction = rng.normal(0, 0.02, size=n)  # retorno previsto
    model_versions = np.where(rng.random(n) < 0.5, 'v1.0.0', 'v1.1.0')
    signals = np.where(prediction > 0.005, 'BUY', np.where(prediction < -0.005, 'SELL', 'HOLD'))
    df = pd.DataFrame({
        'date': dates.strftime('%Y-%m-%d'),
        'ticker': ticker,
        'open': np.round(openp, 2),
        'close': np.round(closep, 2),
        'volume': volume,
        'sentiment_score': np.round(sentiment, 3),
        'ai_model_version': model_versions,
        'prediction_next_day_return': np.round(prediction, 4),
        'signal': signals
    })
    return df

def main():

    tickers = ['AAPL','MSFT','GOOG','AMZN','TSLA','FB','NFLX','NVDA','BTCUSD','ETHUSD']
    days_per_ticker = 520  # ~2 años hábiles
    all_dfs = []
    start_date = '2023-01-01'
    dates = generate_business_days(start_date, days_per_ticker)
    for i, t in enumerate(tickers):
        df_t = synthesize_for_ticker(t, dates, seed=42+i)
        all_dfs.append(df_t)
    full = pd.concat(all_dfs, ignore_index=True)
    
    full = full.sample(frac=1, random_state=1).reset_index(drop=True)
    print(f"Filas generadas: {len(full)}")
    full.to_csv('ia_financial_market_dataset.csv', index=False)

if __name__ == '__main__':
    main()

