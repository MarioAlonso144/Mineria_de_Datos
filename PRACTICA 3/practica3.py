import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


np.random.seed(42)
tickers = ['AAPL', 'GOOG', 'MSFT', 'AMZN', 'TSLA']
dates = pd.date_range(start='2025-01-01', periods=100)

data = pd.DataFrame({
    'date': np.tile(dates, len(tickers)),
    'ticker': np.repeat(tickers, len(dates)),
    'close': np.random.rand(len(tickers)*len(dates))*100 + 100,
    'volume': np.random.randint(1000, 10000, size=len(tickers)*len(dates)),
    'prediction_next_day_return': np.random.randn(len(tickers)*len(dates))
})

for ticker in tickers:
    df = data[data['ticker'] == ticker]

    # --- Gráfico de línea ---
    plt.figure(figsize=(10,4))
    plt.plot(df['date'], df['close'], label=f'Precio de {ticker}')
    plt.title(f'Precio de Cierre de {ticker}')
    plt.xlabel('Fecha')
    plt.ylabel('Precio')
    plt.legend()
    plt.savefig(f'line_{ticker}.png')
    plt.close()

    # --- Histograma ---
    plt.figure(figsize=(6,4))
    plt.hist(df['prediction_next_day_return'], bins=20, color='skyblue', edgecolor='black')
    plt.title(f'Histograma de Retornos Predichos {ticker}')
    plt.xlabel('Retorno')
    plt.ylabel('Frecuencia')
    plt.savefig(f'hist_{ticker}.png')
    plt.close()

    # --- Boxplot ---
    plt.figure(figsize=(6,4))
    sns.boxplot(y=df['close'], color='lightgreen')
    plt.title(f'Boxplot de Precio de Cierre {ticker}')
    plt.savefig(f'box_{ticker}.png')
    plt.close()

# --- Scatter plot: Precio vs Volumen ---
plt.figure(figsize=(8,6))
for ticker in tickers:
    df = data[data['ticker'] == ticker]
    plt.scatter(df['volume'], df['close'], alpha=0.6, label=ticker)
plt.title('Relación Precio vs Volumen')
plt.xlabel('Volumen')
plt.ylabel('Precio de Cierre')
plt.legend()
plt.savefig('scatter_price_volume.png')
plt.close()

# --- Pie chart: Distribución promedio del volumen ---
avg_volume = data.groupby('ticker')['volume'].mean()
plt.figure(figsize=(6,6))
plt.pie(avg_volume, labels=avg_volume.index, autopct='%1.1f%%', startangle=90)
plt.title('Distribución Promedio del Volumen por Acción')
plt.savefig('pie_avg_volume.png')
plt.close()

