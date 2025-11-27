import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score


df = pd.read_csv("ia_financial_market_dataset.csv")

print("\nPrimeras filas del dataset:\n")
print(df.head())


# 1. GRÁFICA – Frecuencia de tíckers

plt.figure(figsize=(10,4))
df['ticker'].value_counts().plot(kind='bar', color='dodgerblue')
plt.title("Frecuencia de cada Ticker")
plt.ylabel("Número de registros")
plt.xlabel("Ticker")
plt.tight_layout()
plt.show()


# 2. GRÁFICA – Frecuencia de modelos de IA

plt.figure(figsize=(8,4))
df['ai_model_version'].value_counts().plot(kind='bar', color='purple')
plt.title("Distribución de versiones del modelo de IA")
plt.ylabel("Número de registros")
plt.xlabel("Versión")
plt.tight_layout()
plt.show()



# 3. MATRIZ DE CORRELACIÓN – evidencia de relaciones débiles

numeric_cols = ['open', 'close', 'volume', 'sentiment_score', 'prediction_next_day_return']
corr = df[numeric_cols].corr()

plt.figure(figsize=(7,5))
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Mapa de Correlaciones")
plt.tight_layout()
plt.show()

print("\nMatriz de correlación:\n")
print(corr)


# 4. COMPARACIÓN DE PATRONES ENTRE ACTIVOS

plt.figure(figsize=(10,5))

for ticker in df['ticker'].unique():
    serie = df[df['ticker'] == ticker]['prediction_next_day_return'].reset_index(drop=True)
    plt.plot(serie, label=ticker)

plt.title("Comportamiento de Retornos por Ticker (Cada uno distinto)")
plt.xlabel("Índice temporal dentro del ticker")
plt.ylabel("Retorno del día siguiente")
plt.legend()
plt.tight_layout()
plt.show()


# 5. REGRESIÓN LINEAL – evidencia de R² bajo (no predice bien)

df_reg = df.dropna(subset=numeric_cols)

X = df_reg[['open', 'close', 'volume', 'sentiment_score']]
y = df_reg['prediction_next_day_return']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = LinearRegression()
model.fit(X_scaled, y)
y_pred = model.predict(X_scaled)

r2 = r2_score(y, y_pred)

print("\n---------------------------")
print("RESULTADO DE LA REGRESIÓN")
print("---------------------------")
print("R² obtenido:", r2)
print("\nInterpretación: El R² es bajo, lo que demuestra que el dataset\n"
      "no tiene variables que expliquen bien el retorno del día siguiente.")


# 6. GRÁFICA de y vs y_pred

plt.figure(figsize=(6,5))
plt.scatter(y, y_pred, alpha=0.5)
plt.xlabel("Valores reales del retorno")
plt.ylabel("Valores predichos (modelo)")
plt.title("Relación real vs predicción (dispersa = mala predicción)")
plt.tight_layout()
plt.show()
