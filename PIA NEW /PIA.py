import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
import numpy as np

df = pd.read_csv("ia_financial_markey_dataset.csv")

# ------------------------------
# 1. Frecuencia desigual de tickers
# ------------------------------

plt.figure(figsize=(10,4))
df['ticker'].value_counts().plot(kind='bar', color='skyblue')
plt.title("Frecuencia de cada Ticker (evidencia de concentración)")
plt.ylabel("Número de registros")
plt.tight_layout()
plt.show()

# ------------------------------
# 2. Frecuencia de modelos de IA (v1 domina)
# ------------------------------

plt.figure(figsize=(8,4))
df['ai_model_version'].value_counts().plot(kind='bar', color='purple')
plt.title("Uso de versiones del modelo de IA")
plt.ylabel("Frecuencia")
plt.tight_layout()
plt.show()

# ------------------------------
# 3. Correlación débil entre variables
# ------------------------------

num_cols = ['open','close','volume','sentiment_score','prediction_next_day_return']
corr = df[num_cols].corr()

plt.figure(figsize=(7,5))
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Mapa de correlaciones (evidencia de relaciones débiles)")
plt.tight_layout()
plt.show()


plt.figure(figsize=(10,5))
for t in df['ticker'].unique():
    subset = df[df['ticker'] == t]['prediction_next_day_return']
    plt.plot(subset.values, label=t)

plt.title("Comparación de retornos por ticker (patrones diferentes por activo)")
plt.xlabel("Índice dentro del ticker")
plt.ylabel("Retorno próximo día")
plt.legend()
plt.tight_layout()
plt.show()

# ------------------------------
# 4. Regresión lineal simple para mostrar bajo poder predictivo
# ------------------------------

df = df.dropna(subset=['open','close','volume','sentiment_score','prediction_next_day_return'])

X = df[['open','close','volume','sentiment_score']]
y = df['prediction_next_day_return']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = LinearRegression()
model.fit(X_scaled, y)
y_pred = model.predict(X_scaled)

r2 = r2_score(y, y_pred)

print("\n----------------------------")
print("RESULTADO REGRESIÓN LINEAL:")
print("----------------------------")
print("R² =", r2)
print("Interpretación: Un R² bajo demuestra que las variables\nno predicen bien el retorno del día siguiente,\ntal como se mencionó en el monólogo.")
