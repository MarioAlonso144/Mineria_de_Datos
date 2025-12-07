import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import os

CSV_PATH = "bitcoin_2020_2024.csv"
OUT_DIR = "pia_outputs"
RANDOM_STATE = 42


os.makedirs(OUT_DIR, exist_ok=True)

df = pd.read_csv(CSV_PATH)
for col in ["Open","High","Low","Close","Adj Close","Volume"]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

print(df.dtypes)
print(df.head())

if "date" in df.columns:
    df.rename(columns={"date": "Date"}, inplace=True)

df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values("Date").reset_index(drop=True)

expected_cols = ["Open","High","Low","Close","Adj Close","Volume"]
df.columns = [c.title() if c.lower() in [e.lower() for e in expected_cols] else c for c in df.columns]

df["return"] = df["Close"].pct_change()
df["target_next_return"] = df["return"].shift(-1)

df["roll_mean_7"] = df["Close"].rolling(7).mean()
df["roll_std_7"] = df["return"].rolling(7).std()
df["momentum_7"] = df["Close"] - df["Close"].shift(7)

df["lag_return_1"] = df["return"].shift(1)
df["lag_return_2"] = df["return"].shift(2)

df_model = df.dropna(subset=["target_next_return", "return", "roll_mean_7", "roll_std_7",
                             "momentum_7", "lag_return_1"])

df_model.head().to_csv(os.path.join(OUT_DIR, "preview_features.csv"), index=False)

plt.figure(figsize=(12,5))
plt.plot(df["Date"], df["Close"], label="Close")
plt.plot(df["Date"], df["Close"].rolling(30).mean(), label="RollingMean30", linestyle="--")
plt.title("Precio de cierre y Rolling Mean 30 días")
plt.xlabel("Fecha")
plt.ylabel("Precio (USD)")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "price_and_rolling_mean.png"))
plt.close()


plt.figure(figsize=(8,5))
sns.histplot(df["return"].dropna(), bins=80, kde=True)
plt.title("Histograma de retornos diarios")
plt.xlabel("Retorno diario")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "hist_returns.png"))
plt.close()


plt.figure(figsize=(10,4))
plt.plot(df["Date"], df["roll_std_7"], color="orange")
plt.title("Volatilidad (rolling std 7 días)")
plt.xlabel("Fecha")
plt.ylabel("Std(Return)")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "rolling_volatility.png"))
plt.close()

# ---------------- REGRESIÓN ----------------

feature_cols = ["return","roll_mean_7","roll_std_7","momentum_7","lag_return_1","lag_return_2","Volume"]
X = df_model[feature_cols].values
y = df_model["target_next_return"].values

split_idx = int(len(X)*0.8)
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

lr = LinearRegression()
lr.fit(X_train_s, y_train)
y_pred = lr.predict(X_test_s)

r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

with open(os.path.join(OUT_DIR, "regression_metrics.txt"), "w") as f:
    f.write(f"R2: {r2}\nMAE: {mae}\n\nCoeficientes:\n")
    for name, coef in zip(feature_cols, lr.coef_):
        f.write(f"{name}: {coef}\n")

# Real vs Predicho
plt.figure(figsize=(6,6))
plt.scatter(y_test, y_pred, alpha=0.5)
mn = min(min(y_test), min(y_pred))
mx = max(max(y_test), max(y_pred))
plt.plot([mn,mx],[mn,mx],'r--')
plt.xlabel("Valor real (target_next_return)")
plt.ylabel("Predicción")
plt.title(f"Regresión Lineal: Real vs Predicho (R2={r2:.4f})")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "regression_real_vs_pred.png"))
plt.close()

resid = y_test - y_pred
plt.figure(figsize=(8,4))
sns.histplot(resid, bins=50, kde=True)
plt.title("Distribución de residuos")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "regression_residuals.png"))
plt.close()

# ---------------- CLUSTERING ----------------

clust_features = df_model[["return","roll_std_7"]].dropna()

scaler2 = StandardScaler()
X_clust = scaler2.fit_transform(clust_features)

best_k = 2
best_score = -1
best_labels = None

for k in range(2,7):
    km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
    labels = km.fit_predict(X_clust)
    score = silhouette_score(X_clust, labels)
    if score > best_score:
        best_k = k
        best_score = score
        best_labels = labels

df_model_clust = df_model.loc[clust_features.index].copy()
df_model_clust["cluster"] = best_labels

with open(os.path.join(OUT_DIR, "clustering_metrics.txt"), "w") as f:
    f.write(f"Mejor k: {best_k}\nSilhouette: {best_score}\n")

# Scatter clustering
plt.figure(figsize=(8,6))
sns.scatterplot(
    x=df_model_clust["return"],
    y=df_model_clust["roll_std_7"],
    hue=df_model_clust["cluster"],
    palette="tab10",
    alpha=0.7
)
plt.title(f"KMeans Clustering (k={best_k}) retorno vs vol (silhouette={best_score:.3f})")
plt.xlabel("Retorno diario")
plt.ylabel("Volatilidad (std 7d)")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "clustering_return_volatility.png"))
plt.close()

# Boxplot por cluster
plt.figure(figsize=(8,5))
sns.boxplot(x="cluster", y="return", data=df_model_clust)
plt.title("Boxplot de retornos por cluster")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "boxplot_return_by_cluster.png"))
plt.close()

# Heatmap de correlación
corr = df_model[feature_cols + ["target_next_return"]].corr()

plt.figure(figsize=(8,6))
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", vmin=-1, vmax=1)
plt.title("Mapa de correlación (features y target)")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "heatmap_features_target.png"))
plt.close()

print("✔ Proceso completado.")
print("Resultados guardados en carpeta:", OUT_DIR)
print(f"R2 = {r2:.6f} | MAE = {mae:.6f} | Mejor k={best_k} | Silhouette={best_score:.4f}")
