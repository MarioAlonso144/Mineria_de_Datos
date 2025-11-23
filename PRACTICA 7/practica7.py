import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


data = pd.read_csv("ia_financial_markey_dataset.csv")


features = ['open', 'close', 'volume', 'sentiment_score']
X = data[features]

# Escalar datos (muy importante para K-Means)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


inertia = []
K_range = range(2, 10)
for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)

# --- Gráfico del método del codo ---
plt.figure(figsize=(8,4))
plt.plot(K_range, inertia, marker='o')
plt.title("Método del Codo para elegir número de clusters")
plt.xlabel("Número de clusters (k)")
plt.ylabel("Inercia (Suma de distancias cuadradas)")
plt.grid(True)
plt.show()


k_optimo = 3  # Puedes ajustar según el gráfico anterior
kmeans = KMeans(n_clusters=k_optimo, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X_scaled)

# Añadir los clusters al dataset original
data['cluster'] = clusters


silhouette = silhouette_score(X_scaled, clusters)
print(f"Silhouette Score (calidad de los clusters): {silhouette:.4f}")


plt.figure(figsize=(8,6))
sns.scatterplot(
    x=data['open'],
    y=data['close'],
    hue=data['cluster'],
    palette='tab10',
    alpha=0.7
)
plt.title("Clusters K-Means (open vs close)")
plt.xlabel("Precio de Apertura")
plt.ylabel("Precio de Cierre")
plt.legend(title="Cluster")
plt.show()


plt.figure(figsize=(8,4))
sns.boxplot(x='cluster', y='sentiment_score', data=data, palette='Set2')
plt.title("Distribución del Sentimiento por Cluster")
plt.show()

# --- Promedios de cada cluster ---
cluster_summary = data.groupby('cluster')[features].mean()
print("\nResumen promedio por cluster:")
print(cluster_summary)
