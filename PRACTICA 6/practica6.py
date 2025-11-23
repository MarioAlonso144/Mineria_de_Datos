import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split


data = pd.read_csv("ia_financial_market_dataset.csv")


X = data[['open', 'volume', 'sentiment_score']]  # variables independientes
y = data['prediction_next_day_return']           # variable dependiente


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


knn = KNeighborsRegressor(n_neighbors=5)
knn.fit(X_train, y_train)

pred = knn.predict(X_test)
# model = LinearRegression()
# model.fit(X_train, y_train)

# pred = model.predict(X_test)


r2 = r2_score(y_test, pred)
mae = mean_absolute_error(y_test, pred)
print("KNN Model Results:")
print(f"Mean Absolute Error: {mae:.6f}")
print("R² score:", r2)



# --- Dispersión real vs predicho ---
plt.figure(figsize=(6,6))
sns.scatterplot(x=y_test, y=pred, alpha=0.6)
plt.xlabel("Valores reales")
plt.ylabel("Valores predichos")
plt.title("Predicción vs Realidad (KNN)")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')  # línea ideal
plt.show()

# --- Residuos ---
residuals = y_test - pred
plt.figure(figsize=(6,4))
sns.histplot(residuals, bins=30, kde=True, color="purple")
plt.title("Distribución de residuos")
plt.xlabel("Error (real - predicho)")
plt.show()

# --- Importancia de coeficientes ---
'''coef = pd.DataFrame({
    "Variable": X.columns,
    "Coeficiente": model.coef_
})
plt.figure(figsize=(6,4))
sns.barplot(x="Variable", y="Coeficiente", data=coef, palette="viridis")
plt.title("Coeficientes del modelo lineal")
plt.show() '''

neighbors = range(1, 21)
scores = []
for k in neighbors:
    knn = KNeighborsRegressor(n_neighbors=k)
    knn.fit(X_train, y_train)
    scores.append(knn.score(X_test, y_test))

plt.figure(figsize=(8,4))
plt.plot(neighbors, scores, marker='o')
plt.title("Precisión (R²) según número de vecinos")
plt.xlabel("Número de vecinos (k)")
plt.ylabel("R² Score")
plt.grid(True)
plt.show()
