import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split


data = pd.read_csv("ia_financial_market_dataset.csv")


X = data[['open', 'volume', 'sentiment_score']]  # variables independientes
y = data['prediction_next_day_return']           # variable dependiente


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)


r2 = r2_score(y_test, y_pred)
print("R² score:", r2)



# --- Dispersión real vs predicho ---
plt.figure(figsize=(6,6))
sns.scatterplot(x=y_test, y=y_pred, alpha=0.6)
plt.xlabel("Valores reales")
plt.ylabel("Valores predichos")
plt.title("Predicción vs Realidad (Regresión Lineal)")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')  # línea ideal
plt.show()

# --- Residuos ---
residuals = y_test - y_pred
plt.figure(figsize=(6,4))
sns.histplot(residuals, bins=30, kde=True, color="purple")
plt.title("Distribución de residuos")
plt.xlabel("Error (real - predicho)")
plt.show()

# --- Importancia de coeficientes ---
coef = pd.DataFrame({
    "Variable": X.columns,
    "Coeficiente": model.coef_
})
plt.figure(figsize=(6,4))
sns.barplot(x="Variable", y="Coeficiente", data=coef, palette="viridis")
plt.title("Coeficientes del modelo lineal")
plt.show()

