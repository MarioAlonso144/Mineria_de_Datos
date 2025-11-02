import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


data = pd.read_csv("ia_financial_markey_dataset.csv")

# Convertir columna de fecha a tipo datetime
data['date'] = pd.to_datetime(data['date'])

# Ordenar por fecha
data = data.sort_values('date')

# Crear una variable temporal (número de días)
data['day_index'] = np.arange(len(data))


features = ['day_index', 'open', 'volume', 'sentiment_score']
target = 'close'

X = data[features]
y = data[target]

# Escalar características
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, shuffle=False)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

print(f"R² score: {r2:.4f}")
print(f"Mean Squared Error: {mse:.4f}")


plt.figure(figsize=(10,5))
plt.plot(data['date'].iloc[-len(y_test):], y_test, label='Real', color='blue')
plt.plot(data['date'].iloc[-len(y_pred):], y_pred, label='Predicción', color='orange')
plt.title("Predicción de precios de cierre (Linear Regression)")
plt.xlabel("Fecha")
plt.ylabel("Precio de cierre")
plt.legend()
plt.grid(True)
plt.show()


ultimo_dia = data['day_index'].max() + 1
nuevo_dato = pd.DataFrame({
    'day_index': [ultimo_dia],
    'open': [data['open'].iloc[-1]],  # Usa el último valor conocido
    'volume': [data['volume'].iloc[-1]],
    'sentiment_score': [data['sentiment_score'].iloc[-1]]
})

nuevo_dato_scaled = scaler.transform(nuevo_dato)
prediccion_futura = model.predict(nuevo_dato_scaled)

print(f"\n Predicción del precio de cierre para el próximo día: {prediccion_futura[0]:.2f}")
