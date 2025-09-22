# análisis_dataset_financiero.py
import pandas as pd
import matplotlib.pyplot as plt
from graphviz import Digraph

# 1. Cargar dataset
df = pd.read_csv("ia_financial_market_dataset.csv")

# 2. Estadísticas descriptivas
print("=== Estadísticas Descriptivas ===")
print(df.describe(include="all"))

# 3. Agrupación por entidades

stats_ticker = df.groupby("ticker").agg({
    "close": "mean",
    "volume": "mean",
    "sentiment_score": "mean",
    "signal": lambda x: (x=="BUY").mean()*100
}).rename(columns={"close": "Precio medio",
                   "volume": "Volumen medio",
                   "sentiment_score": "Sentiment medio",
                   "signal": "% Señales BUY"})

print("\n=== Estadísticas por Ticker ===")
print(stats_ticker)


stats_signal = df.groupby("signal").agg({
    "sentiment_score": "mean",
    "prediction_next_day_return": "mean",
    "ticker": "count"
}).rename(columns={"ticker": "Conteo",
                   "sentiment_score": "Sentiment medio",
                   "prediction_next_day_return": "Predicción media"})

print("\n=== Estadísticas por Signal ===")
print(stats_signal)


stats_model = df.groupby("ai_model_version").agg({
    "ticker": "count",
    "sentiment_score": "mean",
    "prediction_next_day_return": "mean"
}).rename(columns={"ticker": "Nº Predicciones",
                   "sentiment_score": "Sentiment medio",
                   "prediction_next_day_return": "Predicción media"})

print("\n=== Estadísticas por Modelo IA ===")
print(stats_model)

# 4. Diagrama Entidad-Relación con Graphviz
er = Digraph("ER_Diagram", format="png")
er.attr(rankdir="LR")


er.node("Activo", "Activo Financiero\n(ticker)")
er.node("Precio", "Precio Diario\n(open, close, volume, date)")
er.node("Prediccion", "Predicción IA\n(sentiment, predicción, signal)")
er.node("Modelo", "Modelo IA\n(ai_model_version)")


er.edge("Activo", "Precio", label="tiene")
er.edge("Activo", "Prediccion", label="recibe")
er.edge("Precio", "Prediccion", label="asociado a")
er.edge("Modelo", "Prediccion", label="genera")

er.render("ERD_financiero", cleanup=True)

print("\nDiagrama ER exportado como: ERD_financiero.png")

