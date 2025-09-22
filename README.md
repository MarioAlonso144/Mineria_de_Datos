# IA Financial Market Dataset

Descripción:
Conjunto de datos para experimentos de modelos de IA en mercado financiero. Contiene precios diarios, volúmenes, señales y predicciones generadas por un modelo IA.

Columnas:
- date: fecha YYYY-MM-DD
- ticker: símbolo del activo
- open, close: precios
- volume: volumen negociado
- sentiment_score: puntuación de sentimiento (-1 a 1)
- ai_model_version: versión del modelo
- prediction_next_day_return: predicción de retorno (decimal)
- signal: BUY/HOLD/SELL

Filas: 5200 (ejemplo con 10 tickers × ~2 años)

