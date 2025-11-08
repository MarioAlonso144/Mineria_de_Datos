import os
import re
import pandas as pd
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
import nltk

from nltk.corpus import stopwords

CSV_PATH = "ia_financial_market_dataset.csv"  # <- ajusta si tu archivo tiene otro nombre/ruta
OUTPUT_IMAGE = "wordcloud.png"
LANGS = ['spanish', 'english']   # stopwords a usar
MIN_WORD_LENGTH = 3              # ignorar palabras muy cortas
ADDITIONAL_STOPWORDS = {'rt', 'https', 'http', 'amp', 'buy', 'sell', 'hold'}  # stopwords adicionales comunes
MIN_WORD_LENGTH = 2

if not os.path.exists(CSV_PATH):
    raise FileNotFoundError(f"No se encontró el archivo: {CSV_PATH}. Ajusta CSV_PATH al archivo correcto.")

df = pd.read_csv(CSV_PATH)

text_cols = [c for c in df.columns if df[c].dtype == object]
if not text_cols:
    raise ValueError("No se encontró ninguna columna de texto (dtype object) en el CSV. "
                     "Si tu texto está en otra columna o en formato distinto, conviértelo a texto.")
print(f"Columnas de texto detectadas: {text_cols}")

# Elegir la columna de texto con más contenido total (sum longitud strings)
df['combined_text'] = df[text_cols].astype(str).agg(' '.join, axis=1)
all_text = " ".join(df['combined_text'].dropna().astype(str).tolist())  

# 4) Normalizar y limpiar texto
def clean_text(s):
    s = s.lower()
    s = re.sub(r'http\S+', ' ', s)       # quitar URLs
    s = re.sub(r'@\w+', ' ', s)          # quitar menciones
    s = re.sub(r'[^a-z0-9áéíóúüñ\s]', ' ', s)  # solo letras/números/acentos/espacios
    s = re.sub(r'\s+', ' ', s)           # colapsar espacios
    return s.strip()

cleaned = clean_text(all_text)

sw = set(STOPWORDS)
for lang in LANGS:
    try:
        sw |= set(stopwords.words(lang))
    except LookupError:
        # si no se descargó nltk stopwords, avisar y continuar con STOPWORDS base
        print(f"Advertencia: stopwords de nltk para '{lang}' no disponibles. Ejecuta nltk.download('stopwords') si quieres mejor filtrado.")
sw |= set(w.lower() for w in ADDITIONAL_STOPWORDS)

tokens = [w for w in cleaned.split() if len(w) >= MIN_WORD_LENGTH and w not in sw]

# Si no hay tokens tras filtrar, avisar
if not tokens:
    raise ValueError("No se encontraron palabras útiles después del filtrado de stopwords. Revisa la columna de texto o reduce stopwords/min length.")

final_text = " ".join(tokens)

# Generar la WordCloud
wc = WordCloud(
    width=1400,
    height=800,
    background_color='white',
    max_words=400,
    collocations=False,   # evita duplicados de bigramas
    stopwords=sw,
).generate(final_text)

plt.figure(figsize=(14,8))
plt.imshow(wc, interpolation='bilinear')
plt.axis('off')
plt.title("Word Cloud del mercado financiero basado en IA", fontsize=18)
plt.tight_layout()
plt.savefig(OUTPUT_IMAGE, dpi=300)
plt.show()

print(f"Wordcloud guardada en: {OUTPUT_IMAGE}")
