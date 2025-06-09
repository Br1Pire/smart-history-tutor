import json
import os
import re

# Configuración
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_FILE = os.path.join(BASE_DIR, "data", "raw", "specific_wiki_articles.json")
OUTPUT_FILE = os.path.join(BASE_DIR, "data", "processed", "wiki_articles_segmented.json")

CHUNK_SIZE = 300   # palabras por fragmento
OVERLAP = 50       # solapamiento entre fragmentos

# Limpieza de texto
def clean_text(text):
    # Reemplazar saltos de línea por espacios
    text = re.sub(r'\n+', ' ', text)
    # Eliminar encabezados tipo '== Sección =='
    text = re.sub(r'={2,}\s?.+?\s?={2,}', '', text)
    # Eliminar referencias como [1], [2], etc.
    text = re.sub(r'\[\d+\]', '', text)
    # Eliminar espacios múltiples y recortar
    return re.sub(r'\s+', ' ', text).strip()

# Segmentación por palabras
def split_text_by_words(text, chunk_size=CHUNK_SIZE, overlap=OVERLAP):
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += chunk_size - overlap
    return chunks

# Cargar el archivo original
with open(INPUT_FILE, "r", encoding="utf-8") as f:
    articles = json.load(f)

segmented = []
for article in articles:
    title = article.get("title", "Untitled")
    content = clean_text(article.get("content", ""))
    chunks = split_text_by_words(content)
    for i, chunk in enumerate(chunks):
        segmented.append({
            "title": title,
            "chunk_index": i,
            "content": chunk
        })

# Asegurarse de que la carpeta exista
os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

# Guardar el nuevo corpus segmentado
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(segmented, f, ensure_ascii=False, indent=2)

print(f"✅ Segmentación completada: {len(segmented)} chunks guardados en '{OUTPUT_FILE}'.")
