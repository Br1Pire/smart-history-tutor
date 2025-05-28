"""
vectorizer.py

Convierte documentos históricos en embeddings y los almacena en ChromaDB para búsquedas semánticas.
"""

import os
import json
from sentence_transformers import SentenceTransformer
import chromadb
from tqdm import tqdm


# Configuración
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_NAME = 'sentence-transformers/all-MiniLM-L6-v2'
INPUT_PATH = os.path.join(BASE_DIR, "data", "raw","wikipedia_historia_es.json")
CHROMA_DIR = os.path.join(BASE_DIR, "data", "vector_store")
COLLECTION_NAME = "historia_universal"

def load_documents(path):
    """Carga documentos desde un archivo JSON"""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def clean_text(text):
    """Limpieza simple de texto"""
    return text.replace("\n", " ").strip()

def create_embeddings(model, documents):
    """Genera embeddings de los documentos"""
    texts = [clean_text(doc["content"]) for doc in documents]
    embeddings = model.encode(texts, show_progress_bar=True, batch_size=32)
    ids = [doc["title"] for doc in documents]
    return ids, texts, embeddings

def save_to_chroma(ids, texts, embeddings):
    """Guarda embeddings y textos en ChromaDB"""
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = client.get_or_create_collection(name=COLLECTION_NAME)

    for i in tqdm(range(len(ids)), desc="🔄 Guardando en ChromaDB"):
        collection.add(
            documents=[texts[i]],
            embeddings=[embeddings[i].tolist()],
            ids=[ids[i]]
        )

    print(f"\n✅ Guardados {len(ids)} documentos en el repositorio vectorial: {CHROMA_DIR}")

def main():
    print("📦 Cargando modelo y documentos...")
    model = SentenceTransformer(MODEL_NAME)
    documents = load_documents(INPUT_PATH)

    print("🧠 Generando embeddings...")
    ids, texts, embeddings = create_embeddings(model, documents)

    print("💾 Guardando en el repositorio vectorial (ChromaDB)...")
    save_to_chroma(ids, texts, embeddings)

if __name__ == "__main__":
    main()
