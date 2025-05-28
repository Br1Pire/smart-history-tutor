"""
vectorizer.py

Convierte documentos históricos en embeddings y los almacena en un índice FAISS para búsquedas semánticas.
"""

import os
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import pickle

# Configuración
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_NAME = 'sentence-transformers/all-MiniLM-L6-v2'
INPUT_PATH = os.path.join(BASE_DIR, "data", "raw", "wikipedia_historia_es.json")

# Ruta interna segura dentro del proyecto
VECTOR_STORE_DIR = os.path.join(BASE_DIR, "data", "vectorstore_faiss")

FAISS_INDEX_PATH = os.path.join(VECTOR_STORE_DIR, "faiss_index.index")
TEXTS_PATH = os.path.join(VECTOR_STORE_DIR, "texts.pkl")
IDS_PATH = os.path.join(VECTOR_STORE_DIR, "ids.pkl")

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

def save_to_faiss(ids, texts, embeddings):
    """Guarda embeddings y textos en un índice FAISS"""
    os.makedirs(VECTOR_STORE_DIR, exist_ok=True)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    faiss.write_index(index, FAISS_INDEX_PATH)

    with open(TEXTS_PATH, "wb") as f:
        pickle.dump(texts, f)

    with open(IDS_PATH, "wb") as f:
        pickle.dump(ids, f)

    print(f"\n✅ Guardados {len(ids)} documentos en el índice FAISS: {FAISS_INDEX_PATH}")

def main():
    print("📦 Cargando modelo y documentos...")
    model = SentenceTransformer(MODEL_NAME)
    documents = load_documents(INPUT_PATH)

    print("🧠 Generando embeddings...")
    ids, texts, embeddings = create_embeddings(model, documents)
    embeddings = np.array(embeddings).astype("float32")

    print("💾 Guardando en el índice FAISS...")
    save_to_faiss(ids, texts, embeddings)

if __name__ == "__main__":
    main()
