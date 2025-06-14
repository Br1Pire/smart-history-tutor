import os
import json
import pickle
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# Configuración
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "all-mpnet-base-v2")
SEGMENTED_FILE = os.path.join(BASE_DIR, "data", "processed", "wiki_articles_segmented.json")

VECTOR_STORE_DIR = os.path.join(BASE_DIR, "data", "vectorstore_faiss")
os.makedirs(VECTOR_STORE_DIR, exist_ok=True)

FAISS_INDEX_PATH = os.path.join(VECTOR_STORE_DIR, "faiss_index.index")
IDS_PATH = os.path.join(VECTOR_STORE_DIR, "ids.pkl")
TEXTS_PATH = os.path.join(VECTOR_STORE_DIR, "texts.pkl")

def load_chunks(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)

def create_embeddings(model, documents):
    texts = []
    ids = []

    for doc in documents:
        text_parts = [
            f"Título: {doc['title']}",
            f"Sección: {doc.get('section', '')}",
            f"Contenido: {doc['content']}",
            f"Personas: {', '.join(doc['entities'].get('persons', []))}",
            f"Ubicaciones: {', '.join(doc['entities'].get('locations', []))}",
            f"Fechas: {', '.join(doc['entities'].get('dates', []))}",
            f"Eventos: {', '.join(doc['entities'].get('events', []))}"
        ]
        composed_text = "\n".join([part for part in text_parts if part.strip()])
        texts.append(composed_text)
        ids.append(f"{doc['title']}__{doc['chunk_index']}")

    embeddings = model.encode(texts, batch_size=32, show_progress_bar=True)
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    return ids, texts, embeddings

def save_to_faiss(ids, texts, embeddings):
    dim = embeddings.shape[1]

    if os.path.exists(FAISS_INDEX_PATH):
        index = faiss.read_index(FAISS_INDEX_PATH)
        with open(IDS_PATH, "rb") as f:
            existing_ids = pickle.load(f)
        with open(TEXTS_PATH, "rb") as f:
            existing_texts = pickle.load(f)
    else:
        index = faiss.IndexFlatIP(dim)
        existing_ids = []
        existing_texts = []

    unique_indices = [i for i, id_ in enumerate(ids) if id_ not in existing_ids]
    if not unique_indices:
        print("ℹ️ Todos los embeddings ya existen en el index. Nada que agregar.")
        return

    new_embeddings = embeddings[unique_indices]
    new_ids = [ids[i] for i in unique_indices]
    new_texts = [texts[i] for i in unique_indices]

    index.add(new_embeddings)
    faiss.write_index(index, FAISS_INDEX_PATH)

    with open(IDS_PATH, "wb") as f:
        pickle.dump(existing_ids + new_ids, f)
    with open(TEXTS_PATH, "wb") as f:
        pickle.dump(existing_texts + new_texts, f)

    print(f"✅ {len(new_ids)} nuevos embeddings almacenados en FAISS")

def vectorize():
    print(f"📂 Cargando chunks desde {SEGMENTED_FILE}")
    chunks = load_chunks(SEGMENTED_FILE)
    if not chunks:
        print("⚠️ No se encontraron chunks para vectorizar.")
        return

    print("📦 Cargando modelo de embeddings...")
    model = SentenceTransformer(MODEL_PATH if os.path.exists(MODEL_PATH) else "all-mpnet-base-v2")

    print("🧠 Generando embeddings...")
    ids, texts, embeddings = create_embeddings(model, chunks)

    print("💾 Guardando en FAISS...")
    save_to_faiss(ids, texts, embeddings)

    print(f"✨ Vectorización completada. Embeddings almacenados en {VECTOR_STORE_DIR}")

if __name__ == "__main__":
    vectorize()
