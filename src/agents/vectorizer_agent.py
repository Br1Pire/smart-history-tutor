import os
import json
import pickle
import numpy as np
import faiss
import logging
from sentence_transformers import SentenceTransformer

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# Configuración
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "all-mpnet-base-v2")
SEGMENTED_FILE = os.path.join(BASE_DIR, "data", "processed", "wiki_articles_processed.json")

VECTOR_STORE_DIR = os.path.join(BASE_DIR, "data", "vectorstore_faiss")
os.makedirs(VECTOR_STORE_DIR, exist_ok=True)

FAISS_INDEX_PATH = os.path.join(VECTOR_STORE_DIR, "faiss_index.index")
IDS_PATH = os.path.join(VECTOR_STORE_DIR, "ids.pkl")
TEXTS_PATH = os.path.join(VECTOR_STORE_DIR, "texts.pkl")

logging.info("Cargando modelo de embeddings...")
MODEL = SentenceTransformer(MODEL_PATH if os.path.exists(MODEL_PATH) else "all-mpnet-base-v2")

def load_chunks(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)

def create_embeddings(model, documents):
    texts = []
    ids = []

    for doc in documents:
        header_parts = []
        if doc['title']:
            header_parts.append(f"Título: {doc['title']}")
        if doc.get('section') and doc['section'].lower() != 'general':
            header_parts.append(f"Sección: {doc['section']}")
        header_parts.append(doc['content'])

        composed_text = "\n".join(header_parts).strip()
        texts.append(composed_text)
        ids.append(doc["id"])

    logging.info(f"Generando embeddings para {len(texts)} textos...")
    embeddings = model.encode(texts, batch_size=32, show_progress_bar=True)
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    logging.info("Embeddings generados y normalizados.")
    return ids, texts, embeddings

def save_to_faiss(ids, texts, embeddings, persist = True):
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
        logging.info("Todos los embeddings ya existen en el índice. Nada que agregar.")
        return

    new_embeddings = embeddings[unique_indices]
    new_ids = [ids[i] for i in unique_indices]
    new_texts = [texts[i] for i in unique_indices]

    index.add(new_embeddings)
    final_ids = existing_ids + new_ids
    final_texts = existing_texts + new_texts

    if(persist):
        faiss.write_index(index, FAISS_INDEX_PATH)
        with open(IDS_PATH, "wb") as f:
            pickle.dump(final_ids, f)
        with open(TEXTS_PATH, "wb") as f:
            pickle.dump(final_texts, f)

        logging.info(f"{len(new_ids)} nuevos embeddings almacenados en FAISS.")

    return final_ids, final_texts, index

def vectorize_query(query):
    embedding = MODEL.encode([query]).astype("float32")
    embedding = embedding / np.linalg.norm(embedding, axis=1, keepdims=True)
    logging.info("Vector del query generado y normalizado.")
    return embedding

def vectorize_chunks(chunks, persist = True):
    if not chunks:
        logging.warning("No se encontraron chunks para vectorizar.")
        return

    # logging.info("Cargando modelo de embeddings...")
    # model = SentenceTransformer(MODEL_PATH if os.path.exists(MODEL_PATH) else "all-mpnet-base-v2")

    ids, texts, embeddings = create_embeddings(MODEL, chunks)

    logging.info("Guardando en FAISS...")
    save_to_faiss(ids, texts, embeddings, persist)

    logging.info(f"Vectorización completada. Embeddings almacenados en {VECTOR_STORE_DIR}.")

def vectorize():
    logging.info(f"Cargando chunks desde {SEGMENTED_FILE}...")
    chunks = load_chunks(SEGMENTED_FILE)

    vectorize_chunks(chunks)

if __name__ == "__main__":
    vectorize()
