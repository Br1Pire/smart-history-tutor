import os
import json
import pickle
import numpy as np
import faiss
import logging
from sentence_transformers import SentenceTransformer
from src.config import LOG_FILES, MODEL_PATH, PROCESSED_FILE, FAISS_INDEX_PATH, CATEGORY_FAISS_PATH, IDS_PATH, TEXTS_PATH, VECTORSTORE_DIR

# Configuración de logs

LOG_FILE = LOG_FILES["vectorizer"]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, mode='a', encoding='utf-8'),
        logging.StreamHandler()
    ]
)

SEGMENTED_FILE = PROCESSED_FILE

VECTOR_STORE_DIR = VECTORSTORE_DIR
os.makedirs(VECTOR_STORE_DIR, exist_ok=True)

logging.info("🚀 Cargando modelo de embeddings...")
MODEL = SentenceTransformer(MODEL_PATH if os.path.exists(MODEL_PATH) else "all-mpnet-base-v2")
logging.info("✅ Modelo cargado correctamente.")


def load_chunks(file_path):
    """
    Carga chunks desde un archivo JSON.

    Args:
        file_path (str): Ruta del archivo.

    Returns:
        list: Lista de chunks.
    """
    with open(file_path, "r", encoding="utf-8") as f:
        logging.info(f"📂 Chunks cargados desde: {file_path}")
        return json.load(f)


def create_embeddings(model, documents):
    """
    Genera embeddings de texto y de categorías.

    Args:
        model (SentenceTransformer): Modelo de embeddings.
        documents (list): Lista de documentos a vectorizar.

    Returns:
        tuple: IDs, textos, embeddings de texto y embeddings de categorías.
    """
    text_inputs = []
    category_inputs = []
    ids = []

    for doc in documents:
        header_parts = []
        if doc['title']:
            header_parts.append(f"Título: {doc['title']}")
        if doc.get('section') and doc['section'].lower() != 'general':
            header_parts.append(f"Sección: {doc['section']}")
        header_parts.append(doc['content'])

        composed_text = "\n".join(header_parts).strip()
        text_inputs.append(composed_text)

        categories = doc.get('categories', [])
        categories_text = " ".join(categories).strip()
        category_inputs.append(categories_text)

        ids.append(doc["id"])

    logging.info(f"⚡ Generando embeddings de texto ({len(text_inputs)} entradas)...")
    text_embeddings = model.encode(text_inputs, batch_size=32, show_progress_bar=True)
    text_embeddings = text_embeddings / np.linalg.norm(text_embeddings, axis=1, keepdims=True)
    logging.info("✅ Embeddings de texto generados y normalizados.")

    logging.info(f"⚡ Generando embeddings de categorías ({len(category_inputs)} entradas)...")
    category_embeddings = model.encode(category_inputs, batch_size=32, show_progress_bar=True)
    category_embeddings = category_embeddings / np.linalg.norm(category_embeddings, axis=1, keepdims=True)
    logging.info("✅ Embeddings de categorías generados y normalizados.")

    return ids, text_inputs, text_embeddings, category_embeddings


def save_to_faiss(ids, texts, embeddings, category_embeddings, persist=True):
    """
    Guarda embeddings en FAISS y metadatos asociados.

    Args:
        ids (list): Lista de IDs.
        texts (list): Lista de textos.
        embeddings (np.ndarray): Embeddings de texto.
        category_embeddings (np.ndarray): Embeddings de categorías.
        persist (bool): Si True, guarda los datos en disco.

    Returns:
        tuple: IDs finales, textos finales, índice FAISS, índice de categorías.
    """
    dim = embeddings.shape[1]

    if os.path.exists(FAISS_INDEX_PATH):
        index = faiss.read_index(FAISS_INDEX_PATH)
        categories_index = faiss.read_index(CATEGORY_FAISS_PATH)
        with open(IDS_PATH, "rb") as f:
            existing_ids = pickle.load(f)
        with open(TEXTS_PATH, "rb") as f:
            existing_texts = pickle.load(f)
        logging.info("📂 Índices FAISS existentes cargados.")
    else:
        index = faiss.IndexFlatIP(dim)
        categories_index = faiss.IndexFlatIP(dim)
        existing_ids = []
        existing_texts = []
        logging.info("🆕 Nuevos índices FAISS creados.")

    unique_indices = [i for i, id_ in enumerate(ids) if id_ not in existing_ids]
    if not unique_indices:
        logging.info("ℹ️ No hay nuevos embeddings para añadir al índice.")
        return

    new_embeddings = embeddings[unique_indices]
    new_categories = category_embeddings[unique_indices]
    new_ids = [ids[i] for i in unique_indices]
    new_texts = [texts[i] for i in unique_indices]

    index.add(new_embeddings)
    categories_index.add(new_categories)
    final_ids = existing_ids + new_ids
    final_texts = existing_texts + new_texts

    if persist:
        faiss.write_index(index, FAISS_INDEX_PATH)
        faiss.write_index(categories_index, CATEGORY_FAISS_PATH)
        with open(IDS_PATH, "wb") as f:
            pickle.dump(final_ids, f)
        with open(TEXTS_PATH, "wb") as f:
            pickle.dump(final_texts, f)
        logging.info(f"💾 {len(new_ids)} nuevos embeddings guardados en FAISS.")

    return final_ids, final_texts, index, categories_index


def vectorize_query(query):
    """
    Vectoriza una consulta de texto.

    Args:
        query (str): Consulta.

    Returns:
        np.ndarray: Embedding normalizado del query.
    """
    embedding = MODEL.encode([query]).astype("float32")
    embedding = embedding / np.linalg.norm(embedding, axis=1, keepdims=True)
    logging.info("✅ Vector del query generado y normalizado.")
    return embedding


def vectorize_chunks(chunks, persist=True):
    """
    Vectoriza un conjunto de chunks y los guarda en FAISS.

    Args:
        chunks (list): Lista de chunks.
        persist (bool): Si True, guarda en disco.
    """
    if not chunks:
        logging.warning("⚠️ No se encontraron chunks para vectorizar.")
        return
    ids, texts, embeddings, categories = create_embeddings(MODEL, chunks)
    logging.info("💾 Guardando embeddings en FAISS...")
    save_to_faiss(ids, texts, embeddings, categories, persist)
    logging.info(f"🏁 Vectorización finalizada. Embeddings almacenados en {VECTOR_STORE_DIR}.")


def vectorize():
    """
    Vectoriza los chunks del archivo de entrada y los guarda.
    """
    logging.info(f"🚀 Iniciando carga de chunks desde {SEGMENTED_FILE}...")
    chunks = load_chunks(SEGMENTED_FILE)
    vectorize_chunks(chunks)


if __name__ == "__main__":
    vectorize()
