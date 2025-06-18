import os
import pickle
import faiss
import numpy as np
import logging
from sentence_transformers import SentenceTransformer
from src.config import LOG_FILES, FAISS_INDEX_PATH, CATEGORY_FAISS_PATH, IDS_PATH, TEXTS_PATH, MODEL_PATH , VECTORSTORE_DIR, TOP_K_CHUNKS

# Configuración de logs
LOG_FILE = LOG_FILES["retriever"]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, mode='a', encoding='utf-8'),
        logging.StreamHandler()
    ]
)

# Rutas
VECTOR_STORE_DIR = VECTORSTORE_DIR

TOP_K = TOP_K_CHUNKS

# Cargar FAISS y metadatos
logging.info("📂 Cargando índices FAISS y metadatos...")
with open(TEXTS_PATH, "rb") as f:
    texts = pickle.load(f)
with open(IDS_PATH, "rb") as f:
    ids = pickle.load(f)
index = faiss.read_index(FAISS_INDEX_PATH)
categories_index = faiss.read_index(CATEGORY_FAISS_PATH)
logging.info(f"✅ Índices cargados: {index.ntotal} vectores en el índice principal, {categories_index.ntotal} en categorías.")


def normalize_vector(vec):
    """
    Normaliza un vector para que tenga norma unitaria.

    Args:
        vec (np.ndarray): Vector a normalizar.

    Returns:
        np.ndarray: Vector normalizado.
    """
    return vec / np.linalg.norm(vec, axis=1, keepdims=True)


def retrieve_chunks_from_vector(query_vector, top_k):
    """
    Recupera los chunks más relevantes para un vector de query.

    Args:
        query_vector (np.ndarray): Vector de la consulta.
        top_k (int): Número de resultados a devolver.

    Returns:
        list: Lista de diccionarios con los chunks y sus puntuaciones.
    """
    logging.info(f"🔎 Buscando top {top_k} chunks por vector...")
    distances, indices = index.search(query_vector, top_k)
    retrieved = []
    for i, idx in enumerate(indices[0]):
        retrieved.append({
            "id": ids[idx],
            "chunk": texts[idx],
            "score": float(distances[0][i]),
            "rank": i + 1
        })
        logging.info(f"🔹 Rank {i + 1}: ID={ids[idx]}, Score={distances[0][i]:.4f}")
    if not retrieved:
        logging.warning("⚠️ No se encontraron chunks válidos.")
    return retrieved


def retrieve_chunks_from_query_string(query_string, top_k=TOP_K):
    """
    Codifica una consulta de texto y recupera chunks relevantes.

    Args:
        query_string (str): Consulta en texto.
        top_k (int): Número de resultados a devolver.

    Returns:
        list: Lista de chunks recuperados.
    """
    logging.info(f"📝 Generando vector para query: '{query_string}'")
    embedding_model = SentenceTransformer(MODEL_PATH)
    query_vector = embedding_model.encode([query_string]).astype("float32")
    query_vector = normalize_vector(query_vector)
    return retrieve_chunks_from_vector(query_vector, top_k)


def retrieve_chunks_with_category_rerank(query_vector, top_k=5, category_weight=0.3):
    """
    Recupera chunks y reranquea considerando categorías.

    Args:
        query_vector (np.ndarray): Vector de la consulta.
        top_k (int): Número de resultados finales.
        category_weight (float): Peso de la categoría en la puntuación combinada.

    Returns:
        list: Lista de chunks reranqueados.
    """
    logging.info(f"🔎 Buscando top {top_k * 2} para reranking...")
    text_scores, indices = index.search(query_vector, top_k * 2)
    combined = []
    for i, idx in enumerate(indices[0]):
        text_score = text_scores[0][i]
        cat_emb = categories_index.reconstruct(int(idx))
        cat_score = np.dot(cat_emb, query_vector[0])
        combined_score = (1 - category_weight) * text_score + category_weight * cat_score
        combined.append((idx, combined_score))
        logging.info(f"🔹 ID={ids[idx]} TextScore={text_score:.4f}, CatScore={cat_score:.4f}, Combined={combined_score:.4f}")
    combined.sort(key=lambda x: x[1], reverse=True)
    final_results = []
    for rank, (idx, score) in enumerate(combined[:top_k], start=1):
        final_results.append({
            "id": ids[idx],
            "chunk": texts[idx],
            "score": score,
            "rank": rank
        })
        logging.info(f"✅ Rank {rank}: ID={ids[idx]}, Combined Score={score:.4f}")
    return final_results


if __name__ == "__main__":
    query = "cuando fue la primera guerra mundial?"
    results = retrieve_chunks_from_query_string(query)
    for r in results:
        print(f"\n--- Rank {r['rank']} (Score: {r['score']:.4f}) ---")
        print(f"🆔 ID: {r['id']}")
        print(r['chunk'][:500], "...")
