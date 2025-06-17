import os
import pickle
import faiss
import numpy as np
import logging
from sentence_transformers import SentenceTransformer

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Rutas
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VECTOR_STORE_DIR = os.path.join(BASE_DIR, "data", "vectorstore_faiss")
FAISS_INDEX_PATH = os.path.join(VECTOR_STORE_DIR, "faiss_index.index")
TEXTS_PATH = os.path.join(VECTOR_STORE_DIR, "texts.pkl")
IDS_PATH = os.path.join(VECTOR_STORE_DIR, "ids.pkl")
MODEL_PATH = os.path.join(BASE_DIR, "models", "all-mpnet-base-v2")

# Parámetros
TOP_K = 5

# Cargar modelo y recursos al inicio
# logging.info("📦 Cargando modelo de embeddings...")
# embedding_model = SentenceTransformer(MODEL_PATH)
# logging.info("✅ Modelo cargado.")

logging.info("📂 Cargando FAISS index y metadatos...")
with open(TEXTS_PATH, "rb") as f:
    texts = pickle.load(f)

with open(IDS_PATH, "rb") as f:
    ids = pickle.load(f)

index = faiss.read_index(FAISS_INDEX_PATH)
logging.info(f"✅ Index cargado con {index.ntotal} vectores.")

# Normalización
def normalize_vector(vec):
    return vec / np.linalg.norm(vec, axis=1, keepdims=True)

# Función principal de recuperación
def retrieve_chunks_from_vector(query_vector, top_k):
    logging.info(f"Buscando los top {top_k} chunks con el vector recibido...")
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
        logging.warning("⚠ No se encontraron chunks válidos para la query.")

    return retrieved

def retrieve_chunks_from_query_string(query_string, top_k=TOP_K):
    logging.info("📦 Cargando modelo de embeddings...")
    embedding_model = SentenceTransformer(MODEL_PATH)
    logging.info("✅ Modelo cargado.")
    logging.warning("Esta función vectoriza el query internamente. Usa vectorize_query en producción multiagente.")
    query_vector = embedding_model.encode([query_string]).astype("float32")
    query_vector = query_vector / np.linalg.norm(query_vector, axis=1, keepdims=True)
    return retrieve_chunks_from_vector(query_vector, top_k)

# Prueba desde terminal
if __name__ == "__main__":

    query = "cuando fue la primera guerra mundial?"
    results = retrieve_chunks_from_query_string(query)
    for r in results:
        print(f"\n--- Rank {r['rank']} (Score: {r['score']:.4f}) ---")
        print(f"🆔 ID: {r['id']}")
        print(r['chunk'][:500], "...")
