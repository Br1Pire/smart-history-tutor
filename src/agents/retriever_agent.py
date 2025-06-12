import os
import pickle
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

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
embedding_model = SentenceTransformer(MODEL_PATH)

with open(TEXTS_PATH, "rb") as f:
    texts = pickle.load(f)

with open(IDS_PATH, "rb") as f:
    ids = pickle.load(f)

index = faiss.read_index(FAISS_INDEX_PATH)

# Normalización
def normalize_vector(vec):
    return vec / np.linalg.norm(vec, axis=1, keepdims=True)

# Función principal de recuperación
def retrieve_chunks(query, top_k=TOP_K):
    print(f"🔎 Retrieving top {top_k} chunks for query: '{query}'")

    query_vector = embedding_model.encode([query]).astype("float32")
    query_vector = normalize_vector(query_vector)
    distances, indices = index.search(query_vector, top_k)

    retrieved = []
    for i, idx in enumerate(indices[0]):
        retrieved.append({
            "id": ids[idx],
            "chunk": texts[idx],
            "score": float(distances[0][i]),
            "rank": i + 1
        })

    return retrieved

# Prueba desde terminal
if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
        results = retrieve_chunks(query)
        for r in results:
            print(f"\n--- Rank {r['rank']} (Score: {r['score']:.4f}) ---")
            print(f"🆔 ID: {r['id']}")
            print(r['chunk'][:500], "...")
    else:
        print("ℹ️ Usage: python retriever_agent.py 'your historical question'")
