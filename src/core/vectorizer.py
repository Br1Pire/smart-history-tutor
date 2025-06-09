import os
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import pickle

# Configuración
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "all-mpnet-base-v2")
INPUT_PATH = os.path.join(BASE_DIR, "data", "processed", "wiki_articles_segmented.json")

VECTOR_STORE_DIR = os.path.join(BASE_DIR, "data", "vectorstore_faiss")
FAISS_INDEX_PATH = os.path.join(VECTOR_STORE_DIR, "faiss_index.index")
TEXTS_PATH = os.path.join(VECTOR_STORE_DIR, "texts.pkl")
IDS_PATH = os.path.join(VECTOR_STORE_DIR, "ids.pkl")

def load_documents(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def clean_text(text):
    return text.replace("\n", " ").strip()

def create_embeddings(model, documents):
    texts = [clean_text(f"{doc['title']}. {doc['content']}") for doc in documents]
    ids = [f"{doc['title']}__chunk_{doc['chunk_index']}" for doc in documents]
    embeddings = model.encode(texts, show_progress_bar=True, batch_size=32)
    embeddings = np.array(embeddings).astype("float32")
    embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)  # NORMALIZACIÓN
    return ids, texts, embeddings

def save_to_faiss(ids, texts, embeddings):
    os.makedirs(VECTOR_STORE_DIR, exist_ok=True)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # USAMOS PRODUCTO INTERNO = COSENO
    index.add(embeddings)

    faiss.write_index(index, FAISS_INDEX_PATH)

    with open(TEXTS_PATH, "wb") as f:
        pickle.dump(texts, f)

    with open(IDS_PATH, "wb") as f:
        pickle.dump(ids, f)

    print(f"\n✅ Guardados {len(ids)} embeddings en el índice FAISS.")

def main():
    print("📦 Cargando modelo localmente...")
    model = SentenceTransformer(MODEL_PATH)

    print("📚 Cargando documentos...")
    documents = load_documents(INPUT_PATH)

    print("🧠 Generando embeddings...")
    ids, texts, embeddings = create_embeddings(model, documents)

    print("💾 Guardando en el índice FAISS...")
    save_to_faiss(ids, texts, embeddings)

if __name__ == "__main__":
    main()
