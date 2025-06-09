import os
import sys
import pickle
import faiss
import numpy as np
from transformers import pipeline
from sentence_transformers import SentenceTransformer


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VECTOR_STORE_DIR = os.path.join(BASE_DIR, "data", "vectorstore_faiss")
FAISS_INDEX_PATH = os.path.join(VECTOR_STORE_DIR, "faiss_index.index")
TEXTS_PATH = os.path.join(VECTOR_STORE_DIR, "texts.pkl")

EMBEDDING_MODEL_PATH = os.path.join(BASE_DIR, "models", "all-mpnet-base-v2")
QA_MODEL_PATH = os.path.join(BASE_DIR, "models", "bert-base-spanish-wwm-cased-finetuned-spa-squad2-es")
TOP_K = 5

# Cargar índice y textos
print("📂 Cargando índice vectorial y textos...")
index = faiss.read_index(FAISS_INDEX_PATH)
with open(TEXTS_PATH, "rb") as f:
    texts = pickle.load(f)

# Cargar modelos
print("⚙️ Cargando modelos...")
embedding_model = SentenceTransformer(EMBEDDING_MODEL_PATH)
qa_pipeline = pipeline("question-answering", model=QA_MODEL_PATH, tokenizer=QA_MODEL_PATH)
print("✅ Modelos listos.\n")

# Buscar contexto relevante
def retrieve_context(query, k=TOP_K):
    query_vector = embedding_model.encode([query]).astype("float32")
    query_vector /= np.linalg.norm(query_vector, axis=1, keepdims=True)
    distances, indices = index.search(query_vector, k * 2)  # Recupera más para filtrar

    top_chunks = [texts[i] for i in indices[0]]

    # Priorizamos los chunks cuyo título coincide con palabras clave en la pregunta
    filtered = [t for t in top_chunks if any(w in t.lower() for w in query.lower().split())]

    # Si encontramos al menos k relevantes, usamos esos
    if len(filtered) >= k:
        return filtered[:k]

    # Si no, devolvemos los primeros k originales

    print("\n🔎 Resultados FAISS (con distancias):")
    for i, (idx, dist) in enumerate(zip(indices[0], distances[0])):
        print(f"\n--- Chunk {i + 1} (distancia coseno: {dist:.4f}) ---")
        print(texts[idx][:400].strip() + "...\n")

    return top_chunks[:k]


# Responder pregunta
def answer_question(question, context):
    joined_context = "\n".join(context)
    result = qa_pipeline(question=question, context=joined_context)
    return result

# Loop interactivo
while True:
    print("🤔 Escribe tu pregunta sobre historia:")
    question = input(" > ").strip()
    if not question:
        break

    context = retrieve_context(question)
    print("🔎 Contexto recuperado:")
    for i, c in enumerate(context, 1):
        print(f"\n--- Documento {i} ---\n{c[:600]}...\n")

    answer = answer_question(question, context)
    print("\n🧠 Respuesta:")
    print(f"{answer['answer']} (confianza: {answer['score']:.2f})\n")
