"""
rag_agent.py

Agente RAG que responde preguntas de historia consultando un índice FAISS previamente creado.
"""

import os
import pickle
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# Configuración de rutas
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VECTOR_STORE_DIR = os.path.join(BASE_DIR, "data", "vectorstore_faiss")
FAISS_INDEX_PATH = os.path.join(VECTOR_STORE_DIR, "faiss_index.index")
TEXTS_PATH = os.path.join(VECTOR_STORE_DIR, "texts.pkl")
IDS_PATH = os.path.join(VECTOR_STORE_DIR, "ids.pkl")

# Modelo de embeddings
MODEL_NAME = 'sentence-transformers/all-MiniLM-L6-v2'

class RAGAgent:
    def __init__(self, top_k=5):
        self.top_k = top_k
        self.model = SentenceTransformer(MODEL_NAME)
        self.index = faiss.read_index(FAISS_INDEX_PATH)

        with open(TEXTS_PATH, "rb") as f:
            self.texts = pickle.load(f)

        with open(IDS_PATH, "rb") as f:
            self.ids = pickle.load(f)

    def query(self, question):
        embedding = self.model.encode([question]).astype("float32")
        distances, indices = self.index.search(embedding, self.top_k)

        results = []
        for i in indices[0]:
            if 0 <= i < len(self.texts):
                results.append(self.texts[i])

        return results


def main():
    agente = RAGAgent(top_k=10)
    pregunta = input("\n🤔 Escribe tu pregunta sobre historia: ")
    resultados = agente.query(pregunta)

    print("\n📚 Resultados relevantes:")
    for i, texto in enumerate(resultados, 1):
        print(f"\n[{i}] {texto[:500]}...")


if __name__ == "__main__":
    main()
