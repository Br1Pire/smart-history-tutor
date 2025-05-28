from sentence_transformers import SentenceTransformer
import chromadb
import os

CHROMA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "vector_store")
COLLECTION_NAME = "historia_universal"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

class RAGAgent:
    def __init__(self, top_k=5):
        self.vectorizer = SentenceTransformer(MODEL_NAME)
        self.client = chromadb.Client(
            settings=chromadb.Settings(
                persist_directory=CHROMA_DIR,
                anonymized_telemetry=False
            )
        )
        self.collection = self.client.get_or_create_collection(name=COLLECTION_NAME)
        self.top_k = top_k

    def query(self, pregunta):
        embedding = self.vectorizer.encode(pregunta).tolist()
        resultados = self.collection.query(
            query_embeddings=[embedding],
            n_results=self.top_k,
            include=["documents"]
        )
        return resultados["documents"][0]

# Modo de prueba
if __name__ == "__main__":
    agente = RAGAgent()
    pregunta = input("🤔 Escribe tu pregunta sobre historia: ")
    resultados = agente.query(pregunta)
    print("\n📚 Resultados relevantes:")
    for i, doc in enumerate(resultados):
        print(f"\n[{i+1}] {doc[:400]}...")
