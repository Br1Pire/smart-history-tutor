import os
import json
import numpy as np
import logging
from sentence_transformers import SentenceTransformer
from src.core.document_manager import DocumentManager
from src.core.faiss_manager import FaissManager
from src.config import LOG_FILES, MODEL_PATH

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

logging.info("🚀 Cargando modelo de embeddings...")
# Asegúrate de que MODEL_PATH sea una cadena de texto para os.path.exists
MODEL = SentenceTransformer(str(MODEL_PATH) if os.path.exists(str(MODEL_PATH)) else "all-mpnet-base-v2")
logging.info("✅ Modelo cargado correctamente.")

class Vectorizer:
    def __init__(self, document_manager: DocumentManager , faiss_manager: FaissManager, category_manager: FaissManager):
        self.document_manager = document_manager
        self.faiss_manager = faiss_manager
        self.category_manager = category_manager
        
        logging.info("Vectorizer inicializado.")


    def _generate_embeddings(self, documents):
        """
        Genera embeddings para una lista dada de documentos.
        """
        text_inputs = []
        category_inputs = []

        for doc in documents:
            header_parts = []
            if doc.get('title',""):
                header_parts.append(f"Título: {doc['title']}")
            if doc.get('section',""):
                header_parts.append(f"Sección: {doc['section']}")
            header_parts.append(doc['content'])

            composed_text = "\n".join(header_parts).strip()
            text_inputs.append(composed_text)

            categories = doc.get('categories', [])
            categories_text = " ".join(categories).strip()
            category_inputs.append(categories_text)

        logging.info(f"⚡ Generando embeddings de texto ({len(text_inputs)} entradas)...")
        text_embeddings = MODEL.encode(text_inputs, batch_size=32, show_progress_bar=True)
        text_embeddings = text_embeddings / np.linalg.norm(text_embeddings, axis=1, keepdims=True)
        logging.info("✅ Embeddings de texto generados y normalizados.")

        logging.info(f"⚡ Generando embeddings de categorías ({len(category_inputs)} entradas)...")
        category_embeddings = MODEL.encode(category_inputs, batch_size=32, show_progress_bar=True)
        category_embeddings = category_embeddings / np.linalg.norm(category_embeddings, axis=1, keepdims=True)
        logging.info("✅ Embeddings de categorías generados y normalizados.")

        return text_embeddings, category_embeddings

    @staticmethod
    def vectorize_query(query):
        """
        Vectoriza una consulta de texto.
        """
        embedding = MODEL.encode([query]).astype("float32")
        embedding = embedding / np.linalg.norm(embedding, axis=1, keepdims=True)
        logging.info("✅ Vector del query generado y normalizado.")
        return embedding

    def vectorize(self):
        """
        Vectoriza un conjunto de chunks.
        Si chunks es None, utiliza los chunks cargados durante la inicialización (self.chunks).
        Si esos embeddings no se han generado aún, los genera y los almacena.
        De lo contrario, vectoriza la lista de 'chunks' proporcionada.
        """
        new_documents = self.document_manager.get_new_chunks()

        if not new_documents: 
            logging.info("✅ No hay nuevos documentos (chunks) para vectorizar.")
            return

        text_embeddings, category_embeddings = self._generate_embeddings(new_documents)

        self.faiss_manager.add(text_embeddings)
        self.category_manager.add(category_embeddings)

        new_ids = [a['id'] for a in new_documents]

        self.document_manager.add_ids(new_ids)


