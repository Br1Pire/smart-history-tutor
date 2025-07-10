import os
import faiss
import logging
import numpy as np
from src.config import LOG_FILES

# Configuración de logs
LOG_FILE = LOG_FILES["faiss_manager"]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, mode='a', encoding='utf-8'),
        logging.StreamHandler()
    ]
)

class FaissManager:
    def __init__(self, index_path, dim=768):
        self.index_path = index_path
        self.dim = dim

        # Cargar index si existen, sino crear nuevos
        if os.path.exists(index_path):
            self.index = faiss.read_index(str(index_path))
            print(f"✅ Index cargados.")
        else:
            self.index = faiss.IndexFlatIP(dim)
            print("🔧 Index nuevo creado.")

    def add(self, vectors):
        """
        Añade vectores al índice.
        :param vectors: np.ndarray de forma (n, dim)
        """
        if not self.check_dim(vectors):
            raise ValueError(f"Dimensiones incorrectas: se esperaba {self.dim}, pero se recibió {vectors.shape[1]}.")

        self.index.add(vectors)
        
        print(f"✅ Añadidos textos al índice.")
        self.save()

    def add_unique(self, vectors, threshold=0.999):
        """
        Añade vectores si no existen ya en el índice.
        :param vectors: np.ndarray (n, dim)
        :param threshold: umbral de similitud para considerar duplicado
        """
        if not self.check_dim(vectors):
            raise ValueError(f"Dimensiones incorrectas: se esperaba {self.dim}, pero se recibió {vectors.shape[1]}.")

        new_vectors = []
        for i in range(vectors.shape[0]):
            vector = vectors[i:i+1]
            if not self.vector_exists(vector, threshold):
                new_vectors.append(vector)
            else:
                print(f"⚠️ Vector {i} ya existe. No se añade.")

        if new_vectors:
            batch = np.vstack(new_vectors)
            self.index.add(batch)
            print(f"✅ Añadidos {batch.shape[0]} vectores nuevos.")
        else:
            print("✅ Ningún vector nuevo añadido (todos duplicados).")

    def vector_exists(self, vector, threshold=0.999):
        """
        Comprueba si un vector ya está en el índice.
        Para IndexFlatIP (dot product), se espera que los vectores estén normalizados si se interpreta como coseno.
        :param vector: np.ndarray shape (1, dim)
        :param threshold: umbral de similitud (ej: 0.999 ~ igual)
        :return: True si existe, False si no.
        """
        if self.index.ntotal == 0:
            return False

        distances, indices = self.index.search(vector, 1)
        similarity = distances[0][0]
        return similarity >= threshold

    def search(self, query_vector, top_k=5):
        """
        Realiza una búsqueda con un vector ya generado y devuelve (distancias, indices).
        :param query_vector: np.ndarray de forma (1, dim)
        :return: tuple (distances, indices)
        """
        distances, indices = self.index.search(query_vector, top_k)
        return distances, indices
    
    def check_dim(self, vectors):
        return vectors.shape[1] == self.dim
    
    def reconstruct(self, idx):
        """
        Reconstruye un vector del índice por su índice.
        :param idx: índice del vector a reconstruir
        :return: np.ndarray del vector reconstruido
        """
        return self.index.reconstruct(idx)

    def save(self):
        """
        Guarda el índice y los textos asociados.
        """
        faiss.write_index(self.index, str(self.index_path))
        print(f"💾 Index guardados en {self.index_path}.")

    def clean_index(self):
        self.index = faiss.IndexFlatIP(self.dim)
        print('✅Indice reseteado')
        
