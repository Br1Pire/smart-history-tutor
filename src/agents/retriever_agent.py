import numpy as np
import logging
from src.core.faiss_manager import FaissManager
from src.core.document_manager import DocumentManager
from src.agents.generator_agent import Generator
from src.agents.vectorizer_agent import Vectorizer
from src.agents.crawler_agent import Crawler
from src.agents.preprocessor_agent import Preprocessor
from src.config import LOG_FILES, TOP_K_CHUNKS, CATEGORY_WEIGHT

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

STRATEGY_SEQUENCE = [
    {"name": "basic_top5", "top_k": 5, "refine": False, "rerank": False},
    {"name": "rerank_top5", "top_k": 5, "refine": False, "rerank": True},
    {"name": "basic_top10", "top_k": 10, "refine": False, "rerank": False},
    {"name": "rerank_top10", "top_k": 10, "refine": False, "rerank": True},
    {"name": "crawler", "crawler": True}
]


class Retriever:
    """
    Agente de recuperación de información que utiliza FAISS y embeddings para recuperar chunks relevantes.
    """

    def __init__(self, generator : Generator, vectorizer : Vectorizer, crawler: Crawler, preprocessor: Preprocessor, document_manager: DocumentManager, text_manager: FaissManager, category_manager: FaissManager = None  ):
        self.text_manager = text_manager
        self.category_manager = category_manager
        self.crawler = crawler
        self.preprocessor = preprocessor
        self.document_manager = document_manager
        self.generator = generator
        self.vectorizer = vectorizer

    @staticmethod
    def estimate_tokens( chunks):
        """
        Calcula el número total de tokens de una lista de chunks.

        Args:
            chunks (list): Lista de diccionarios con los chunks.

        Returns:
            int: Número total de tokens (palabras).
        """
        return sum(len(chunk["chunk"].split()) for chunk in chunks)
    
    def retrieve_chunks_from_vector(self, query_vector, top_k = TOP_K_CHUNKS, combined=False):
        """
        Recupera los chunks más relevantes para un vector de query.

        Args:
            query_vector (np.ndarray): Vector de la consulta.
            top_k (int): Número de resultados a devolver.

        Returns:
            list: Lista de diccionarios con el índice FAISS, la puntuación de similitud de coseno y el rank.
                  Esta lista NO contiene el contenido completo del chunk aún.
        """
        logging.info(f"🔎 Buscando top {top_k} chunks por vector...")
        distances, indices = self.text_manager.search(query_vector, top_k)
        retrieved_raw = [] 
 
        if distances.ndim == 2 and distances.shape[0] == 1:
            distances_flat = distances[0]
            indices_flat = indices[0]
        else: 
            distances_flat = distances
            indices_flat = indices
        
        for i, idx_faiss in enumerate(indices_flat):
            if idx_faiss == -1: 
                continue
            
            if idx_faiss < len(self.document_manager.processed_documents):
                chunk_id = self.document_manager.processed_documents[idx_faiss].get('id', 'N/A')
            else:
                chunk_id = f"INVALID_INDEX_{idx_faiss}"
                logging.warning(f"⚠️ Índice FAISS {idx_faiss} fuera de rango para chunks. Saltando.")
                continue

            retrieved_raw.append({
                "index": int(idx_faiss), 
                "score": float(distances_flat[i]), 
                "rank": i + 1
            })
           
            logging.info(f"🔹Chunk {chunk_id} (FAISS_idx={idx_faiss}) recuperado con score {distances_flat[i]:.4f}.")

        logging.info(f"✅ Recuperados {len(retrieved_raw)} chunks.")
        if not retrieved_raw:
            logging.warning("⚠️ No se encontraron chunks válidos.")
        
        if not combined:
            final_results = []
            for item in retrieved_raw:
            
                faiss_pos = item["index"]
                if 0 <= faiss_pos < len(self.document_manager.processed_documents):
                    chunk_data = self.document_manager.processed_documents[faiss_pos]
                    original_id = chunk_data.get('id', f"FAISS_ID_{faiss_pos}")
                    chunk_content = chunk_data.get('content', '') 
                else:
                    logging.warning(f"⚠️ Índice FAISS {faiss_pos} fuera de rango para self.chunks. Se omite este resultado final.")
                    continue


                final_results.append({
                    "id": original_id, 
                    "chunk": chunk_content, 
                    "score": item["score"],
                    "rank": item["rank"]
                })
        
            return final_results
           
        return retrieved_raw


    def retrieve_chunks_with_category_rerank(self, query_vector, top_k=TOP_K_CHUNKS, category_weight=0.3):
        """
        Recupera chunks y reranquea considerando categorías.

        Args:
            query_vector (np.ndarray): Vector de la consulta.
            top_k (int): Número de resultados finales.
            category_weight (float): Peso de la categoría en la puntuación combinada (0.0 a 1.0).

        Returns:
            list: Lista de diccionarios con los chunks reranqueados (id, chunk_content, scores, rank).
        """
        
        if self.category_manager is None:
            logging.warning("⚠️ FaissManager de categorías no proporcionado. El reranking por categoría no se realizará. Devolviendo resultados solo por similitud de texto.")
           
            raw_results = self.retrieve_chunks_from_vector(query_vector, top_k)
            return raw_results


        logging.info(f"🔎 Buscando top {top_k * 2} chunks iniciales para reranking...")
    
        retrieved_candidates_meta = self.retrieve_chunks_from_vector(query_vector, top_k * 2, combined=True)

        if not retrieved_candidates_meta:
            logging.warning("⚠️ No se encontraron chunks iniciales para reranking.")
            return []

        if query_vector.ndim == 2 and query_vector.shape[0] == 1:
            query_vector_1d = query_vector[0]
        else:
            query_vector_1d = query_vector 

        combined_results_temp = []
        logging.info(f"🔄 Rerankeando {len(retrieved_candidates_meta)} chunks con categoría...")

        for doc_meta in retrieved_candidates_meta:
            faiss_pos = doc_meta["index"]
            text_similarity_score = doc_meta["score"] 

            cat_similarity_score = 0.0 

            try:
                cat_emb = self.category_manager.reconstruct(faiss_pos)
                
                if cat_emb.ndim == 2 and cat_emb.shape[0] == 1:
                    cat_emb = cat_emb[0]
                
                cat_similarity_score = np.dot(cat_emb, query_vector_1d) 
            except Exception as e:
                logging.warning(f"⚠️ No se pudo reconstruir o calcular similitud de categoría para el índice {faiss_pos}: {e}. La categoría tendrá peso 0.")
                    
            category_weight = np.clip(category_weight, 0.0, 1.0)
            
            combined_score = (1 - category_weight) * text_similarity_score + category_weight * cat_similarity_score
            
            combined_results_temp.append({
                "index": faiss_pos, 
                "combined_score": combined_score,
                "text_similarity_score": text_similarity_score,
                "category_similarity_score": cat_similarity_score, 
            })
            
            chunk_id = self.document_manager.processed_documents[faiss_pos].get('id', 'N/A') if 0 <= faiss_pos < len(self.document_manager.processed_documents) else "INVALID"
            logging.info(f"🔹Chunk {chunk_id} (FAISS_idx={faiss_pos}) combinado con score {combined_score:.4f}.")


        combined_results_temp.sort(key=lambda x: x["combined_score"], reverse=True)

        final_results = []
        for rank, item in enumerate(combined_results_temp[:top_k], start=1):
            
            faiss_pos = item["index"]
            if 0 <= faiss_pos < len(self.document_manager.processed_documents):
                chunk_data = self.document_manager.processed_documents[faiss_pos]
                original_id = chunk_data.get('id', f"FAISS_ID_{faiss_pos}")
                chunk_content = chunk_data.get('content', '') 
            else:
                logging.warning(f"⚠️ Índice FAISS {faiss_pos} fuera de rango para self.chunks. Se omite este resultado final.")
                continue


            final_results.append({
                "id": original_id, 
                "chunk": chunk_content, 
                "combined_score": item["combined_score"],
                "text_similarity_score": item["text_similarity_score"],
                "category_similarity_score": item["category_similarity_score"], 
                "rank": rank
            })
        
        logging.info(f"✅ Rerankeados y finalizados {len(final_results)} chunks.")
        return final_results
    
    def strategic_retrieve(self, question: str):
        """
        Ejecuta una sesión de tutoría para responder una pregunta.

        Aplica estrategias secuenciales de recuperación y generación.

        Args:
            question (str): Pregunta del usuario.

        Returns:
            dict: Resultado con respuesta, estrategia usada y tokens consumidos.
        """
        original_question = question
        question = self.generator.fix_question(question)
        total_tokens_used = 0
        
        attempt_round = 0

        while True: 
            corpus_enriched = False 
            query_embedding = self.vectorizer.vectorize_query(question) 
            
            for attempt, strat in enumerate(STRATEGY_SEQUENCE):
                logging.info(f"🔎 Intento {attempt + attempt_round + 1}: Estrategia = {strat['name']}")
                q = question
                qe = query_embedding

                if strat.get("refine"):
                    q = self.generator.refine_question(q)
                    qe = self.vectorizer.vectorize_query(q)

                if strat.get("crawler"):
                    logging.info("🌐 Activando crawler dinámico...")
                    if self.strategy_crawler(q):
                        logging.info("✅ Crawler enriqueció el corpus. Reiniciando ciclo de búsqueda...")
                        corpus_enriched = True
                        attempt_round += 1 
                        
                        break 
                    else:
                        logging.warning("❌ El crawler no pudo recuperar un artículo nuevo.")
                        break 

                if not strat.get("crawler"):
                    if strat.get("rerank"):
                        context_results = self.retrieve_chunks_with_category_rerank(
                            qe,
                            top_k=strat["top_k"],
                            category_weight=CATEGORY_WEIGHT
                        )
                    else:
                        context_results = self.retrieve_chunks_from_vector(
                            qe,
                            top_k=strat["top_k"]
                        )

                    total_tokens_used += self.estimate_tokens(context_results)

                    if self.generator.check_context(q, context_results):
                        logging.info("✅ Contexto suficiente. Generando respuesta...")
                        answer = self.generator.generate_answer(q, context_results)
                        return {
                            "answer": answer,
                            "strategy": strat["name"],
                            "tokens_used": total_tokens_used
                        }
                    else:
                        logging.info("⚠️ Contexto insuficiente. Probando otra estrategia...")
            
            if not corpus_enriched:
                break 

        logging.warning("❌ No se pudo generar una respuesta adecuada tras varios intentos.")
        return {
            "answer": "Lo siento, no pude generar una respuesta adecuada tras varios intentos.",
            "strategy": "failed",
            "tokens_used": total_tokens_used
        }
    
    def strategy_crawler(self,question):
        """
        Ejecuta el crawler para enriquecer el corpus con un nuevo artículo.

        Args:
            question (str): Pregunta del usuario.

        Returns:
            bool: True si el corpus se enriqueció; False si no.
        """
        query = self.generator.wiki_query(question)
        article = self.crawler.crawl_single_title(query)
        if not article:
            return False
        chunks = self.preprocessor.preprocess()
        if not chunks:
            return False
        logging.info(f"✅ Crawler añadió {len(chunks)} chunks nuevos.")
        self.vectorizer.vectorize()
        return True