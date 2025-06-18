import logging
from retriever_agent import retrieve_chunks_with_category_rerank, retrieve_chunks_from_vector
from generator_agent import generate_answer, check_context, refine_question, fix_question, wiki_query
from crawler_agent import crawl_single_title
from vectorizer_agent import vectorize_query
from vectorizer_agent import vectorize_chunks
from preprocessor_agent import process_file

# =============================
# Configuración
# =============================
MAX_ATTEMPTS = 3
TOP_K_CHUNKS = 5
CATEGORY_WEIGHT = 0.3

# =============================
# Funciones de estrategia
# =============================

def strategy_basic(query_embedding):
    return retrieve_chunks_from_vector(query_embedding, top_k=TOP_K_CHUNKS)

def strategy_refine(question):
    refined = refine_question(question)
    logging.info(f"🔹 Pregunta refinada: {refined}")
    return refined

def strategy_rerank(query_embedding):
    logging.info("⚡ Aplicando re-ranking con categoría...")
    return retrieve_chunks_with_category_rerank(query_embedding, top_k=TOP_K_CHUNKS, category_weight=CATEGORY_WEIGHT)


def strategy_crawler(question):
    logging.info("⚡ Activando crawler dinámico...")
    query = wiki_query(question)
    article = crawl_single_title(query)
    if not article:
        logging.warning("⚠ El crawler no pudo recuperar un artículo nuevo.")
        return False

    chunks = process_file([article])

    if not chunks:
        logging.warning("⚠ El artículo recuperado no generó chunks procesables.")
        return False

    logging.info(f"✅ {len(chunks)} chunks nuevos obtenidos del crawler. Vectorizando y guardando en FAISS...")
    vectorize_chunks(chunks)  # Esto se encarga de embeddings + FAISS + persistencia

    return True


# =============================
# Estimador simple de tokens
# =============================

def estimate_tokens(chunks):
    return sum(len(chunk.split()) for chunk in chunks)

# =============================
# Tutor principal
# =============================

def tutor_session(question: str):
    question = fix_question(question)
    logging.info(f"\n👤 Usuario: {question}")
    total_tokens_used = 0
    query_embedding = vectorize_query(question)

    attempt = 0
    context_results = strategy_basic(query_embedding)

    while attempt < MAX_ATTEMPTS:
        logging.info(f"🔎 Intento {attempt + 1}: Validando contexto...")
        total_tokens_used += estimate_tokens([c["chunk"] for c in context_results])

        if check_context(question, context_results):
            logging.info("✅ Contexto suficiente. Generando respuesta...")
            answer = generate_answer(question, context_results)
            return {
                "answer": answer,
                "strategy": ["basic", "refine", "rerank", "crawler"][attempt],
                "tokens_used": total_tokens_used
            }

        if attempt == 0:
            # Intentar refinar
            refined_question = strategy_refine(question)
            query_embedding = vectorize_query(refined_question)
            context_results = strategy_basic(query_embedding)

        elif attempt == 1:
            # Intentar re-rankin
            context_results = strategy_rerank(query_embedding)

        elif attempt == 2:
            # Intentar crawler
            if strategy_crawler(question):
                attempt = 0
                logging.info(
                    f"✅ Reintentando con todas las estrategias")


        attempt += 1

    logging.warning("❌ No se pudo generar una respuesta adecuada tras varios intentos.")
    return {
        "answer": "Lo siento, no pude generar una respuesta adecuada tras varios intentos.",
        "strategy": "failed",
        "tokens_used": total_tokens_used
    }
# =============================
# Ejecución directa (para pruebas)
# =============================
if __name__ == "__main__":
    result = tutor_session("cuando fue la primera guerra mundial?")
    logging.info(f"\n📝 Respuesta generada:\n{result['answer']}")
    logging.info(f"🔹 Estrategia usada: {result['strategy']}")
    logging.info(f"🔹 Tokens usados: {result['tokens_used']}")
