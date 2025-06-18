import logging
from src.agents.retriever_agent import retrieve_chunks_with_category_rerank, retrieve_chunks_from_vector
from src.agents.generator_agent import generate_answer, check_context, refine_question, fix_question, wiki_query
from src.agents.crawler_agent import crawl_single_title, crawl_titles
from src.agents.vectorizer_agent import vectorize_query, vectorize_chunks, vectorize
from src.agents.preprocessor_agent import process_file, preprocess
from src.config import LOG_FILES, CATEGORY_WEIGHT
import time
import os

# Configuración de logs
LOG_FILE = LOG_FILES["tutor"]

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
    {"name": "basic_top10", "top_k": 10, "refine": False, "rerank": False},
    {"name": "rerank_top5", "top_k": 5, "refine": False, "rerank": True},
    {"name": "rerank_top10", "top_k": 10, "refine": False, "rerank": True},
    {"name": "refine_basic_top5", "top_k": 5, "refine": True, "rerank": False},
    {"name": "refine_basic_top10", "top_k": 10, "refine": True, "rerank": False},
    {"name": "refine_rerank_top5", "top_k": 5, "refine": True, "rerank": True},
    {"name": "refine_rerank_top10", "top_k": 10, "refine": True, "rerank": True},
    {"name": "crawler", "crawler": True}
]


def estimate_tokens(chunks):
    """
    Calcula el número total de tokens de una lista de chunks.

    Args:
        chunks (list): Lista de diccionarios con los chunks.

    Returns:
        int: Número total de tokens (palabras).
    """
    return sum(len(chunk["chunk"].split()) for chunk in chunks)


def tutor_session(question: str):
    """
    Ejecuta una sesión de tutoría para responder una pregunta.

    Aplica estrategias secuenciales de recuperación y generación.

    Args:
        question (str): Pregunta del usuario.

    Returns:
        dict: Resultado con respuesta, estrategia usada y tokens consumidos.
    """
    question = fix_question(question)
    time.sleep(20)
    logging.info(f"👤 Usuario preguntó: '{question}'")
    total_tokens_used = 0
    query_embedding = vectorize_query(question)

    for attempt, strat in enumerate(STRATEGY_SEQUENCE):
        logging.info(f"🔎 Intento {attempt + 1}: Estrategia = {strat['name']}")
        q = question
        qe = query_embedding

        if strat.get("refine"):
            q = strategy_refine(q)
            qe = vectorize_query(q)

        if strat.get("crawler"):
            logging.info("🌐 Activando crawler dinámico...")
            if strategy_crawler(q):
                logging.info("✅ Crawler enriqueció el corpus. Reiniciando ciclo de búsqueda...")
                query_embedding = vectorize_query(question)
                continue
            else:
                logging.warning("❌ El crawler no pudo recuperar un artículo nuevo.")
                break

        if strat.get("rerank"):
            context_results = retrieve_chunks_with_category_rerank(
                qe,
                top_k=strat["top_k"],
                category_weight=CATEGORY_WEIGHT
            )
        else:
            context_results = retrieve_chunks_from_vector(
                qe,
                top_k=strat["top_k"]
            )

        total_tokens_used += estimate_tokens(context_results)

        if check_context(q, context_results):
            logging.info("✅ Contexto suficiente. Generando respuesta...")
            answer = generate_answer(q, context_results)
            return {
                "answer": answer,
                "strategy": strat["name"],
                "tokens_used": total_tokens_used
            }
        else:
            logging.info("⚠️ Contexto insuficiente. Probando otra estrategia...")

    logging.warning("❌ No se pudo generar una respuesta adecuada tras varios intentos.")
    return {
        "answer": "Lo siento, no pude generar una respuesta adecuada tras varios intentos.",
        "strategy": "failed",
        "tokens_used": total_tokens_used
    }


def strategy_refine(question):
    """
    Refina una pregunta usando el generador.

    Args:
        question (str): Pregunta original.

    Returns:
        str: Pregunta refinada.
    """
    return refine_question(question)


def strategy_crawler(question):
    """
    Ejecuta el crawler para enriquecer el corpus con un nuevo artículo.

    Args:
        question (str): Pregunta del usuario.

    Returns:
        bool: True si el corpus se enriqueció; False si no.
    """
    query = wiki_query(question)
    article = crawl_single_title(query)
    if not article:
        return False
    chunks = process_file([article])
    if not chunks:
        return False
    logging.info(f"✅ Crawler añadió {len(chunks)} chunks nuevos.")
    vectorize_chunks(chunks)
    return True


def tutor_loop():
    """
    Inicia el bucle interactivo del tutor, esperando preguntas del usuario.
    """
    logging.info("💬 Bienvenido al Tutor de Historia. Escribe tu pregunta o 'exit' para salir.")
    while True:
        user_input = input("👤 Tu pregunta: ").strip()
        if user_input.lower() == "exit":
            logging.info("👋 Sesión finalizada por el usuario.")
            break
        if not user_input:
            logging.warning("⚠️ Entrada vacía. Por favor escribe una pregunta o 'exit'.")
            continue

        result = tutor_session(user_input)
        logging.info(f"\n📝 Respuesta:\n{result['answer']}")
        logging.info(f"🔹 Estrategia usada: {result['strategy']}")
        logging.info(f"🔹 Tokens usados: {result['tokens_used']}")

def menu_inicial():
    while True:
        print("\nSelecciona una opción:")
        print("1. Ejecutar crawler dinámico (usar archivo de títulos por defecto)")
        print("2. Ejecutar postprocesador (usar archivo raw por defecto)")
        print("3. Ejecutar vectorización (usar archivo procesado por defecto)")
        print("4. Iniciar tutor/chat")
        print("5. Salir")

        opcion = input("Opción: ").strip()

        if opcion == "1":
            print("🚀 Ejecutando crawler dinámico...")
            crawl_titles()
        elif opcion == "2":
            print("🚀 Ejecutando postprocesador...")
            preprocess()
        elif opcion == "3":
            print("🚀 Ejecutando vectorización...")
            vectorize()
        elif opcion == "4":
            tutor_loop()
            break
        elif opcion == "5":
            print("👋 Saliendo del sistema.")
            break
        else:
            print("❌ Opción no válida, intenta de nuevo.")

if __name__ == "__main__":
    menu_inicial()
