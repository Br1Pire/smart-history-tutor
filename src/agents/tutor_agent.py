import logging
import os
from retriever_agent import retrieve_chunks_from_vector
from generator_agent import generate_answer
from generator_agent import check_context
from generator_agent import refine_question
from generator_agent import fix_question
from crawler_agent import crawl_single_title
from vectorizer_agent import vectorize_chunks
from vectorizer_agent import vectorize_query
from preprocessor_agent import process_article

# Configuración de logging (asegúrate de que sea consistente con otros agentes)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Parámetros para el bucle de mejora de contexto
MAX_CONTEXT_ATTEMPTS = 3  # Número máximo de intentos para obtener buen contexto
TOP_K_CHUNKS = 5  # Cuántos chunks iniciales recuperar


def tutor_session(question: str, auto_crawl: bool = True) -> str:
    global TOP_K_CHUNKS
    print(f"\n👤 Usuario: {question}")
    current_question = fix_question(question)
    context_sufficient = False
    context_results = []

    # Lista para almacenar los resultados de cada intento para mostrarlos al final
    all_context_attempts_info = []

    for attempt in range(MAX_CONTEXT_ATTEMPTS):
        logging.info(f"✨ Intento de recuperación de contexto: {attempt + 1}/{MAX_CONTEXT_ATTEMPTS}")

        # 1. Vectorizar la pregunta actual
        vectorized_question = vectorize_query(current_question)

        # 2. Recuperar contexto con la pregunta vectorizada
        context_results = retrieve_chunks_from_vector(vectorized_question, top_k=TOP_K_CHUNKS)
        combined_context = " ".join([r['chunk'] for r in context_results])

        print(f"\n📚 Contexto recuperado (longitud total: {len(combined_context)} caracteres) [Intento {attempt + 1}]")
        for i, r in enumerate(context_results):
            print(f"🔹 Chunk {i + 1} | Score: {r['score']:.4f}")
            # El split "__" está bien, pero si el id no tiene sección, el split[1] podría fallar.
            # Mejorar el manejo de ID si la sección puede ser nula.
            id_parts = r.get('id', 'Desconocido').split("__")
            title_part = id_parts[0] if len(id_parts) > 0 else 'Desconocido'
            section_part = id_parts[1] if len(id_parts) > 1 else 'General'  # 'General' como fallback
            print(f"   📘 Título: {title_part} | Sección: {section_part}")
            print(f"   📄 Contenido: {r['chunk']}\n")

        # Almacenar info del intento actual para depuración/logging
        all_context_attempts_info.append({
            "attempt": attempt + 1,
            "question_used": current_question,
            "context_length": len(combined_context),
            "chunks_count": len(context_results),
            "top_chunks_scores": [r['score'] for r in context_results]
        })

        # 3. Chequear si el contexto es suficiente para responder a la pregunta original
        # Aquí es crucial pasar la PREGUNTA ORIGINAL al check_context, no la refinada,
        # para saber si el contexto sirve para la necesidad inicial del usuario.
        context_sufficient = check_context(question, [r['chunk'] for r in context_results])

        if context_sufficient:
            logging.info("✅ Contexto actual suficiente para responder a la pregunta.")
            break  # Salir del bucle, tenemos suficiente contexto
        else:
            logging.warning("⚠️ Contexto insuficiente. Intentando mejorar la situación...")

            if attempt == 0:
                # Primer intento fallido: Refinar la pregunta
                logging.info("🧠 Refinando la pregunta para una mejor recuperación...")
                current_question = question# Usa la pregunta ORIGINAL para refinar
                TOP_K_CHUNKS = 10

            elif attempt == 1 and auto_crawl:
                # Segundo intento fallido: Usar fix_question (si lo consideras útil) o ir directo al crawler
                logging.info(
                    "📝 La pregunta refinada no mejoró el contexto. Intentando arreglar pregunta (o ir directo a crawl)...")
                # Puedes elegir entre fix_question o directamente pasar a crawl.
                # Si fix_question es muy similar a refine_question, podrías saltártelo.
                current_question = fix_question(question)  # Usa la pregunta ORIGINAL para arreglar

                # Si aún no es suficiente después de refinar/arreglar, y el auto_crawl está activado
                logging.info("🌐 Contexto sigue siendo insuficiente. Activando Agente Crawler...")
                new_article_data = crawl_single_title(current_question)  # Crawler busca usando la pregunta mejorada

                if new_article_data:
                    print(f"✅ Artículo encontrado por el Crawler: {new_article_data['title']}")
                    logging.info(
                        f"🧠 Procesando y vectorizando el nuevo contenido del artículo '{new_article_data['title']}'...")

                    # Procesa el artículo (limpia, segmenta, extrae entidades)
                    processed_chunks = process_article(new_article_data)

                    # Vectoriza los nuevos chunks y los añade al índice FAISS
                    vectorize_chunks(processed_chunks, persist=True)  # Asegura persist=True para que se guarden
                    logging.info("Nuevo contenido procesado y añadido al índice FAISS.")

                    # Para el siguiente intento, la pregunta sigue siendo la misma (mejorada si se aplicó)
                    # y el retriever ahora tendrá más datos.
                else:
                    logging.warning("❌ El Crawler no encontró un nuevo artículo relevante o ya existía.")
            else:
                # Último intento o auto_crawl desactivado, no hay más acciones de mejora
                logging.warning("❌ No se pudo mejorar el contexto tras múltiples intentos o auto_crawl desactivado.")
                break  # Salir del bucle, no podemos hacer más

    # Si después de todos los intentos, el contexto aún no es suficiente, el generador debe indicarlo.
    # El prompt del generator_agent ya maneja esto: "Si no encuentras la información en los fragmentos,
    # reponde que no puedes responder a la pregunta con el context proporcionado."

    # Paso Final: Generar respuesta
    chunks_text = [r['chunk'] for r in context_results]
    response = generate_answer(question, chunks_text)

    # Paso 4: Devolver al usuario
    print("\n🤖 Tutor:")
    print(response)

    # Imprimir resumen de intentos de contexto
    print("\n--- Resumen de Intentos de Contexto ---")
    for attempt_info in all_context_attempts_info:
        print(f"Intento {attempt_info['attempt']}:")
        print(f"  Pregunta usada: '{attempt_info['question_used']}'")
        print(f"  Longitud contexto: {attempt_info['context_length']} chars, Chunks: {attempt_info['chunks_count']}")
        print(f"  Scores: {[f'{s:.2f}' for s in attempt_info['top_chunks_scores']]}")
    print("--------------------------------------")

    return response


if __name__ == "__main__":
    while True:
        question = input("\n🟢 Pregunta del usuario (o escribe 'exit' para salir):\n> ")
        if question.lower() in ["exit", "quit", "salir"]:
            print("👋 ¡Sesión finalizada!")
            break
        tutor_session(question, auto_crawl=True)