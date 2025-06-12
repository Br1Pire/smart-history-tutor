from retriever_agent import retrieve_chunks
from generator_agent import generate_answer
from crawler_agent import crawl_single
from vectorizer_agent import segment_article, process_chunks

# Ruta del corpus segmentado
SEGMENTED_PATH = "data/processed/wiki_articles_segmented.json"

# Umbral mínimo de contexto aceptable (puedes ajustar)
MIN_CONTEXT_LENGTH = 300  # número mínimo de caracteres combinados

def tutor_session(question, auto_crawl=True):
    print(f"\n👤 Usuario: {question}")

    # Paso 1: recuperar contexto inicial
    context_results = retrieve_chunks(question, top_k=5)
    combined_context = " ".join([r['chunk'] for r in context_results])
    context_length = len(combined_context)

    print(f"\n📚 Contexto recuperado (longitud total: {context_length} caracteres)\n")

    # Imprimir chunks con sus scores
    for i, r in enumerate(context_results):
        print(f"🔹 Chunk {i + 1} | Score: {r['score']:.4f}")
        print(f"   📘 Título: {r.get('id', 'Desconocido').split("__")[0]} | Index: {r.get('id', 'Desconocido').split("__")[1]}")
        #print(f"   📘 ID: {r.get('id', 'Desconocido')}")
        print(f"   📄 Contenido: {r['chunk']}\n")

    # Paso 2: decidir si buscar nuevo artículo
    if context_length < MIN_CONTEXT_LENGTH and auto_crawl:
        print("⚠️ Contexto insuficiente. Buscando artículo relevante con el Agente Crawler...")

        # Crawler busca un artículo
        new_article = crawl_single(question)
        if new_article:
            print(f"✅ Artículo encontrado: {new_article['title']}")

            # Segmentar + vectorizar
            print("🧠 Segmentando y vectorizando nuevo contenido...")
            chunks = segment_article(new_article, save_path=SEGMENTED_PATH)
            process_chunks(chunks)

            # Volver a recuperar contexto
            context_results = retrieve_chunks(question, top_k=5)

            print("\n🔄 Contexto actualizado tras segmentar nuevo artículo:")
            for i, r in enumerate(context_results):
                print(f"🔹 Chunk {i + 1} | Score: {r['score']:.4f}")
                print(f"   📘 Título: {r.get('id', 'Desconocido').split("__")[0]} | Index: {r.get('id', 'Desconocido').split("__")[1]}")
                #print(f"   📘 ID: {r.get('id', 'Desconocido')}")
                print(f"   📄 Contenido: {r['chunk']}\n")

    # Paso 3: Generar respuesta
    chunks_text = [r['chunk'] for r in context_results]
    response = generate_answer(question, chunks_text)

    # Paso 4: Devolver al usuario
    print("\n🤖 Tutor:")
    print(response)
    return response


if __name__ == "__main__":
    while True:
        question = input("\n🟢 Pregunta del usuario (o escribe 'exit' para salir):\n> ")
        if question.lower() in ["exit", "quit", "salir"]:
            print("👋 ¡Sesión finalizada!")
            break
        tutor_session(question)
