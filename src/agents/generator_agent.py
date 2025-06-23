import os
import google.generativeai as genai
import logging
from src.config import LOG_FILES, GOOGLE_API_KEY, PROMPT_FILES,GENERATIVE_MODEL_NAME
import time

# Configuración de logs
LOG_FILE = LOG_FILES["generator"]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, mode='a', encoding='utf-8'),
        logging.StreamHandler()
    ]
)

if not GOOGLE_API_KEY:
    logging.error("❌ La clave de API no está configurada.")
    raise ValueError("La variable de entorno GOOGLE_API_KEY no está configurada.")

genai.configure(api_key=GOOGLE_API_KEY)

MODEL_NAME = GENERATIVE_MODEL_NAME
model = genai.GenerativeModel(MODEL_NAME)
logging.info(f"✅ Modelo generativo '{MODEL_NAME}' configurado.")


def load_prompt_template(path):
    """
    Carga el contenido de un archivo de plantilla de prompt.

    Args:
        path (str): Ruta al archivo de plantilla.

    Returns:
        str: Contenido de la plantilla.
    """
    with open(path, "r", encoding="utf-8") as f:
        logging.info(f"📂 Prompt template cargado: {path}")
        return f.read()


# Rutas a plantillas
ANSWER_PROMPT_PATH = PROMPT_FILES["answer"]
CHECK_PROMPT_PATH = PROMPT_FILES["check"]
REFINE_PROMPT_PATH = PROMPT_FILES["refine"]
FIX_PROMPT_PATH = PROMPT_FILES["fix"]
WIKI_ARTICLE_PROMPT_PATH = PROMPT_FILES["wiki"]

# Cargar plantillas
ANSWER_PROMPT_TEMPLATE = load_prompt_template(ANSWER_PROMPT_PATH)
CHECK_PROMPT_TEMPLATE = load_prompt_template(CHECK_PROMPT_PATH)
REFINE_PROMPT_TEMPLATE = load_prompt_template(REFINE_PROMPT_PATH)
FIX_PROMPT_TEMPLATE = load_prompt_template(FIX_PROMPT_PATH)
WIKI_ARTICLE_PROMPT_TEMPLATE = load_prompt_template(WIKI_ARTICLE_PROMPT_PATH)


def generate_answer(question: str, context_chunks: list[str]) -> str:
    """
    Genera una respuesta usando el modelo generativo y un contexto dado.

    Args:
        question (str): Pregunta del usuario.
        context_chunks (list): Lista de fragmentos de contexto.

    Returns:
        str: Respuesta generada o mensaje de error.
    """
    context_text = "\n".join(f"- {chunk['chunk'].strip()}" for chunk in context_chunks)
    prompt = ANSWER_PROMPT_TEMPLATE.format(question=question, context=context_text)
    try:
        logging.info(f"📝 Generando respuesta para: '{question}'")
        response = model.generate_content(prompt)
        logging.info("✅ Respuesta generada con éxito.")
        return response.text.strip()
    except Exception as e:
        logging.error(f"⚠️ Error generando respuesta: {e}")
        return f"⚠️ Error generando respuesta: {e}"


def check_context(question: str, context_chunks: list[str]) -> bool:
    """
    Verifica si el contexto es suficiente para responder la pregunta.

    Args:
        question (str): Pregunta del usuario.
        context_chunks (list): Fragmentos de contexto.

    Returns:
        bool: True si el contexto es suficiente, False en caso contrario.
    """
    context_text = "\n".join(f"- {chunk['chunk'].strip()}" for chunk in context_chunks)
    prompt = CHECK_PROMPT_TEMPLATE.format(question=question, context=context_text)
    try:
        logging.info(f"🔍 Chequeando contexto para: '{question}'")
        response = model.generate_content(prompt)
        result = response.text.strip().lower()
        logging.info(f"Resultado del chequeo: {result}")
        return "true" in result
    except Exception as e:
        logging.error(f"⚠️ Error durante el chequeo: {e}")
        return False


def refine_question(original_question: str) -> str:
    """
    Mejora la redacción de una pregunta dada.

    Args:
        original_question (str): Pregunta original.

    Returns:
        str: Pregunta refinada o la original en caso de error.
    """
    prompt = REFINE_PROMPT_TEMPLATE.format(original_question=original_question)
    try:
        logging.info(f"✨ Refinando pregunta: '{original_question}'")
        response = model.generate_content(prompt)
        refined = response.text.strip()
        logging.info(f"✅ Pregunta refinada: '{refined}'")
        return refined
    except Exception as e:
        logging.error(f"⚠️ Error refinando pregunta: {e}")
        return original_question


def fix_question(original_question: str) -> str:
    """
    Corrige posibles errores en la pregunta.

    Args:
        original_question (str): Pregunta original.

    Returns:
        str: Pregunta corregida o la original en caso de error.
    """
    prompt = FIX_PROMPT_TEMPLATE.format(original_question=original_question)
    try:
        logging.info(f"🛠 Arreglando pregunta: '{original_question}'")
        response = model.generate_content(prompt)
        fixed = response.text.strip()
        logging.info(f"✅ Pregunta arreglada: '{fixed}'")
        return fixed
    except Exception as e:
        logging.error(f"⚠️ Error arreglando pregunta: {e}")
        return original_question


def wiki_query(question: str) -> str:
    """
    Genera una query para buscar en Wikipedia basada en la pregunta.

    Args:
        question (str): Pregunta del usuario.

    Returns:
        str: Query generada o la pregunta original en caso de error.
    """
    prompt = WIKI_ARTICLE_PROMPT_TEMPLATE.format(question=question)
    try:
        logging.info(f"🌐 Generando query para Wikipedia: '{question}'")
        response = model.generate_content(prompt)
        query = response.text.strip()
        logging.info(f"✅ Query generado: '{query}'")
        return query
    except Exception as e:
        logging.error(f"⚠️ Error generando query: {e}")
        return question


if __name__ == "__main__":
    q1 = "¿Cuándo comenzó la Revolución Francesa?"
    q2 = "¿Quién fue Napoleón?"
    q3 = "¿Qué causó la Segunda Guerra Mundial?"

    refine_question(q1)
    refine_question(q2)
    refine_question(q3)
