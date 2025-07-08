import google.generativeai as genai
import logging
import json
import time
import re

from src.core.document_manager import DocumentManager
from src.config import LOG_FILES, GOOGLE_API_KEY, GENERATIVE_MODEL_NAME


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

class Generator:
    def __init__(self, document_manager: DocumentManager):
        self.document_manager = document_manager
        logging.info("Generator agent inicializado con el modelo generativo.")

    def _call_gemini_with_retry(self, prompt, max_retries=5):
        retries = 0
        while retries < max_retries:
            try:
                response = model.generate_content(prompt)
                # Attempt to get text directly, let SDK raise exception if not available
                return response.text.strip()
            except Exception as e:
                error_message = str(e)
                if "429" in error_message and "quota" in error_message:
                    retries += 1
                    logging.warning(f"⚠️ Error de cuota (429) detectado. Intento {retries}/{max_retries}. Mensaje: {error_message}")
                    
                    retry_delay_match = re.search(r"seconds:\s*(\d+)", error_message)
                    if retry_delay_match:
                        delay = int(retry_delay_match.group(1)) + 5
                    else:
                        delay = 60

                    logging.info(f"Esperando {delay} segundos antes de reintentar...")
                    time.sleep(delay)
                else:
                    # Re-raise any other exception, including those from safety blocks
                    # which would cause response.text to fail.
                    raise e
        raise Exception(f"Máximo de {max_retries} reintentos alcanzado para la llamada a Gemini (no por 429).")

    def generate_answer(self, question: str, context_chunks: list[str]) -> str:
        context_text = "\n".join(f"- {chunk['chunk'].strip()}" for chunk in context_chunks)
        prompt = self.document_manager.prompts["answer_prompt"].format(question=question, context=context_text)
        try:
            logging.info(f"📝 Generando respuesta para: '{question}'")
            answer = self._call_gemini_with_retry(prompt) # Expecting string directly
            logging.info("✅ Respuesta generada con éxito.")
            return answer
        except Exception as e:
            logging.error(f"⚠️ Error generando respuesta: {e}")
            return f"⚠️ Error generando respuesta: {e}"

    def check_context(self, question: str, context_chunks: list[str]) -> bool:
        context_text = "\n".join(f"- {chunk['chunk'].strip()}" for chunk in context_chunks)
        prompt = self.document_manager.prompts["check_prompt"].format(question=question, context=context_text)
        try:
            logging.info(f"🔍 Chequeando contexto para: '{question}'")
            result = self._call_gemini_with_retry(prompt) # Expecting string directly
            logging.info(f"Resultado del chequeo: {result}")
            return "true" in result.lower() # Ensure lower() is applied to the result
        except Exception as e:
            logging.error(f"⚠️ Error durante el chequeo: {e}")
            return False

    def refine_question(self, original_question: str) -> str:
        prompt = self.document_manager.prompts["refine_prompt"].format(original_question=original_question)
        try:
            logging.info(f"✨ Refinando pregunta: '{original_question}'")
            refined = self._call_gemini_with_retry(prompt) # Expecting string directly
            logging.info(f"✅ Pregunta refinada: '{refined}'")
            return refined
        except Exception as e:
            logging.error(f"⚠️ Error refinando pregunta: {e}")
            return original_question

    def fix_question(self, original_question: str) -> str:
        prompt = self.document_manager.prompts["fix_prompt"].format(original_question=original_question)
        try:
            logging.info(f"🛠 Arreglando pregunta: '{original_question}'")
            fixed = self._call_gemini_with_retry(prompt) # Expecting string directly
            logging.info(f"✅ Pregunta arreglada: '{fixed}'")
            return fixed
        except Exception as e:
            logging.error(f"⚠️ Error arreglando pregunta: {e}")
            return original_question

    def wiki_query(self, question: str) -> str:
        prompt = self.document_manager.prompts["wiki_article_prompt"].format(question=question)
        try:
            logging.info(f"🌐 Generando query para Wikipedia: '{question}'")
            query = self._call_gemini_with_retry(prompt) # Expecting string directly
            logging.info(f"✅ Query generado: '{query}'")
            return query
        except Exception as e:
            logging.error(f"⚠️ Error generando query: {e}")
            return question
        
    def generate_subtopics(self, topic, min_subtopics=5, max_subtopics=10):
        prompt_template = self.document_manager.prompts["generate_subtopics_prompt"]
        prompt = prompt_template.format(
            topic=topic,
            min_subtopics=min_subtopics,
            max_subtopics=max_subtopics
        )
        
        try:
            text = self._call_gemini_with_retry(prompt) # Expecting string directly

            subtopic_query_pairs = []
            lines = text.split("\n")
            i = 0
            while i < len(lines):
                if lines[i].startswith("Subtema:"):
                    subtema = lines[i].replace("Subtema:", "").strip()
                    if i+1 < len(lines) and lines[i+1].startswith("Query:"):
                        query = lines[i+1].replace("Query:", "").strip()
                        subtopic_query_pairs.append((subtema, query))
                        i += 2
                    else:
                        i += 1
                else:
                    i += 1

            print(f"🔎 Subtemas y queries generados para '{topic}': {subtopic_query_pairs}")
            return subtopic_query_pairs
        except Exception as e:
            logging.error(f"⚠️ Error generando subtemas para {topic}: {e}")
            return []

    def generate_text_for_subtopic(self, subtopic, chunks):
        context_text = "\n".join([f"- {chunk['text'].strip()}" for chunk in chunks])

        prompt = self.document_manager.prompts["class_generation_prompt"].format(
            subtopic=subtopic,
            context=context_text
        )

        try:
            logging.info(f"📚 Generando clase para subtopic: {subtopic}")
            text = self._call_gemini_with_retry(prompt) # Expecting string directly
            return text
        except Exception as e:
            logging.error(f"⚠️ Error generando clase para {subtopic}: {e}")
            return f"⚠️ Error generando clase: {e}"

    def generate_development_question(self, subtopic, class_text):
        prompt = self.document_manager.prompts["development_question_prompt"].format(
            subtopic=subtopic,
            class_text=class_text
        )
        try:
            logging.info(f"📝 Generando pregunta de desarrollo para subtopic: {subtopic}")
            raw = self._call_gemini_with_retry(prompt) # Expecting string directly

            lines = raw.split("\n")
            pregunta = next((line.replace("Pregunta:", "").strip() for line in lines if "Pregunta:" in line), None)
            respuesta = next((line.replace("Respuesta:", "").strip() for line in lines if "Respuesta:" in line), None)

            return {
                "pregunta": pregunta,
                "respuesta_correcta": respuesta
            }
        except Exception as e:
            logging.error(f"⚠️ Error generando pregunta de desarrollo para {subtopic}: {e}")
            return {
                "pregunta": f"⚠️ Error generando pregunta: {e}",
                "respuesta_correcta": None
            }

    def answer_question_student(self, question, context):
        prompt = self.document_manager.prompts["answer_student_prompt"].format(
                    question=question,
                    context=context)
        try:
            logging.info(f"📝 Generando respuesta para la pregunta: {question}")
            answer = self._call_gemini_with_retry(prompt) # Expecting string directly

            return answer

        except Exception as e:
            logging.error(f"⚠️ Error generando respuesta para la pregunta '{question}': {e}")
            return f"⚠️ Error generando respuesta: {e}"