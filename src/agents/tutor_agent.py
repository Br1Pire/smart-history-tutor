import logging
import random
import numpy as np
from src.agents.retriever_agent import Retriever
from src.agents.generator_agent import Generator
from src.agents.crawler_agent import Crawler
from src.agents.vectorizer_agent import Vectorizer
from src.agents.preprocessor_agent import Preprocessor
from src.core.faiss_manager import FaissManager
from src.core.document_manager import DocumentManager
from src.config import LOG_FILES

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

class Tutor:
    """
    Tutor inteligente capaz de generar subtemas, textos, pruebas, planes de clases y evaluar resultados estudiantiles.
    """

    def __init__(self, crawler: Crawler, preprocessor: Preprocessor, vectorizer: Vectorizer,
                 generator: Generator, retriever: Retriever, index_manager: FaissManager,
                 category_manager: FaissManager, document_manager: DocumentManager):
        """
        Inicializa el Tutor con todos sus agentes y managers asociados.
        """
        self.crawler = crawler
        self.preprocessor = preprocessor
        self.vectorizer = vectorizer
        self.generator = generator
        self.retriever = retriever
        self.index_manager = index_manager
        self.category_manager = category_manager
        self.document_manager = document_manager
        logging.info("✅ Tutor inicializado correctamente.")

    def answer_question(self, question):
        """
        Responde una pregunta utilizando recuperación estratégica.
        """
        logging.info(f"🔎 Respondiendo pregunta: {question}")
        return self.retriever.strategic_retrieve(question)

    def generate_subtopics(self, topic, amount_range):
        """
        Genera subtemas para un tema dado dentro de un rango de cantidad.
        """
        logging.info(f"📚 Generando subtemas para {topic}, rango: {amount_range}")
        return self.generator.generate_subtopics(topic, amount_range[0], amount_range[1])

    def retrive_chunks_from_subtopics(self, topics):
        """
        Recupera chunks relevantes para cada subtema.
        """
        logging.info("🔍 Recuperando chunks para subtemas seleccionados.")
        subtopics = [pair[0] for pair in topics]
        querys = [pair[1] for pair in topics]
        subtopics_with_chunks = {}
        for subtopic, query in zip(subtopics, querys):
            retrieved_chunks = self.retriever.retrieve_chunks_from_vector(
                self.vectorizer.vectorize_query(query), top_k=15)
            subtopics_with_chunks[subtopic] = {
                "subtopic": subtopic,
                "query": query,
                "chunks": [{"chunk_id": chunk["id"], "text": chunk["chunk"]} for chunk in retrieved_chunks]
            }
        logging.info("✅ Chunks recuperados para todos los subtemas.")
        return subtopics_with_chunks

    def generate_texts_from_subtopics(self, subtopics_with_chunks: dict):
        """
        Genera textos explicativos para cada subtema usando sus chunks.
        """
        logging.info("📝 Generando textos para subtemas.")
        subtopics_with_text = {}
        for subtopic, value in subtopics_with_chunks.items():
            text = self.generator.generate_text_for_subtopic(subtopic, value['chunks'])
            chunked_texts = [x[1] for x in self.preprocessor._chunk_section_text(subtopic, text, clean=False)]
            subtopics_with_text[subtopic] = {
                "subtopic": subtopic,
                "query": value["query"],
                "chunks": value['chunks'],
                "texts": chunked_texts
            }
        logging.info("✅ Textos generados para todos los subtemas.")
        return subtopics_with_text

    def generate_tests_from_subtopics(self, subtopics_with_text: dict):
        """
        Genera pruebas de desarrollo para cada subtema a partir de sus textos.
        """
        logging.info("🧪 Generando pruebas de desarrollo para subtemas.")
        tests = {}
        for subtopic, value in subtopics_with_text.items():
            test_item = self.generator.generate_development_question(subtopic, "".join(value['texts']))
            tests[subtopic] = {
                "subtopic": subtopic,
                "question": test_item.get("pregunta"),
                "answer": test_item.get("respuesta_correcta")
            }
        logging.info("✅ Pruebas generadas para todos los subtemas.")
        return tests

    def generate_plan(self, class_distribution, texts, tests):
        """
        Genera un plan de clases completo combinando la distribución, los textos y las pruebas.
        """
        logging.info("📅 Generando plan de clases completo.")
        sessions_with_text = []
        topics = set()
        for session in class_distribution["sessions"]:
            sessions_with_text.append({
                "session_id": session["session_id"],
                "type": session['type'],
                "subtopics": session["subtopics"],
                "texts": [(subtopic, texts[subtopic]['texts']) for subtopic in session["subtopics"]]
            })
            topics.update(topic for topic in session['subtopics'])
        add_tests = [tests[topic] for topic in topics]
        plan = {
            "coverage_used": class_distribution["coverage"],
            "num_classes_used": class_distribution["num_classes"],
            "sessions": sessions_with_text,
            "test": add_tests,
        }
        logging.info(f"✅ Plan generado con {plan['num_classes_used']} clases, cobertura: {plan['coverage_used']:.2%}.")
        return plan
    
    def generate_class_distribution(self, subtopic_query_pairs, constraints):
        """
        Genera una distribución heurística de clases mezclando sesiones de contenido y repaso
        hasta alcanzar una cobertura objetivo dentro de los rangos especificados.

        Args:
            subtopic_query_pairs (list of tuples): Lista de subtemas con sus consultas.
            constraints (dict): Diccionario con cobertura mínima y rango de clases.

        Returns:
            dict: Distribución generada con número de clases, cobertura lograda y detalle de sesiones.
        """
        logging.info("🎯 Generando distribución de clases (heurística).")
        subtopic_list = [pair[0] for pair in subtopic_query_pairs]

        min_coverage = constraints["coverage"]
        min_classes, max_classes = constraints["num_classes_range"]
        num_classes = random.randint(min_classes, max_classes)

        plan = []
        taught_subtopics = set()
        review_counts = {}

        target_coverage = random.uniform(min_coverage, 1.0)

        subtopics_shuffled = subtopic_list.copy()
        random.shuffle(subtopics_shuffled)

        class_id = 1

        while len(plan) < num_classes:
            current_coverage = len(taught_subtopics) / len(subtopic_list)

            if current_coverage < target_coverage and (random.random() < 0.7 or len(taught_subtopics) == 0):
                available_content = [s for s in subtopics_shuffled if s not in taught_subtopics]
                if available_content:
                    num_subtopics = random.randint(1, min(2, len(available_content)))
                    subtopics_for_class = random.sample(available_content, num_subtopics)

                    plan.append({
                        "session_id": class_id,
                        "type": "content",
                        "subtopics": subtopics_for_class
                    })
                    class_id += 1

                    taught_subtopics.update(subtopics_for_class)
                    for s in subtopics_for_class:
                        review_counts.setdefault(s, 0)
            else:
                if random.random() < 0.5:
                    available_review = [s for s in taught_subtopics if review_counts.get(s, 0) < 1]
                    if available_review:
                        num_subtopics = random.randint(1, min(3, len(available_review)))
                        subtopics_for_review = random.sample(available_review, num_subtopics)

                        plan.append({
                            "session_id": class_id,
                            "type": "review",
                            "subtopics": subtopics_for_review
                        })
                        class_id += 1

                        for s in subtopics_for_review:
                            review_counts[s] += 1
                    else:
                        continue
                else:
                    if current_coverage < target_coverage:
                        continue
                    else:
                        break

        coverage = len(taught_subtopics) / len(subtopic_list)

        logging.info(f"✅ Distribución generada con {len(plan)} clases. Cobertura objetivo: {target_coverage:.2%}. Cobertura alcanzada: {coverage:.2%}")
        for s in plan:
            logging.info(f"🗂️ {s}")

        return {
            "num_classes": len(plan),
            "coverage": coverage,
            "sessions": plan
        }


    def evaluate_student_tests(self, student_tests, tests_generated):
        """
        Evalúa respuestas de estudiantes comparándolas con las correctas usando similaridad de coseno.

        Returns:
            list of dict: resultados con promedio y detalles por estudiante.
        """
        logging.info("🔬 Evaluando tests de estudiantes.")
        results = []
        
        for question in tests_generated:
            question['vectorized_answer'] = self.vectorizer.vectorize_query(question['answer'])
        for student_test in student_tests:
            scores = []
            details = []
            for question in student_test:
                student_vec = self.vectorizer.vectorize_query(question['answer'])[0]
                for x in tests_generated:
                    if x['subtopic'] == question['subtopic']:
                        correct_vec = x['vectorized_answer'][0]
                        correct_ans = x['answer']
                similarity = np.dot(student_vec, correct_vec)
                scores.append(similarity)
                details.append({
                    "subtopic": question['subtopic'],
                    "student_response": question['answer'],
                    "correct_answer": correct_ans,
                    "score": float(similarity)
                })
            avg_score = np.mean(scores)
            results.append({
                "student_name": student_test[0]['name'],
                "average_score": float(avg_score),
                "details": details
            })
        logging.info("✅ Evaluación de tests completada.")
        return results
