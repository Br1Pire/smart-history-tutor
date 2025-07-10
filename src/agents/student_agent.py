import random
import numpy as np
import spacy
import logging
from src.core.faiss_manager import FaissManager
from src.core.document_manager import DocumentManager
from src.agents.generator_agent import Generator
from src.agents.vectorizer_agent import Vectorizer
from src.agents.retriever_agent import Retriever
from src.core.fuzzy_system import calculate_learning_and_skip
from src.core.motivation_fuzzy_system import calculate_delta_motivation
from src.config import LOG_FILES

LOG_FILE = LOG_FILES["document_manager"]
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE, mode='a', encoding='utf-8'),
        logging.StreamHandler()
    ]
)

nlp = spacy.load("es_core_news_md")


class StudentAgent:
    """Agente estudiante que simula procesamiento, retención y recuerdo de información
    basado en parámetros de aprendizaje, motivación y entorno."""

    def __init__(
        self,
        name,
        faiss_manager: FaissManager,
        document_manager: DocumentManager,
        retriever: Retriever,
        generator: Generator,
        vectorizer: Vectorizer,
        learning_rate=0.8,
        detail_preference="neutral",
        skip_probability=0.0,
        forgetting_rate=0.0,
        noise_factor=0.05,
        motivation_value=10,
        state_value=10,
        environment_value=2,
    ):
        """Inicializa un agente estudiante con sus preferencias y recursos.

        Args:
            name (str): Nombre del estudiante.
            faiss_manager (FaissManager): Gestor del índice FAISS.
            document_manager (DocumentManager): Gestor de documentos procesados.
            retriever (Retriever): Agente recuperador de chunks.
            generator (Generator): Generador de respuestas.
            vectorizer (Vectorizer): Vectorizador de textos.
            learning_rate (float, opcional): Tasa base de aprendizaje. Default 0.8.
            detail_preference (str, opcional): Preferencia de detalle ('start', 'end' o 'neutral'). Default 'neutral'.
            skip_probability (float, opcional): Probabilidad base de omisión. Default 0.0.
            forgetting_rate (float, opcional): Probabilidad de olvido. Default 0.0.
            noise_factor (float, opcional): Factor de variación aleatoria en el aprendizaje. Default 0.05.
            motivation_value (float, opcional): Valor inicial de motivación. Default 10.
            state_value (float, opcional): Valor de estado interno. Default 10.
            environment_value (float, opcional): Valor del entorno. Default 2.
        """
        self.name = name
        self.faiss_manager = faiss_manager
        self.document_manager = document_manager
        self.vectorizer = vectorizer
        self.generator = generator
        self.retriever = retriever

        self.base_learning_rate = learning_rate
        self.base_skip_probability = skip_probability

        self.learning_rate = learning_rate
        self.detail_preference = detail_preference
        self.skip_probability = skip_probability
        self.forgetting_rate = forgetting_rate
        self.noise_factor = noise_factor
        self.motivation = motivation_value
        self.environment = environment_value
        self.state = state_value

        logging.info(f"🎓 Estudiante '{self.name}' inicializado.")

    def clear_memory(self):
        """Limpia toda la memoria almacenada del estudiante."""
        logging.info(f"{self.name} está limpiando su memoria.")
        self.document_manager.clear()
        self.faiss_manager.clean_index()

    def learn_chunk(self, text, subtopic, index):
        """Procesa y retiene parte de un texto según sus características de aprendizaje.

        Args:
            text (str): Texto de entrada a procesar.
            subtopic (str): Subtema o etiqueta del texto.
            index (int): Índice del chunk para referencia.
        """
        self.update_learning_and_skip()

        if random.random() < self.skip_probability:
            logging.info(
                f"⏩ {self.name} omitió el chunk '{subtopic}_{index}' completamente "
                f"(prob_omisión={self.skip_probability:.2f})."
            )
            return

        doc = nlp(text)
        sentences = [sent.text.strip() for sent in doc.sents]
        n = len(sentences)

        effective_lr = max(
            0.0,
            min(1.0, self.learning_rate + random.uniform(-self.noise_factor, self.noise_factor)),
        )
        num_to_retain = max(1, int(n * effective_lr))

        if self.detail_preference == "start":
            retained_sentences = sentences[:num_to_retain]
        elif self.detail_preference == "end":
            retained_sentences = sentences[-num_to_retain:]
        else:
            retained_sentences = (
                random.sample(sentences, num_to_retain) if num_to_retain < n else sentences
            )

        retained_text = " ".join(retained_sentences)

        save_chunk = {
            "id": f"{subtopic}_{index}",
            "title": subtopic,
            "content": retained_text,
        }

        self.document_manager.add_chunks([save_chunk])

        logging.info(
            f"✅ {self.name} aprendió el chunk '{subtopic}_{index}' reteniendo "
            f"{num_to_retain}/{n} oraciones (tasa efectiva={effective_lr:.2f})."
        )

    def take_session(self, session):
        """Realiza una sesión de estudio procesando múltiples temas y textos.

        Args:
            session (dict): Diccionario con la clave 'texts' conteniendo los temas y textos.
        """
        logging.info(f"{self.name} inicia sesión con {len(session['texts'])} temas.")
        for topic in session["texts"]:
            logging.info(f"{self.name} procesando tema '{topic[0]}' con {len(topic[1])} textos.")
            for i, text in enumerate(topic[1]):
                self.learn_chunk(text, topic[0], i + 1)

    def forget(self):
        """Aplica olvido probabilístico a un chunk almacenado."""
        total_chunks = len(self.document_manager.processed_documents)

        if total_chunks == 0:
            logging.info(f"🔔 {self.name} no tiene chunks en memoria para olvidar.")
            return

        if np.random.rand() < self.forgetting_rate:
            idx_to_forget = np.random.choice(total_chunks)
            forgotten_chunk = self.document_manager.processed_documents.pop(idx_to_forget)
            logging.info(
                f"🧠 Olvido aplicado: chunk {idx_to_forget} ('{forgotten_chunk.get('title', 'sin título')}') "
                f"eliminado (prob_olvido={self.forgetting_rate:.2f})."
            )
        else:
            logging.info(f"🧠 No se aplicó olvido (prob_olvido={self.forgetting_rate:.2f}).")

    def persist_faiss(self):
        """Persiste el índice FAISS con los embeddings actualizados."""
        logging.info(f"{self.name} está persistiendo el índice FAISS.")
        self.vectorizer.vectorize(False, False)

    def update_learning_and_skip(self):
        """Actualiza la tasa de aprendizaje y probabilidad de omisión usando lógica difusa."""
        old_lr, old_skip = self.learning_rate, self.skip_probability
        self.learning_rate, self.skip_probability = calculate_learning_and_skip(
            self.base_learning_rate,
            self.base_skip_probability,
            self.motivation,
            self.state,
            self.environment,
        )
        logging.info(
            f"{self.name} actualizó learning_rate de {old_lr:.2f} a {self.learning_rate:.2f} "
            f"y skip_probability de {old_skip:.2f} a {self.skip_probability:.2f}."
        )

    def update_environment(self, value):
        """Actualiza el valor del entorno del estudiante.

        Args:
            value (float): Nuevo valor de entorno.
        """
        logging.info(f"{self.name} actualiza environment de {self.environment} a {value}.")
        self.environment = value

    def update_motivation(self, score):
        """Ajusta la motivación según desempeño reciente.

        Args:
            score (float): Puntuación o retroalimentación recibida.
        """
        delta = calculate_delta_motivation(self.learning_rate, self.skip_probability, score)
        new_motivation = max(1, min(self.motivation + delta, 10))
        logging.info(
            f"{self.name} actualiza motivación en {delta:.2f} basado en score {score}. "
            f"Nueva motivación: {new_motivation:.2f}."
        )
        self.motivation = round(new_motivation, 1)

    def get_environment(self):
        """Obtiene el valor actual de entorno.

        Returns:
            float: Valor de environment.
        """
        logging.info(f"{self.name} consulta environment: {self.environment}.")
        return self.environment

    def answer_test(self, test):
        """Responde un test generando respuestas con chunks relevantes.

        Args:
            test (list of dict): Cada dict con 'question' y 'subtopic'.

        Returns:
            list of dict: Respuestas generadas por el estudiante para cada pregunta.
        """
        results = []
        logging.info(f"{self.name} comienza test con {len(test)} preguntas.")

        for item in test:
            question = item["question"]
            logging.info(f"{self.name} respondiendo pregunta: '{question}'.")

            question_embedding = self.vectorizer.vectorize_query(question)
            retrieved_chunks = self.retriever.retrieve_chunks_from_vector(question_embedding, 10)

            context = "\n".join([f"- {chunk['chunk'].strip()}" for chunk in retrieved_chunks])

            answer = self.generator.answer_question_student(question, context)

            results.append(
                {
                    "name": self.name,
                    "subtopic": item["subtopic"],
                    "question": question,
                    "answer": answer,
                }
            )

            logging.info(f"📝 {self.name} respondió pregunta '{question}' con respuesta generada.")

        return results
