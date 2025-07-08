import random
import numpy as np
import spacy
from src.core.faiss_manager import FaissManager
from src.core.document_manager import DocumentManager
from src.agents.generator_agent import Generator
from src.agents.vectorizer_agent import Vectorizer
from src.agents.crawler_agent import Crawler
from src.agents.preprocessor_agent import Preprocessor
from src.agents.retriever_agent import Retriever
from src.core.fuzzy_system import calculate_learning_and_skip
from src.core.motivation_fuzzy_system import calculate_delta_motivation

nlp = spacy.load("es_core_news_md")

class StudentAgent:
    def __init__(self, name, faiss_manager: FaissManager, document_manager:DocumentManager, retriever: Retriever, generator: Generator, vectorizer: Vectorizer,
                 learning_rate=0.8,
                 detail_preference="neutral",
                 skip_probability=0.0,
                 forgetting_rate=0.0,
                 noise_factor=0.05,
                 motivation_value = 10,
                 state_value = 10,
                 environment_value = 2,

                ):
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

        print(f"🎓 Student '{self.name}' initialized.")

    def learn_chunk(self, text, subtopic, index):
        """
        El estudiante procesa el texto del chunk según sus características y genera su embedding parcial.
        """
        self.update_learning_and_skip()

        # 1. Skip probability
        if random.random() < self.skip_probability:
            print(f"⏩ {self.name} omitió el chunk completamente.")
            return

        # 2. Procesa oraciones con SpaCy
        doc = nlp(text)
        sentences = [sent.text.strip() for sent in doc.sents]
        n = len(sentences)

        # 3. Determina cuántas oraciones retiene según learning_rate y noise
        effective_lr = max(0.0, min(1.0, self.learning_rate + random.uniform(-self.noise_factor, self.noise_factor)))
        num_to_retain = max(1, int(n * effective_lr))

        # 4. Selecciona oraciones según detail_preference
        if self.detail_preference == "start":
            retained_sentences = sentences[:num_to_retain]
        elif self.detail_preference == "end":
            retained_sentences = sentences[-num_to_retain:]
        else:  # neutral or random
            retained_sentences = random.sample(sentences, num_to_retain) if num_to_retain < n else sentences

        # 5. Crea texto parcial retenido
        retained_text = " ".join(retained_sentences)

        save_chunk = {
            'id': f"{subtopic}_{index}",
            'title': subtopic,
            'content': retained_text
        }

        self.document_manager.add_chunks([save_chunk])
        
        print(f"✅ {self.name} aprendió un chunk parcial con {num_to_retain}/{n} oraciones retenidas.")

    def take_session(self, session):

        for topic in session['texts']:
            for i, text in enumerate(topic[1]):
                self.learn_chunk(text, topic[0], i+1)

        

    def forget(self):
        total_chunks = len(self.document_manager.processed_documents)

        if total_chunks == 0:
            print("🔔 No hay chunks en memoria para olvidar.")
            return

        # Decide probabilísticamente si olvida
        if np.random.rand() < self.forgetting_rate:
            idx_to_forget = np.random.choice(total_chunks)
            forgotten_chunk = self.document_manager.processed_documents.pop(idx_to_forget)
            print(f"🧠 Forget aplicado: chunk {idx_to_forget} olvidado -> {forgotten_chunk.get('title', 'sin título')}.")

        else:
            print(f"🧠 Forget no aplicado esta vez (ratio={self.forgetting_rate}).")

    def persist_faiss(self):
        self.vectorizer.vectorize(False,False)

    def update_learning_and_skip(self):
        self.learning_rate, self.skip_probability = calculate_learning_and_skip(self.base_learning_rate,self.base_skip_probability,self.motivation,self.state,self.environment)

    def update_environment(self, value):
        self.environment = value

    def update_motivation(self, score):
        delta = calculate_delta_motivation(self.learning_rate, self.skip_probability, score)
        new_motivation = max(1, min(self.motivation+delta, 10))
        self.motivation = round(new_motivation,1)

    def get_environment(self):
        return self.environment

    def answer_test(self, test):
        """
        El estudiante responde un test de desarrollo.

        Args:
            test (list of dict): cada dict con keys 'pregunta' y 'respuesta_correcta'.

        Returns:
            list of dict: respuestas generadas con evaluación simple.
        """
        results = []

        for item in test:
            question = item["question"]

            question_embedding = self.vectorizer.vectorize_query(question)

            retrieved_chunks = self.retriever.retrieve_chunks_from_vector(question_embedding, 10)

            context = "\n".join([f"- {chunk['chunk'].strip()}" for chunk in retrieved_chunks])

            answer = self.generator.answer_question_student(question,context)

            results.append({
                "name": self.name,
                "subtopic": item["subtopic"],
                "question": question,
                "answer": answer,
            })

            print(f"📝 {self.name} respondió la pregunta: '{question}'")

        return results
