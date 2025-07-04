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
from src import config

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

    def learn_chunk(self, chunk_text):
        """
        El estudiante procesa el texto del chunk según sus características y genera su embedding parcial.
        """
        self.update_learning_and_skip()

        # 1. Skip probability
        if random.random() < self.skip_probability:
            print(f"⏩ {self.name} omitió el chunk completamente.")
            return

        # 2. Procesa oraciones con SpaCy
        doc = nlp(chunk_text)
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

        # 6. Vectoriza texto retenido
        retained_embedding = self.vectorizer.vectorize_query(retained_text)

        # 7. Normaliza
        retained_embedding = retained_embedding / np.linalg.norm(retained_embedding, axis=1, keepdims=True)

        # 8. Guarda en FAISS
        self.faiss_manager.add(retained_embedding)

        # 9. Registra en conocimiento (opcional)
        # self.knowledge.append((retained_text, retained_embedding))

        print(f"✅ {self.name} aprendió un chunk parcial con {num_to_retain}/{n} oraciones retenidas.")

    def forget(self):
        """
        Opcional: modela olvido reduciendo magnitud o eliminando embeddings en FAISS.
        """
        pass

    def update_learning_and_skip(self):
        self.learning_rate, self.skip_probability = calculate_learning_and_skip(self.base_learning_rate,self.base_skip_probability,self.motivation,self.state,self.environment)
