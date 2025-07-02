import logging
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

    def __init__(self, crawler: Crawler, preprocessor: Preprocessor, vectorizer: Vectorizer, generator: Generator, retriever: Retriever,index_manager: FaissManager, category_manager: FaissManager, document_manager: DocumentManager):
        self.crawler = crawler
        self.preprocessor = preprocessor
        self.vectorizer = vectorizer
        self.generator = generator
        self.retriever = retriever
        self.index_manager = index_manager
        self.category_manager = category_manager
        self.document_manager = document_manager

    def answer_question(self, question):
        return self.retriever.strategic_retrieve(question)
        
