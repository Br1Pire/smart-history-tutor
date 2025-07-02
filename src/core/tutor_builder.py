import logging
from src.core.faiss_manager import FaissManager
from src.core.document_manager import DocumentManager
from src.agents.generator_agent import Generator
from src.agents.vectorizer_agent import Vectorizer
from src.agents.crawler_agent import Crawler
from src.agents.preprocessor_agent import Preprocessor
from src.agents.retriever_agent import Retriever
from src.agents.tutor_agent import Tutor
from src import config


LOG_FILE = config.LOG_FILES["tutor_builder"] 

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, mode='a', encoding='utf-8'),
        logging.StreamHandler()
    ]
)

def create_tutor_instance() -> Tutor:
    """
    Instancia y configura todos los agentes y managers necesarios
    para crear una instancia de Tutor.

    Returns:
        Tutor: Una instancia completamente inicializada del agente Tutor.
    """
    logging.info("⚙️ Iniciando la creación de la instancia del Tutor...")

    doc_manager = DocumentManager(
        raw_path=config.RAW_FILE,
        processed_path=config.PROCESSED_FILE,
        prompts_path=config.PROMPTS_FILE,
        titles_path=config.TITLES_FILE,
        ids_path=config.IDS_FILE
    )
    
    text_faiss_manager = FaissManager(config.FAISS_INDEX_PATH)
    category_faiss_manager = FaissManager(config.CATEGORY_FAISS_PATH)
    logging.info("✅ FaissManagers inicializados.")

    preprocessor = Preprocessor(document_manager=doc_manager)
    logging.info("✅ Preprocessor inicializado.")

    crawler = Crawler(document_manager=doc_manager)
    logging.info("✅ Crawler inicializado.")

    vectorizer = Vectorizer(
        document_manager=doc_manager,
        faiss_manager=text_faiss_manager,
        category_manager=category_faiss_manager
    )
    logging.info("✅ Vectorizer inicializado.")

    generator = Generator(document_manager=doc_manager)
    logging.info("✅ Generator inicializado.")

    retriever = Retriever(
        generator=generator,
        vectorizer=vectorizer,
        crawler=crawler,
        preprocessor=preprocessor,
        document_manager=doc_manager,
        text_manager=text_faiss_manager,
        category_manager=category_faiss_manager
    )
    logging.info("✅ Retriever inicializado.")

    tutor = Tutor(
        crawler=crawler,
        preprocessor=preprocessor,
        vectorizer=vectorizer,
        generator=generator,
        retriever=retriever,
        index_manager=text_faiss_manager, 
        category_manager=category_faiss_manager,
        document_manager=doc_manager
    )
    logging.info("✅ Tutor completamente inicializado.")

    return tutor

if __name__ == "__main__":
    # Ejemplo de uso en tu script principal
    print("Iniciando la aplicación de tutor...")
    my_tutor = create_tutor_instance()

    # my_tutor.vectorizer.vectorize()
    
    assert len(my_tutor.document_manager.processed_documents) == my_tutor.index_manager.index.ntotal, \
    f"Mismatch: processed_documents={len(my_tutor.document_manager.processed_documents)}, FAISS={my_tutor.index_manager.index.ntotal}"
