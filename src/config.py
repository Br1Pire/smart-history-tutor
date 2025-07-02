import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).parent.parent

LOG_DIR = PROJECT_ROOT / "src" / "logs"
DATA_DIR = PROJECT_ROOT / "src" / "data"
MODEL_DIR = PROJECT_ROOT / "src" / "models"
VECTORSTORE_DIR = DATA_DIR / "vectorstore_faiss"

# CREAR DIRECTORIOS SI NO EXISTEN
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(VECTORSTORE_DIR, exist_ok=True)

# LOG FILES
LOG_FILES = {
    "crawler": LOG_DIR / "crawler_agent.log",
    "preprocessor": LOG_DIR / "preprocessor_agent.log",
    "vectorizer": LOG_DIR / "vectorizer_agent.log",
    "retriever": LOG_DIR / "retriever_agent.log",
    "generator": LOG_DIR / "generator_agent.log",
    "tutor": LOG_DIR / "tutor_agent.log",
    "chunking": LOG_DIR / "metaheuristic_chunking.log",
    "faiss_manager": LOG_DIR / "faiss_manager.log",
    "document_manager": LOG_DIR / "document_manager.log",
    "tutor_builder": LOG_DIR / "tutor_builder.log"
}

# DATA FILES
TITLES_FILE = DATA_DIR / "titles" / "specific_wiki_titles.json"
RAW_FILE = DATA_DIR / "raw" / "wiki_articles_raw.json"
PROCESSED_FILE = DATA_DIR / "processed" / "wiki_articles_processed.json"

# FAISS FILES
FAISS_INDEX_PATH = VECTORSTORE_DIR / "faiss_index.index"
CATEGORY_FAISS_PATH = VECTORSTORE_DIR / "category_faiss.index"
IDS_FILE = VECTORSTORE_DIR / "ids.json"
TEXTS_FILE = VECTORSTORE_DIR / "texts.pkl"

# MODELS
MODEL_PATH = MODEL_DIR / "all-mpnet-base-v2"
GENERATIVE_MODEL_NAME = "gemini-2.5-flash"

# PROMPT FILES
PROMPTS_FILE = DATA_DIR / "prompts" / "tutor_prompts.json"

# HYPERPARAMETERS
MAX_CHUNK_SIZE = 500
MIN_CHUNK_SIZE = 400
MAX_ITER = 5000
TOP_K_CHUNKS = 5
MAX_ATTEMPTS = 3
CATEGORY_WEIGHT = 0.3

# API KEYS
GOOGLE_API_KEY = "AIzaSyAAY_YacYAzOV-klmHA_uFjyFDSMrEFtDI"