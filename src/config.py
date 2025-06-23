import os

# BASE PATHS
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(BASE_DIR, "logs")
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")
VECTORSTORE_DIR = os.path.join(DATA_DIR, "vectorstore_faiss")

# CREAR DIRECTORIOS SI NO EXISTEN
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(VECTORSTORE_DIR, exist_ok=True)

# LOG FILES
LOG_FILES = {
    "crawler": os.path.join(LOG_DIR, "crawler_agent.log"),
    "preprocessor": os.path.join(LOG_DIR, "preprocessor_agent.log"),
    "vectorizer": os.path.join(LOG_DIR, "vectorizer_agent.log"),
    "retriever": os.path.join(LOG_DIR, "retriever_agent.log"),
    "generator": os.path.join(LOG_DIR, "generator_agent.log"),
    "tutor": os.path.join(LOG_DIR, "tutor_agent.log"),
    "chunking": os.path.join(LOG_DIR, "metaheuristic_chunking.log"),
}

# DATA FILES
TITLES_FILE = os.path.join(DATA_DIR, "titles", "specific_wiki_titles.json")
RAW_FILE = os.path.join(DATA_DIR, "raw", "wiki_articles_raw.json")
PROCESSED_FILE = os.path.join(DATA_DIR, "processed", "wiki_articles_processed.json")

# FAISS FILES
FAISS_INDEX_PATH = os.path.join(VECTORSTORE_DIR, "faiss_index.index")
CATEGORY_FAISS_PATH = os.path.join(VECTORSTORE_DIR, "category_faiss.index")
IDS_PATH = os.path.join(VECTORSTORE_DIR, "ids.pkl")
TEXTS_PATH = os.path.join(VECTORSTORE_DIR, "texts.pkl")

# MODELS
MODEL_PATH = os.path.join(MODEL_DIR, "all-mpnet-base-v2")
GENERATIVE_MODEL_NAME = "gemini-2.5-flash"

# PROMPT FILES
PROMPTS_DIR = os.path.join(DATA_DIR, "prompts")
PROMPT_FILES = {
    "answer": os.path.join(PROMPTS_DIR, "answer_prompt.txt"),
    "check": os.path.join(PROMPTS_DIR, "check_prompt.txt"),
    "refine": os.path.join(PROMPTS_DIR, "refine_prompt.txt"),
    "fix": os.path.join(PROMPTS_DIR, "fix_prompt.txt"),
    "wiki": os.path.join(PROMPTS_DIR, "wiki_article_prompt.txt"),
}

# HYPERPARAMETERS
MAX_CHUNK_SIZE = 500
MIN_CHUNK_SIZE = 400
MAX_ITER = 5000
TOP_K_CHUNKS = 5
MAX_ATTEMPTS = 3
CATEGORY_WEIGHT = 0.3

# API KEYS
GOOGLE_API_KEY = "AIzaSyAAY_YacYAzOV-klmHA_uFjyFDSMrEFtDI"


