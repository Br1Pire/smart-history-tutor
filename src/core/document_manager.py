import json
import logging
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

class DocumentManager:
    def __init__(self, raw_path = None, processed_path = None, prompts_path = None, titles_path = None, ids_path = None, load = True):
        
        self.raw_path = raw_path
        self.processed_path = processed_path
        self.prompts_path = prompts_path
        self.titles_path = titles_path
        self.ids_path = ids_path

        self.raw_documents = []
        self.processed_documents = []
        self.prompts = {}
        self.titles = []
        self.ids = []

        if load : self.load_all()
            
        else: self.prompts = self._load_json(self.prompts_path)

    def _load_json(self, path):
        
        try:
            with open(path, "r", encoding="utf-8") as f:
                logging.info(f"📂 Datos cargados desde: {path}")
                return json.load(f)
        except FileNotFoundError:
            logging.error(f"❌ Error: Archivo '{path}' no encontrado.")
            return []
        except json.JSONDecodeError:
            logging.error(f"❌ Error: No se pudo decodificar el archivo JSON '{path}'. Asegúrate de que sea un JSON válido.")
            return []
        except Exception as e:
            logging.error(f"❌ Error al cargar chunks desde '{path}': {e}")
            return []    
        
    def clear(self):
        self.raw_documents = []
        self.processed_documents = []
        self.titles = []
        self.ids = []
    
    def _save_json(self, data, path):
        
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def load_all(self):
       
        if self.raw_path is None: self.raw_documents = None
        else: self.raw_documents = self._load_json(self.raw_path)

        if self.processed_path is None : self.processed_documents = None
        else: self.processed_documents = self._load_json(self.processed_path)

        if self.prompts_path is None : self.prompts = None
        else: self.prompts = self._load_json(self.prompts_path)

        if self.titles_path is None : self.titles = None
        else: self.titles = self._load_json(self.titles_path)

        if self.ids_path is None : self.ids = None
        else: self.ids = self._load_json(self.ids_path)

    def add_articles(self, new_articles):
        if self.raw_documents is None:
            logging.warning("No se pueden añadir artículos raw porque la ruta raw_path fue None al inicializar. Ignorando.")
            return
    
        existing_articles = {a["title"] for a in self.raw_documents}

        add_articles = []
        for art in new_articles:
            if art["title"] in existing_articles: continue
            add_articles.append(art) 

        self.raw_documents.extend(add_articles)
        self._save_json(self.raw_documents,self.raw_path)
        return add_articles

    def add_chunks(self, new_chunks):
        if self.processed_documents is None:
            logging.warning("No se pueden añadir chunks porque la ruta processed_path fue None al inicializar. Ignorando.")
            return
        
        existing_chunks = {a["id"] for a in self.processed_documents}

        add_chunks = []
        for chunk in new_chunks:
            if chunk["id"] in existing_chunks: continue
            add_chunks.append(chunk) 

        self.processed_documents.extend(add_chunks)
        self._save_json(self.processed_documents,self.processed_path)
        return add_chunks
    
    def get_new_articles(self):
        if self.processed_documents is None or self.raw_documents is None:
            logging.warning("No se pueden obtener titulos nuevos porque la ruta processed_path o raw_path fue None al inicializar. Ignorando.")
            return
        
        processed_titles = {chunk["title"] for chunk in self.processed_documents}
        new_articles = []

        for article in self.raw_documents:
            if article["title"] in processed_titles: continue
            new_articles.append(article)

        return new_articles

    def add_ids(self, new_ids):
        if self.ids is None:
            logging.warning("No se pueden añadir ids porque la ruta ids_path fue None al inicializar. Ignorando.")
            return
        
        add_ids = []
        for id in new_ids:
            if id in self.ids: continue
            add_ids.append(id)
        
        self.ids.extend(add_ids)    
        self._save_json(self.ids,self.ids_path)
        return add_ids
    
    def get_new_chunks(self):
        if self.processed_documents is None or self.ids is None:
            logging.warning("No se pueden obtener titulos nuevos porque la ruta processed_path o ids_path fue None al inicializar. Ignorando.")
            return
        
        new_chunks = []
        temp = set(self.ids)

        for chunk in self.processed_documents:
            if chunk["id"] in temp: continue
            new_chunks.append(chunk)

        return new_chunks


           