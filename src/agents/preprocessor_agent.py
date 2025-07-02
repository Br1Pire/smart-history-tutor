import re
import spacy
import logging
from src.core.metaheuristic_chunking import chunk_section_text_metaheuristic
from src.core.document_manager import DocumentManager
from src.config import LOG_FILES,MAX_CHUNK_SIZE, MIN_CHUNK_SIZE

# Configuración de logs
LOG_FILE = LOG_FILES["preprocessor"]

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE, mode='a', encoding='utf-8'),
        logging.StreamHandler()
    ]
)

# Configuración de SpaCy
logging.info("🚀 Cargando modelo SpaCy (es_core_news_md)...")
nlp = spacy.load("es_core_news_md", disable=["parser", "lemmatizer", "textcat"])
nlp.add_pipe("sentencizer")
logging.info("✅ Modelo SpaCy cargado correctamente.")

EXCLUDED_SECTIONS = [
    "referencias",
    "bibliografía",
    "bibliografía recomendada",
    "enlaces externos",
    "véase también",
    "notas",
    "véase asimismo",
    "anexos",
    "otros proyectos",
]

class Preprocessor:
    """
    Agente de preprocesamiento que carga artículos, limpia texto, divide en secciones y genera chunks.
    """

    def __init__(self, document_manager: DocumentManager):
        self.document_manager = document_manager
        logging.info(f"Preprocessor inicializado.")

    def _clean_text(self, text):
        """
        Limpia un texto eliminando patrones irrelevantes y normalizando espacios.

        Args:
            text (str): Texto original.

        Returns:
            str: Texto limpio.
        """
        original_len = len(text)
        text = re.sub(r'\[[^\]]+\]', '', text)
        text = re.sub(r'=+\s*[^=]+?\s*=+', '', text)
        text = re.sub(r'[\u200b\u200c\u200d\uFEFF]', '', text)
        text = re.sub(r'ISBN[\s\d\-]+', '', text)
        text = re.sub(r'ISSN[\s\d\-]+', '', text)
        text = re.sub(r'\((?:[Vv]er )[^)]+\)', '', text)
        text = re.sub(r'\b(Véase también|Referencias|Bibliografía|Enlaces externos)\b.*', '', text, flags=re.IGNORECASE)
        text = text.replace('"', '').replace("'", '')
        text = re.sub(r'\n+', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        logging.debug(f"Texto limpiado: {original_len} -> {len(text)} caracteres")
        return text


    def _split_by_section(self, content):
        """
        Divide un contenido en secciones según encabezados de Wikipedia.

        Args:
            content (str): Texto del artículo.

        Returns:
            list: Lista de tuplas (nombre_sección, texto).
        """
        pattern = r"(==+)\s*(.*?)\s*\1"
        matches = list(re.finditer(pattern, content))

        splits = []
        if not matches:
            splits.append((None, content.strip()))
            return splits

        pre_section_text = content[:matches[0].start()].strip()
        if pre_section_text:
            splits.append((None, pre_section_text))

        for i in range(len(matches)):
            section_name = matches[i].group(2).strip()
            section_start = matches[i].end()
            section_end = matches[i + 1].start() if i + 1 < len(matches) else len(content)
            section_content = content[section_start:section_end].strip()
            if section_content:
                splits.append((section_name, section_content))

        return splits


    def _chunk_section_text(self, section_name, text):
        """
        Genera chunks de un texto aplicando la metaheurística de chunking.

        Args:
            section_name (str or None): Nombre de la sección.
            text (str): Texto de la sección.

        Returns:
            list: Lista de chunks generados.
        """
        logging.info(f"🔹 Chunking sección: '{section_name or 'General'}'")
        doc = nlp(self._clean_text(text))
        sentences = [sent.text.strip() for sent in doc.sents]
        chunk_results, score = chunk_section_text_metaheuristic(section_name, sentences, MAX_CHUNK_SIZE, MIN_CHUNK_SIZE)
        logging.info(f"✅ {len(chunk_results)} chunks generados (Score: {score:.2f}) para sección: '{section_name or 'General'}'")
        return chunk_results


    def _extract_entities(self, text):
        """
        Extrae entidades nombradas del texto.

        Args:
            text (str): Texto de entrada.

        Returns:
            dict: Diccionario con listas de personas, ubicaciones y organizaciones.
        """
        doc = nlp(text)
        entities = {"persons": [], "locations": [], "organizations": []}
        for ent in doc.ents:
            if ent.label_ == "PER":
                entities["persons"].append(ent.text.strip())
            elif ent.label_ == "LOC":
                entities["locations"].append(ent.text.strip())
            elif ent.label_ == "ORG":
                entities["organizations"].append(ent.text.strip())
        return {k: list(set(v)) for k, v in entities.items()}


    def _process_article(self, article):
        """
        Procesa un artículo completo en chunks y extrae entidades.

        Args:
            article (dict): Artículo a procesar.

        Returns:
            list: Lista de chunks con metadatos.
        """
        title = article.get("title", "Sin título")
        content = article.get("content", "")
        logging.info(f"🚀 Procesando artículo: '{title}'")

        section_splits = [
            (name, text) for (name, text) in self._split_by_section(content)
            if name is None or name.lower().strip() not in EXCLUDED_SECTIONS
        ]

        final_chunks = []
        for section_name, sec_text in section_splits:
            chunks = self._chunk_section_text(section_name, sec_text)
            for i, (sec_name, chunk) in enumerate(chunks):
                entities = self._extract_entities(chunk)
                final_chunks.append({
                    "id": f"{title}__{sec_name or 'General'}__{i}",
                    "title": title,
                    "section": sec_name or "General",
                    "content": chunk,
                    "categories": article.get("categories", []),
                    "entities": entities,
                    "token_count": len(list(nlp(chunk)))
                })
        logging.info(f"✅ Artículo '{title}' generó {len(final_chunks)} chunks.")
        return final_chunks


    def preprocess(self):
        """
        Procesa un conjunto de artículos y guarda los chunks resultantes.

        Args:
            articles (list): Lista de artículos.

        Returns:
            list: Lista de nuevos chunks generados.
        """
        new_articles = self.document_manager.get_new_articles()
        new_chunks = []

        if not new_articles:
            logging.info("✅ No hay nuevos artículos para preprocesar.")
            return new_chunks

        for article in new_articles:
            chunks = self._process_article(article)
            new_chunks.extend(chunks)

        self.document_manager.add_chunks(new_chunks)

        logging.info("🏁 Proceso de preprocesamiento finalizado.")
        return new_chunks




