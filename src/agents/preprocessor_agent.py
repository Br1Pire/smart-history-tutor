import os
import json
import re
import spacy
import logging
from src.core.metaheuristic_chunking import chunk_section_text_metaheuristic
from src.config import LOG_FILES, RAW_FILE, PROCESSED_FILE, MAX_CHUNK_SIZE, MIN_CHUNK_SIZE

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

INPUT_FILE = RAW_FILE
OUTPUT_FILE = PROCESSED_FILE

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


def load_articles(path):
    """
    Carga artículos desde un archivo JSON.

    Args:
        path (str): Ruta del archivo de entrada.

    Returns:
        list: Lista de artículos cargados o lista vacía si no existe.
    """
    if not os.path.exists(path):
        logging.error(f"❌ Archivo no encontrado: {path}")
        return []
    with open(path, "r", encoding="utf-8") as f:
        logging.info(f"📂 Artículos cargados desde: {path}")
        return json.load(f)


def load_existing_chunks(path):
    """
    Carga chunks previamente generados.

    Args:
        path (str): Ruta del archivo JSON.

    Returns:
        list: Lista de chunks existentes o lista vacía.
    """
    if not os.path.exists(path):
        logging.info("ℹ️ No se encontraron chunks existentes previos.")
        return []
    with open(path, "r", encoding="utf-8") as f:
        logging.info(f"📂 Chunks existentes cargados desde: {path}")
        return json.load(f)


def clean_text(text):
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


def split_by_section(content):
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


def chunk_section_text(section_name, text):
    """
    Genera chunks de un texto aplicando la metaheurística de chunking.

    Args:
        section_name (str or None): Nombre de la sección.
        text (str): Texto de la sección.

    Returns:
        list: Lista de chunks generados.
    """
    logging.info(f"🔹 Chunking sección: '{section_name or 'General'}'")
    doc = nlp(clean_text(text))
    sentences = [sent.text.strip() for sent in doc.sents]
    chunk_results, score = chunk_section_text_metaheuristic(section_name, sentences, MAX_CHUNK_SIZE, MIN_CHUNK_SIZE)
    logging.info(f"✅ {len(chunk_results)} chunks generados (Score: {score:.2f}) para sección: '{section_name or 'General'}'")
    return chunk_results


def extract_entities(text):
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


def process_article(article):
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
        (name, text) for (name, text) in split_by_section(content)
        if name is None or name.lower().strip() not in EXCLUDED_SECTIONS
    ]

    final_chunks = []
    for section_name, sec_text in section_splits:
        chunks = chunk_section_text(section_name, sec_text)
        for i, (sec_name, chunk) in enumerate(chunks):
            entities = extract_entities(chunk)
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


def process_file(articles):
    """
    Procesa un conjunto de artículos y guarda los chunks resultantes.

    Args:
        articles (list): Lista de artículos.

    Returns:
        list: Lista de nuevos chunks generados.
    """
    existing_chunks = load_existing_chunks(OUTPUT_FILE)
    processed_titles = {chunk["title"] for chunk in existing_chunks}
    logging.info(f"ℹ️ {len(existing_chunks)} chunks existentes de {len(processed_titles)} artículos previos cargados.")

    all_chunks = existing_chunks.copy()
    new_chunks = []

    for article in articles:
        title = article.get("title")
        if title in processed_titles:
            logging.info(f"⏩ Artículo '{title}' ya procesado previamente. Se omite.")
            continue
        chunks = process_article(article)
        all_chunks.extend(chunks)
        new_chunks.extend(chunks)
        processed_titles.add(title)

    save_chunks(all_chunks, OUTPUT_FILE)
    logging.info(f"💾 Chunks guardados. Total: {len(all_chunks)}")
    return new_chunks


def save_chunks(chunks, path):
    """
    Guarda los chunks en un archivo JSON.

    Args:
        chunks (list): Lista de chunks.
        path (str): Ruta del archivo de salida.
    """
    with open(path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)


def preprocess():
    """
    Ejecuta el preprocesamiento de los artículos desde el archivo de entrada.
    """
    articles = load_articles(INPUT_FILE)
    if not articles:
        logging.warning("⚠️ No se encontraron artículos para procesar.")
        return
    process_file(articles)
    logging.info("🏁 Proceso de preprocesamiento finalizado.")


if __name__ == "__main__":
    preprocess()
