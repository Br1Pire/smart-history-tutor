import os
import json
import re
import spacy
import logging
from src.core.metaheuristic_chunking import chunk_section_text_metaheuristic

# Configuración de logs
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Configuración de SpaCy
logging.info("Cargando modelo SpaCy...")
nlp = spacy.load("es_core_news_md", disable=["parser", "lemmatizer", "textcat"])
nlp.add_pipe("sentencizer")
logging.info("Modelo SpaCy cargado.")

# Configuración de rutas
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_FILE = os.path.join(BASE_DIR, "data", "raw", "wiki_articles_raw.json")
OUTPUT_FILE = os.path.join(BASE_DIR, "data", "processed", "wiki_articles_processed.json")

MAX_CHUNK_SIZE = 500
MIN_CHUNK_SIZE = 400

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
    if not os.path.exists(path):
        logging.error(f"Archivo no encontrado: {path}")
        return []
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def load_existing_chunks(path):
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def clean_text(text):
    # Elimina referencias tipo [1], [12], [b], [nota 1], [abc], etc.
    text = re.sub(r'\[[^\]]+\]', '', text)

    # Elimina encabezados tipo === Sección ===
    text = re.sub(r'=+\s*[^=]+?\s*=+', '', text)

    # Elimina caracteres invisibles Unicode
    text = re.sub(r'[\u200b\u200c\u200d\uFEFF]', '', text)

    # Elimina ISBN y ISSN
    text = re.sub(r'ISBN[\s\d\-]+', '', text)
    text = re.sub(r'ISSN[\s\d\-]+', '', text)

    # Elimina (ver algo)
    text = re.sub(r'\((?:[Vv]er )[^)]+\)', '', text)

    # Elimina secciones finales comunes
    text = re.sub(r'\b(Véase también|Referencias|Bibliografía|Enlaces externos)\b.*', '', text, flags=re.IGNORECASE)

    # Elimina comillas
    text = text.replace('"', '').replace("'", '')

    # Normaliza saltos de línea y espacios
    text = re.sub(r'\n+', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()

    return text

def split_by_section(content):
    pattern = r"(==+)\s*(.*?)\s*\1"
    matches = list(re.finditer(pattern, content))

    splits = []

    if not matches:
        splits.append((None, content.strip()))
        return splits

    first_match = matches[0]
    pre_section_text = content[:first_match.start()].strip()
    if pre_section_text:
        splits.append((None, pre_section_text))

    for i in range(len(matches)):
        current = matches[i]
        section_name = current.group(2).strip()
        section_start = current.end()
        section_end = matches[i + 1].start() if i + 1 < len(matches) else len(content)
        section_content = content[section_start:section_end].strip()
        if section_content:
            splits.append((section_name, section_content))

    return splits

def chunk_section_text(section_name, text):
    doc = nlp(clean_text(text))

    sentences = [sent.text.strip() for sent in doc.sents]
    # for sent in sentences:
    #     print(f"ORACIÓN: '{sent.strip()}' (len={len(sent.strip())})")
    chunk_results, score = chunk_section_text_metaheuristic(section_name, sentences, MAX_CHUNK_SIZE, MIN_CHUNK_SIZE)
    #print(f"******** {chunk_results[0][1]}")
    return chunk_results

def extract_entities(text):
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
    title = article.get("title", "Sin título")
    content = article.get("content", "")

    logging.info(f"Procesando artículo: {title}")
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

    logging.info(f"Artículo '{title}' generó {len(final_chunks)} chunks.")
    return final_chunks

def process_file(articles):
    # Cargar chunks ya procesados
    existing_chunks = load_existing_chunks(OUTPUT_FILE)
    processed_titles = {chunk["title"] for chunk in existing_chunks}
    logging.info(f"Ya hay {len(existing_chunks)} chunks procesados de {len(processed_titles)} artículos previos.")

    all_chunks = existing_chunks.copy()
    new_chunks = []

    for article in articles:
        title = article.get("title")
        if title in processed_titles:
            logging.info(f"Artículo '{title}' ya procesado previamente. Se omite.")
            continue

        # Procesar y añadir nuevos chunks
        chunks = process_article(article)
        #print_list(chunks)
        all_chunks.extend(chunks)
        new_chunks.extend(chunks)
        #print_list(all_chunks)
        processed_titles.add(title)

    # Guardar al final
    save_chunks(all_chunks, OUTPUT_FILE)
    logging.info(f"Proceso finalizado. Total de chunks: {len(all_chunks)}")
    return new_chunks

def print_list(li):
    # for key, value in li[0].items():
    #     print(f"{key} : {value}\n")
    print(li[0]["content"])

def save_chunks(chunks, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)

def main():
    articles = load_articles(INPUT_FILE)
    if not articles:
        return
    process_file(articles)


if __name__ == "__main__":
    main()
