import os
import json
import re
import spacy
import logging

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

# Configuración
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_FILE = os.path.join(BASE_DIR, "data", "raw", "specific_wiki_articles.json")
OUTPUT_FILE = os.path.join(BASE_DIR, "data", "processed", "wiki_articles_segmented.json")

CHUNK_SIZE = 400  # palabras
MIN_CHUNK_LENGTH = 50
OVERLAP_WORDS = 50

# Inicializar SpaCy
logging.info("Cargando modelo SpaCy...")
nlp = spacy.load("es_core_news_md", disable=["parser", "lemmatizer", "textcat"])
nlp.add_pipe("sentencizer")
logging.info("Modelo SpaCy cargado.")

def clean_text(text):
    text = re.sub(r'==.*?==|\[\[|\]\]|\[[^\]]+\]|\{\{.*?\}\}', '', text)
    text = re.sub(r'\s+', ' ', text).strip()

    date_patterns = [
        (r'(\d{1,2})[/\-\.](\d{1,2})[/\-\.](\d{3,4})', r'\1 de \2 de \3'),
        (r'\b(siglo|Siglo) ([IVXLCDM]+)\b', r'\1 \2')
    ]
    for pattern, replacement in date_patterns:
        text = re.sub(pattern, replacement, text)

    return text

def split_by_sentences(text):
    doc = nlp(text)
    return [sent.text.strip() for sent in doc.sents if len(sent.text.strip()) > 10]

def trim_overlap(chunk_sentences, max_words):
    trimmed = []
    word_count = 0
    for sent in reversed(chunk_sentences):
        sent_words = len(sent.split())
        if word_count + sent_words > max_words:
            break
        trimmed.insert(0, sent)
        word_count += sent_words
    return trimmed

def split_historical_content(text):
    sentences = split_by_sentences(text)
    chunks = []
    current_chunk = []
    current_length = 0

    for sent in sentences:
        sent_len = len(sent.split())
        if current_length + sent_len > CHUNK_SIZE:
            if current_chunk:
                chunk_text = ' '.join(current_chunk)
                if len(chunk_text.split()) >= MIN_CHUNK_LENGTH:
                    chunks.append(chunk_text)
                overlap_chunk = trim_overlap(current_chunk, OVERLAP_WORDS)
                current_chunk = overlap_chunk.copy()
                current_length = sum(len(s.split()) for s in current_chunk)

        current_chunk.append(sent)
        current_length += sent_len

    if current_chunk:
        final_chunk = ' '.join(current_chunk)
        if len(final_chunk.split()) >= MIN_CHUNK_LENGTH:
            chunks.append(final_chunk)

    return chunks

def extract_entities(chunk):
    doc = nlp(chunk)
    entities = {
        "persons": [],
        "locations": [],
        "dates": [],
        "organizations": [],
        "events": []
    }

    for ent in doc.ents:
        text = ent.text.strip()
        if not text or len(text) < 2:
            continue
        if ent.label_ == "PER" and len(text.split()) < 4:
            entities["persons"].append(text)
        elif ent.label_ == "LOC":
            entities["locations"].append(text)
        elif ent.label_ == "DATE":
            entities["dates"].append(text)
        elif ent.label_ == "ORG":
            entities["organizations"].append(text)
        elif ent.label_ == "EVENT":
            entities["events"].append(text)

    date_patterns = [
        r'\b\d{1,2} de [a-z]+ de \d{1,4}\b',
        r'\b(?:siglo|Siglo) [IVXLCDM]+\b',
        r'\b\d{1,4}\b',
        r'\b(?:años|año) \d{1,4}(?:-\d{1,4})?'
    ]
    for pattern in date_patterns:
        entities["dates"].extend(re.findall(pattern, chunk, flags=re.IGNORECASE))

    for key in entities:
        entities[key] = list(set(entities[key]))

    return entities

def process_articles():
    logging.info(f"Buscando archivo: {INPUT_FILE}")
    if not os.path.exists(INPUT_FILE):
        logging.error(f"Archivo no encontrado: {INPUT_FILE}")
        return

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        articles = json.load(f)

    logging.info(f"{len(articles)} artículos cargados para procesar.")

    processed_chunks = []

    for idx, article in enumerate(articles):
        title = article.get("title", "Sin título")
        logging.info(f"Procesando artículo {idx + 1}/{len(articles)}: {title}")

        content = clean_text(article.get("content", ""))
        chunks = split_historical_content(content)
        logging.info(f"Generados {len(chunks)} chunks para el artículo.")

        for i, chunk in enumerate(chunks):
            entities = extract_entities(chunk)
            processed_chunks.append({
                "title": title,
                "chunk_index": i,
                "content": chunk,
                "entities": entities,
                "word_count": len(chunk.split()),
                "char_count": len(chunk)
            })
        logging.info(f"Entidades extraídas para {len(chunks)} chunks.")

        # Guardar tras cada artículo
        try:
            with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
                json.dump(processed_chunks, f, ensure_ascii=False, indent=2)
            logging.info(f"Progreso guardado tras procesar el artículo: {title}")
        except Exception as e:
            logging.error(f"Error al guardar el archivo tras {title}: {e}")

    if not processed_chunks:
        logging.warning("No se generaron chunks procesados.")
    else:
        logging.info(f"Proceso finalizado. Total chunks: {len(processed_chunks)}")

if __name__ == "__main__":
    logging.info("🚀 Iniciando Preprocessor Agent...")
    process_articles()
    logging.info("✨ Proceso finalizado.")
