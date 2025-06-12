import os
import json
import re
import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer

# Rutas
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "all-mpnet-base-v2")
INPUT_FILE = os.path.join(BASE_DIR, "data", "raw", "specific_wiki_articles.json")
OUTPUT_FILE = os.path.join(BASE_DIR, "data", "processed", "wiki_articles_segmented.json")

VECTOR_STORE_DIR = os.path.join(BASE_DIR, "data", "vectorstore_faiss")
FAISS_INDEX_PATH = os.path.join(VECTOR_STORE_DIR, "faiss_index.index")
TEXTS_PATH = os.path.join(VECTOR_STORE_DIR, "texts.pkl")
IDS_PATH = os.path.join(VECTOR_STORE_DIR, "ids.pkl")

# Parámetros de segmentación
CHUNK_SIZE = 300
OVERLAP_WORDS = 50
OVERLAP_SENTENCES = 1

# --- LIMPIEZA DE TEXTO ---
def clean_text(text):
    text = re.sub(r'\n+', ' ', text)
    text = re.sub(r'={2,}\s?.+?\s?={2,}', '', text)
    text = re.sub(r'\[\d+\]', '', text)
    return re.sub(r'\s+', ' ', text).strip()

# --- SEGMENTACIÓN POR PALABRAS (versión original) ---
def split_text_by_words(text, chunk_size=CHUNK_SIZE, overlap=OVERLAP_WORDS):
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += chunk_size - overlap
    return chunks

# --- SEGMENTACIÓN POR ORACIONES (nueva versión por defecto) ---
def split_text_by_sentences(text, chunk_size=CHUNK_SIZE, overlap=OVERLAP_SENTENCES):
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    chunks = []
    current_chunk = []
    current_length = 0

    for sentence in sentences:
        sentence_words = sentence.split()
        sentence_len = len(sentence_words)

        if current_length + sentence_len > chunk_size:
            if current_chunk:
                chunks.append(" ".join(current_chunk))
                current_chunk = current_chunk[-overlap:] if overlap > 0 else []
                current_length = sum(len(s.split()) for s in current_chunk)

        current_chunk.append(sentence)
        current_length += sentence_len

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks

# --- SEGMENTAR ARTÍCULO ---
def segment_article(article, use_sentences=True, save_path=OUTPUT_FILE):
    title = article.get("title", "Untitled")

    # Cargar chunks existentes si hay ruta de guardado
    existing_chunks = []
    if save_path and os.path.exists(save_path):
        with open(save_path, "r", encoding="utf-8") as f:
            existing_chunks = json.load(f)
        if any(c['title'] == title for c in existing_chunks):
            print(f"⏭️ Skipping '{title}': already in segment file.")
            return []

    # Si no estaba, proceder a segmentar
    content = clean_text(article.get("content", ""))
    chunks_raw = (
        split_text_by_sentences(content) if use_sentences
        else split_text_by_words(content)
    )

    new_chunks = [
        {"title": title, "chunk_index": i, "content": chunk}
        for i, chunk in enumerate(chunks_raw)
    ]

    # Guardar si corresponde
    if save_path:
        combined = existing_chunks + new_chunks
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(combined, f, ensure_ascii=False, indent=2)
        print(f"📁 Added {len(new_chunks)} chunks from '{title}' to '{save_path}'")

    return new_chunks



# --- SEGMENTAR ARCHIVO JSON COMPLETO ---
def segment_file(json_path, use_sentences=True, save_path=OUTPUT_FILE):
    with open(json_path, "r", encoding="utf-8") as f:
        articles = json.load(f)

    # Cargar los ya segmentados
    existing_chunks = []
    existing_titles = set()
    if save_path and os.path.exists(save_path):
        with open(save_path, "r", encoding="utf-8") as f:
            existing_chunks = json.load(f)
        existing_titles = {chunk['title'] for chunk in existing_chunks}

    all_new_chunks = []

    for article in articles:
        title = article.get("title", "Untitled")
        if title in existing_titles:
            print(f"⏭️ Skipping '{title}' (already in segment file)")
            continue

        content = clean_text(article.get("content", ""))
        chunks_raw = (
            split_text_by_sentences(content) if use_sentences
            else split_text_by_words(content)
        )
        new_chunks = [
            {"title": title, "chunk_index": i, "content": chunk}
            for i, chunk in enumerate(chunks_raw)
        ]
        all_new_chunks.extend(new_chunks)

    if save_path and all_new_chunks:
        combined = existing_chunks + all_new_chunks
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(combined, f, ensure_ascii=False, indent=2)
        print(f"📁 Added {len(all_new_chunks)} new chunks to '{save_path}'")

    return all_new_chunks



# --- CREAR EMBEDDINGS ---
def create_embeddings(model, documents):
    texts = [f"{doc['title']}. {doc['content']}" for doc in documents]
    ids = [f"{doc['title']}__chunk_{doc['chunk_index']}" for doc in documents]
    embeddings = model.encode(texts, show_progress_bar=True, batch_size=32)
    embeddings = np.array(embeddings).astype("float32")
    embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)
    return ids, texts, embeddings

# --- GUARDAR EN FAISS ---
def save_to_faiss(ids, texts, embeddings):
    os.makedirs(VECTOR_STORE_DIR, exist_ok=True)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    faiss.write_index(index, FAISS_INDEX_PATH)

    with open(TEXTS_PATH, "wb") as f:
        pickle.dump(texts, f)
    with open(IDS_PATH, "wb") as f:
        pickle.dump(ids, f)

    print(f"\n✅ Stored {len(ids)} embeddings into FAISS index.")

# --- PROCESAR CHUNKS ---
def process_chunks(chunks):
    print("📦 Loading local model...")
    model = SentenceTransformer(MODEL_PATH)
    print("🧠 Creating embeddings...")
    ids, texts, embeddings = create_embeddings(model, chunks)
    print("💾 Saving into FAISS...")
    save_to_faiss(ids, texts, embeddings)

# --- PROCESAR UN SOLO ARTÍCULO ---
def process_single_article(article, use_sentences=True):
    print(f"\n🔹 Processing single article: {article['title']}")
    chunks = segment_article(article, use_sentences=use_sentences)
    process_chunks(chunks)

# --- PROCESAR ARCHIVO JSON COMPLETO ---
def process_file(json_path = INPUT_FILE, use_sentences=True):
    print(f"\n📂 Processing file: {json_path}")
    chunks = segment_file(json_path, use_sentences=use_sentences)
    process_chunks(chunks)

# --- USO DESDE TERMINAL ---
if __name__ == "__main__":
    import sys
    if len(sys.argv) == 2:
        process_file(sys.argv[1])
    else:
        print("ℹ️ Usage: python vectorizer_agent.py path_to_json_file")
