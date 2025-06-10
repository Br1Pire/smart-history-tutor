import os
import sys
import pickle
import faiss
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Rutas base
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VECTOR_STORE_DIR = os.path.join(BASE_DIR, "data", "vectorstore_faiss")
FAISS_INDEX_PATH = os.path.join(VECTOR_STORE_DIR, "faiss_index.index")
TEXTS_PATH = os.path.join(VECTOR_STORE_DIR, "texts.pkl")
GEN_MODEL_PATH = os.path.join(BASE_DIR, "models", "tinyllama-1.1b-chat-v1.0")

# Parámetros
TOP_K = 5
MAX_TOKENS = 300  # Puedes ajustar este valor

# Cargar FAISS
print("📂 Cargando índice vectorial y textos...")
index = faiss.read_index(FAISS_INDEX_PATH)
with open(TEXTS_PATH, "rb") as f:
    texts = pickle.load(f)

# Cargar modelo generativo
print("⚙️ Cargando modelo generativo...")
tokenizer = AutoTokenizer.from_pretrained(GEN_MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(GEN_MODEL_PATH)
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
print("✅ Modelo listo.\n")

# Modelo de embeddings
from sentence_transformers import SentenceTransformer
embedding_model = SentenceTransformer(os.path.join(BASE_DIR, "models", "all-mpnet-base-v2"))

# Recuperar contexto
def retrieve_context(query, k=TOP_K):
    query_vector = embedding_model.encode([query]).astype("float32")
    query_vector /= (query_vector**2).sum(axis=1, keepdims=True)**0.5
    distances, indices = index.search(query_vector, k)

    chunks = [texts[i] for i in indices[0]]
    print("\n🔎 Contexto recuperado:")
    for i, (c, d) in enumerate(zip(chunks, distances[0]), 1):
        print(f"\n--- Chunk {i} (distancia: {d:.4f}) ---\n{c[:400]}...\n")
    return chunks

# Construir prompt e inferir respuesta
def generate_answer(question, context_chunks):
    context = "\n".join(context_chunks[:TOP_K])
    prompt = f"""Contesta la siguiente pregunta de historia usando la información provista.

### Contexto:
{context}

### Pregunta:
{question}

### Respuesta:"""

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=MAX_TOKENS,
        do_sample=True,
        top_k=50,
        top_p=0.95,
        temperature=0.7
    )
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return answer.split("### Respuesta:")[-1].strip()

# Loop interactivo
while True:
    print("🤔 Escribe tu pregunta sobre historia:")
    question = input(" > ").strip()
    if not question:
        break

    context = retrieve_context(question)
    answer = generate_answer(question, context)
    print("\n🧠 Respuesta generada:")
    print(answer + "\n")
