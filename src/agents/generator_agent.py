import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Rutas
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "tinyllama-1.1b-chat-v1.0")  # ajusta al modelo que usas

# Parámetros de generación
MAX_TOKENS = 300
TEMPERATURE = 0.7
TOP_K = 50
TOP_P = 0.95

# Carga del modelo generativo
print("⚙️ Loading generative model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH)
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
print("✅ Generative model ready.\n")

# 🔁 Función genérica de generación
def generate_text(prompt, max_tokens=MAX_TOKENS):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_tokens,
        do_sample=True,
        temperature=TEMPERATURE,
        top_k=TOP_K,
        top_p=TOP_P
    )
    output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return output_text.split("### Respuesta:")[-1].strip()

# ✨ Función común: responder con contexto
def generate_answer(query, context_chunks):
    context = "\n".join(context_chunks)
    prompt = f"""Contesta la siguiente pregunta de historia usando la información provista.

### Contexto:
{context}

### Pregunta:
{query}

### Respuesta:"""
    return generate_text(prompt)

# Test desde consola
if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
        ctx = ["La Revolución francesa comenzó en 1789...", "Una de las causas fue la desigualdad social."]
        result = generate_answer(query, ctx)
        print("\n🧠 Generated Answer:\n", result)
    else:
        print("ℹ️ Usage: python generator_agent.py 'Your question here'")
