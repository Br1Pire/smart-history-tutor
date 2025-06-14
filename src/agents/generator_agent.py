import os
import google.generativeai as genai

# Configura tu clave API desde variable de entorno
GOOGLE_API_KEY="AIzaSyAAY_YacYAzOV-klmHA_uFjyFDSMrEFtDI"
genai.configure(api_key=GOOGLE_API_KEY)

# Modelo recomendado (precisión vs. tokens)
MODEL_NAME = "gemini-1.5-flash"
model = genai.GenerativeModel(MODEL_NAME)


def generate_answer(question: str, context_chunks: list[str]) -> str:
    """
    Genera una respuesta en español utilizando los fragmentos de contexto.
    """
    prompt = f"""
Eres un tutor experto en historia. Tu tarea es responder en español de forma clara, precisa y educativa
a la pregunta de un estudiante utilizando únicamente el siguiente contexto recuperado.

Si no encuentras la información en los fragmentos, puedes dar una explicación general, pero indícalo.

🔎 Pregunta del estudiante:
{question}

📚 Fragmentos de contexto:
{chr(10).join(f"- {chunk.strip()}" for chunk in context_chunks)}

✍️ Responde usando un lenguaje académico, pero fácil de entender. No inventes datos históricos.
"""

    try:
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"⚠️ Error generando respuesta: {e}"


# Modo prueba rápida desde consola
if __name__ == "__main__":
    sample_question = "¿Cuándo comenzó la Segunda Guerra Mundial?"
    sample_context = [
        "La invasión alemana a Polonia comenzó el 1 de septiembre de 1939.",
        "Francia y Reino Unido declararon la guerra a Alemania el 3 de septiembre de ese mismo año."
    ]
    print(generate_answer(sample_question, sample_context))
