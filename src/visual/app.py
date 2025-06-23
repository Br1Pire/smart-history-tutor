# app.py
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
import streamlit as st
from src.agents.tutor_agent import tutor_session, crawl_titles, preprocess, vectorize

st.set_page_config(page_title="History Smart Tutor", page_icon="📜")

st.title("📜 History Smart Tutor")

# Inicializa historial
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# BOTONES DE UTILIDAD
col1, col2, col3 = st.columns(3)
if col1.button("🚀 Crawler dinámico"):
    with st.spinner("Running crawler..."):
        try:
            crawl_titles()
            response = "✅ Crawler dinámico ejecutado correctamente."
        except Exception as e:
            response = f"❌ Error al ejecutar crawler: {e}"
    st.session_state.chat_history.append({
        "question": "Crawler dinámico solicitado",
        "answer": response,
        "strategy": "utility_action",
        "tokens": 0
    })

if col2.button("⚙️ Postprocesador"):
    with st.spinner("Running postprocessor..."):
        try:
            preprocess()
            response = "✅ Postprocesador ejecutado correctamente."
        except Exception as e:
            response = f"❌ Error al ejecutar postprocesador: {e}"
    st.session_state.chat_history.append({
        "question": "Postprocesador solicitado",
        "answer": response,
        "strategy": "utility_action",
        "tokens": 0
    })

if col3.button("📈 Vectorización"):
    with st.spinner("Running vectorizer..."):
        try:
            vectorize()
            response = "✅ Vectorización ejecutada correctamente."
        except Exception as e:
            response = f"❌ Error al ejecutar vectorización: {e}"
    st.session_state.chat_history.append({
        "question": "Vectorización solicitada",
        "answer": response,
        "strategy": "utility_action",
        "tokens": 0
    })

# Muestra el historial
for entry in st.session_state.chat_history:
    with st.chat_message("user"):
        st.markdown(entry["question"])
    with st.chat_message("assistant"):
        st.markdown(entry["answer"])
        if entry["strategy"] != "utility_action":
            st.markdown(f"_Strategy: {entry['strategy']} | Tokens used: {entry['tokens']}_")

# Barra de chat
user_input = st.chat_input("Pregúntame lo que desees sobre historia!")

if user_input:
    st.session_state.chat_history.append({
        "question": user_input,
        "answer": None,
        "strategy": None,
        "tokens": None
    })

    with st.chat_message("user"):
        st.markdown(user_input)

    with st.spinner("Thinking..."):
        result = tutor_session(user_input)

    st.session_state.chat_history[-1]["answer"] = result["answer"]
    st.session_state.chat_history[-1]["strategy"] = result["strategy"]
    st.session_state.chat_history[-1]["tokens"] = result["tokens_used"]

    with st.chat_message("assistant"):
        st.markdown(result["answer"])
        st.markdown(f"_Strategy: {result['strategy']} | Tokens used: {result['tokens_used']}_")
