import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)
import streamlit as st
from src.core.tutor_builder import create_tutor_instance

@st.cache_resource
def get_tutor_instance():
    """Crea y cachea la instancia del Tutor."""
    return create_tutor_instance()

tutor = get_tutor_instance()

st.set_page_config(page_title="History Smart Tutor", page_icon="📜")

st.title("📜 History Smart Tutor")

# Inicializa historial
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

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
        result = tutor.answer_question(user_input)

    st.session_state.chat_history[-1]["answer"] = result["answer"]
    st.session_state.chat_history[-1]["strategy"] = result["strategy"]
    st.session_state.chat_history[-1]["tokens"] = result["tokens_used"]

    with st.chat_message("assistant"):
        st.markdown(result["answer"])
        if result["strategy"] != "utility_action":
            st.markdown(f"_Strategy: {result['strategy']} | Tokens used: {result['tokens_used']}_")

    st.rerun()