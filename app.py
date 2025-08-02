import streamlit as st
import uuid
from src.constants import SYSTEM_PROMPT
from src.chat_state import initialize_chat, get_current_session
from src.llm_client import query_llm
from src.ui import render_sidebar, render_chat

# Init session state
initialize_chat()

# Sidebar & session selector
render_sidebar()

# Title
st.title("🩺 Medical Assistant Chatbot")

# Load current chat
current_id, messages = get_current_session()

# Render chat
render_chat(messages)

# Input and LLM response
if prompt := st.chat_input("Enter your symptoms or questions..."):
    st.chat_message("user").markdown(prompt)
    messages.append({"role": "user", "content": prompt})

    reply = query_llm(messages)
    st.chat_message("assistant").markdown(reply)
    messages.append({"role": "assistant", "content": reply})
