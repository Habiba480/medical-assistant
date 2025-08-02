import streamlit as st
import uuid
from src.constants import SYSTEM_PROMPT

def render_sidebar():
    st.sidebar.title("💬 Saved Chats")

    for sid in st.session_state.chat_sessions:
        preview = "(Empty chat)"
        for msg in st.session_state.chat_sessions[sid]:
            if msg["role"] == "user":
                preview = (msg["content"][:30] + "...") if msg["content"] else "(Empty chat)"
                break
        if st.sidebar.button(preview, key=sid):
            st.session_state.current_session_id = sid

    if st.sidebar.button("➕ New Chat"):
        new_id = str(uuid.uuid4())
        st.session_state.current_session_id = new_id
        st.session_state.chat_sessions[new_id] = [
            {"role": "system", "content": SYSTEM_PROMPT}
        ]

def render_chat(messages):
    for msg in messages[1:]:  # skip system prompt
        role = msg["role"]
        if role == "user":
            st.chat_message("user").markdown(msg["content"])
        elif role == "assistant":
            st.chat_message("assistant").markdown(msg["content"])
