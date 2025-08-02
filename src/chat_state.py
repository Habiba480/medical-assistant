import streamlit as st
import uuid
from src.constants import SYSTEM_PROMPT

def initialize_chat():
    if "chat_sessions" not in st.session_state:
        st.session_state.chat_sessions = {}

    if "current_session_id" not in st.session_state:
        new_id = str(uuid.uuid4())
        st.session_state.current_session_id = new_id
        st.session_state.chat_sessions[new_id] = [
            {"role": "system", "content": SYSTEM_PROMPT}
        ]

def get_current_session():
    session_id = st.session_state.current_session_id
    return session_id, st.session_state.chat_sessions[session_id]
