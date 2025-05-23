import streamlit as st
import uuid
import requests


st.title("💬 Chatbot")
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]
if 'session_id' not in st.session_state:
    st.session_state["session_id"] = str(uuid.uuid4())


for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])


if prompt := st.chat_input():
    url = 'http://127.0.0.1:8001/chat'
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)
    response = requests.get(url, json={"user_id": 1, "session_id": st.session_state["session_id"], "message": prompt})
    msg = response.json()["output"]
    st.session_state.messages.append({"role": "assistant", "content": msg})
    st.chat_message("assistant").write(msg)
