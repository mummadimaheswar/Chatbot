# frontend/app.py
import streamlit as st
import requests

st.title("Chatbot Demo")
user_input = st.text_input("You:")

if st.button("Send"):
    res = requests.post("http://localhost:8000/chat", json={"text": user_input})
    if res.status_code == 200:
        st.write("Bot:", res.json()["response"])
    else:
        st.write("Error connecting to API")
