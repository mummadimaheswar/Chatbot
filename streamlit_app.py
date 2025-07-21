import streamlit as st
import requests

st.title("Chatbot UI")

user_input = st.text_input("You:", "")

if st.button("Send"):
    if user_input.strip():
        try:
            response = requests.get("http://localhost:8000/", timeout=10)
            if response.status_code == 200:
                st.write("Chatbot Response:", response.json()["message"])
            else:
                st.error(f"API Error: {response.status_code}")
        except Exception as e:
            st.error(f"Error contacting API: {e}")
