import streamlit as st
from connection import DatabaseManager
from my_agent import SmartBusinessAssistant
import openai
import os
db=DatabaseManager.get_db()
agent = SmartBusinessAssistant(db)

st.title("Smart Business Assistant")

user_input = st.text_area("Enter your business query here:")
if st.button("Get Answer"):
    if user_input.strip() == "":
        st.warning("Please enter a valid query.")
    else:
        with st.spinner("Processing your query..."):
            response = agent.smart_run(user_input)
        st.subheader("Response:")
        st.write(response)
