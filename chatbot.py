import os
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
import streamlit as st 
from dotenv import load_dotenv

load_dotenv()

langchain_api_key = os.getenv('LANGCHAIN_API_KEY')
groq_api_key = os.getenv('GROQ_API_KEY')

# Define prompt 
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant"),
    ('user', "Question: {question}")
])

# Generate a response
def generate_response(question, engine, temperature, max_token):
    llm = ChatGroq(model=engine, temperature=temperature, max_tokens=max_token)  # Pass parameters here
    output_parser = StrOutputParser()
    chain = prompt | llm | output_parser
    answer = chain.invoke({"question": question})
    return answer

# Build the Streamlit app
st.title("QnA Chatbot")
engine = st.sidebar.selectbox("Select a model", ['gemma2-9b-it', 'llama3-groq-70b-8192-tool-use-preview', 'mixtral-8x7b-32768', 'llama-guard-3-8b'])
temperature = st.sidebar.slider('Temperature', min_value=0.0, max_value=1.0, value=0.5)
max_token = st.sidebar.slider("Maximum Tokens", min_value=50, max_value=600, value=60)
st.write("I'm K.A, Your assistant. What can I help you today?")
user_input = st.text_input("You: ")

if user_input:
    response = generate_response(user_input, engine, temperature, max_token)
    st.write(response)
else:
    st.error("Please drop your question in the field!")

# Footer
st.markdown("---")
st.markdown("Made by Kaletsidik Ayalew © 2024")
