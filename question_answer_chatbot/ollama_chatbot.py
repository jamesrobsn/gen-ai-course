import streamlit as st
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.llms import Ollama
import os
from dotenv import load_dotenv

load_dotenv()

# Langsmith Tracing
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_PROJECT"] = "Q&A Chatbot with Ollama"

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Please respond to the user queries."),
    ("human", "Question: {question}"),
])

def generate_answer(question, model="llama3"):
    # Initialize the OpenAI model
    model = Ollama(model=model)
    output_parser = StrOutputParser()
    chain = prompt | model | output_parser
    answer = chain.invoke({"question": question})

    return answer

# Title and description
st.title("Q&A Chatbot with Opensource models")
st.write("This chatbot uses opensource models to answer user questions.")

# Dropdown to select model
model = st.sidebar.selectbox(
    "Select Model",
    ["llama3", "gemma2", "mistral"],
    index=0
)

# Main interface for user question
st.write("Go ahead and ask any question:")
question = st.text_input("Enter your question:")
if st.button("Get Answer"):
    if question:
        with st.spinner("Generating answer..."):
            try:
                answer = generate_answer(question, model)
                st.success(f"Answer: {answer}")
            except Exception as e:
                st.error(f"Error: {str(e)}")
    else:
        st.warning("Please enter a question.")