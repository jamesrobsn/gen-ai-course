import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
import os
from dotenv import load_dotenv

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Langsmith Tracing
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_PROJECT"] = "Q&A Chatbot with OpenAI"

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Please respond to the user queries."),
    ("human", "Question: {question}"),
])

def generate_answer(question, api_key=OPENAI_API_KEY, model="gpt-4.1-mini", temperature=0.7, max_tokens=150):
    # Initialize the OpenAI model
    model = ChatOpenAI(
        model_name=model,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    output_parser = StrOutputParser()
    chain = prompt | model | output_parser
    answer = chain.invoke({"question": question})

    return answer

# Title and description
st.title("Q&A Chatbot with OpenAI")
st.write("This chatbot uses OpenAI models to answer user questions.")

# Sidebar for user input
api_key = st.sidebar.text_input("OpenAI API Key", type="password")

# Dropdown to select model
model = st.sidebar.selectbox(
    "Select Model",
    ["gpt-4o-mini", "gpt-4.1-mini", "gpt-4.1-nano"],  # Check these are valid model names for your OpenAI setup
    index=0
)

temperature = st.sidebar.slider("Temperature", 0.0, 1.0, 0.7)
max_tokens = st.sidebar.slider("Max Tokens", 50, 1000, 150)

# Main interface for user question
st.write("Go ahead and ask any question:")
question = st.text_input("Enter your question:")
if st.button("Get Answer"):
    if question:
        with st.spinner("Generating answer..."):
            try:
                answer = generate_answer(question, api_key, model, temperature, max_tokens)
                st.success(f"Answer: {answer}")
            except Exception as e:
                st.error(f"Error: {str(e)}")
    else:
        st.warning("Please enter a question.")