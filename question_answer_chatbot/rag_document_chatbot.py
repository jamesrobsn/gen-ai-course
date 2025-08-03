import streamlit as st
from langchain_groq import ChatGroq
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader

import os
from dotenv import load_dotenv

load_dotenv()

# Langsmith Tracing
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_PROJECT"] = "Q&A Chatbot with Groq"

os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

llm = ChatGroq(
    model_name="gemma2-9b-it",
    api_key=os.getenv("GROQ_API_KEY"),
)

# Use from_template for a single prompt string
prompt = ChatPromptTemplate.from_template(
    """
    Answer the question based on the context provided. If the answer is not in the context, say "I don't know".
    Please provide the most accurate response based on the question.
    <context>
    {context}
    </context>
    Question: {input}
    """
)

def create_vector_embeddings():
    if "vectors" not in st.session_state:
        # st.session_state.embeddings = OllamaEmbeddings(model="llama3")
        st.session_state.embeddings = OpenAIEmbeddings()
        # Use absolute path for robustness
        pdf_dir = os.path.join(os.path.dirname(__file__), "research_papers")
        st.session_state.loader = PyPDFDirectoryLoader(pdf_dir)
        st.session_state.documents = st.session_state.loader.load()
        doc_count = len(st.session_state.documents)
        if not st.session_state.documents:
            st.warning(f"No documents found in '{pdf_dir}'. Please add PDF files.")
            return
        st.info(f"Loaded {doc_count} document(s) from '{pdf_dir}'.")
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        st.session_state.split_docs = st.session_state.text_splitter.split_documents(st.session_state.documents)
        if not st.session_state.split_docs:
            st.warning("No text chunks created from documents. Check your PDFs.")
            return
        st.session_state.vectors = FAISS.from_documents(
            st.session_state.split_docs,
            st.session_state.embeddings
        )

user_prompt = st.text_input("Enter your question from the research paper:")

if st.button("Document Embeddings", disabled="vectors" in st.session_state):
    create_vector_embeddings()
    st.success("Document embeddings created successfully!")

import time

# Only run if embeddings are created
if user_prompt:
    if "vectors" not in st.session_state:
        st.warning("Please create document embeddings first!")
    else:
        document_chain = create_stuff_documents_chain(llm, prompt)
        retriever = st.session_state.vectors.as_retriever()
        retrieval_chain = create_retrieval_chain(
            retriever=retriever,
            combine_docs_chain=document_chain
        )

        start = time.process_time()
        response = retrieval_chain.invoke({"input": user_prompt})
        end = time.process_time()
        # Display only the answer
        answer = response.get("answer", response)
        st.write(f"Answer: {answer}")
        st.write(f"Time taken: {end - start} seconds")

        # With streamlit expander
        with st.expander("Show Context"):
            context = response.get("context", [])
            for i, doc in enumerate(context):
                st.write(f"Document {i + 1}:")
                st.write(doc.page_content)