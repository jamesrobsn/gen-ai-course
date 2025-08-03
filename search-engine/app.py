import streamlit as st
from langchain_groq import ChatGroq

from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun, DuckDuckGoSearchRun
from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper

from langchain.agents import create_react_agent
from langchain.prompts import PromptTemplate

from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

import os
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import openai

load_dotenv()
os.environ["USER_AGENT"] = "gen-ai-course-app"

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

llm = ChatGroq(
    model_name="Llama3-8b-8192",
    groq_api_key=GROQ_API_KEY,
    streaming=True
)

api_wrapper_wikipedia = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=250)
wiki_tool = WikipediaQueryRun(api_wrapper=api_wrapper_wikipedia)

api_wrapper_arxiv = ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=250)
arxiv_tool = ArxivQueryRun(api_wrapper=api_wrapper_arxiv)

search = DuckDuckGoSearchRun(name="Search")

tools = [wiki_tool, arxiv_tool, search]


agent_prompt = PromptTemplate(
    input_variables=["input", "agent_scratchpad", "tools", "tool_names"],
    template=(
        "You are an assistant. Answer the following question:\n"
        "{input}\n"
        "Available tools: {tools}\n"
        "Tool names: {tool_names}\n"
        "{agent_scratchpad}"
    )
)

# Initialize chat history

if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "Hi, I'm a chatbot who can search the web, Wikipedia, and Arxiv for you. Ask me anything!"}
    ]

# Display chat history
for msg in st.session_state["messages"]:
    st.chat_message(msg["role"]).write(msg["content"])

# Chat input

from langchain.agents import initialize_agent, AgentType
from langchain.callbacks import StreamlitCallbackHandler

if prompt := st.chat_input(placeholder="What is machine learning?"):
    st.session_state["messages"].append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            agent_executor = initialize_agent(
                tools,
                llm,
                agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                handle_parsing_errors=True
            )
            st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
            response = agent_executor.run(prompt, callbacks=[st_cb])
            st.session_state["messages"].append({"role": "assistant", "content": response})
            st.write(response)