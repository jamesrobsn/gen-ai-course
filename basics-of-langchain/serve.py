import os
from fastapi import FastAPI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langserve import add_routes
from dotenv import load_dotenv

load_dotenv()

GROG_API_KEY = os.getenv("GROG_API_KEY")

app = FastAPI(title="LangChain Groq Example",
    description="A simple example of using LangChain with Groq for translation tasks.",
    version="0.1.0",
)

# Initialize the Groq model
model = ChatGroq(
    model_name=os.getenv("MODEL_NAME", "gemma2-9b-it"),
    api_key=GROG_API_KEY
)

# 1. Create a prompt template
system_template = "Translate the following into {language}"
prompt_template = ChatPromptTemplate.from_messages([
    ("system", system_template),
    ("user", "{text}")
])

# 2. Create an output parser
output_parser = StrOutputParser()

# 3. Create a chain using LCEL
chain = prompt_template | model | output_parser

# 4. Add routes to the FastAPI app
add_routes(app, chain, path="/chain")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)