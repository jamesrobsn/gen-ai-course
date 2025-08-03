import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
import streamlit as st

from langchain.chains import LLMMathChain
from langchain.prompts import PromptTemplate
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.agents.agent_types import AgentType
from langchain.agents import create_react_agent, AgentExecutor, Tool
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler
from langchain import hub

from dotenv import load_dotenv

# 1. Load environment and set user agent
load_dotenv()
os.environ["USER_AGENT"] = "gen-ai-course-app"
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

# 2. Prompt for API key if missing
if not GROQ_API_KEY:
    GROQ_API_KEY = st.sidebar.text_input("Groq API Key", type="password")
    if not GROQ_API_KEY:
        st.info("Please enter your Groq API Key.")

# 3. Initialize the Groq LLM
from langchain_groq import ChatGroq
llm = ChatGroq(
    model_name="Llama3-8b-8192",
    groq_api_key=GROQ_API_KEY,
    streaming=True
)

# Set up the Stramlit app
st.set_page_config(page_title="Math GPT", page_icon="🤖")
st.title("Math GPT")

# Initialize Wikipedia API wrapper
wiki = WikipediaAPIWrapper()
wiki_tool = Tool(
    name="Wikipedia",
    func=wiki.run,
    description="Useful for answering questions about general knowledge."
)


# Enhanced safe calculator tool
import math
def safe_calculator(expression: str):
    """Evaluate a math expression safely with better error handling."""
    # Clean the expression
    expression = expression.strip()
    
    # Allow common math operations and constants
    allowed_names = {k: v for k, v in math.__dict__.items() if not k.startswith("__")}
    allowed_names.update({
        "abs": abs,
        "round": round,
        "sum": sum,
        "max": max,
        "min": min
    })
    
    try:
        result = eval(expression, {"__builtins__": {}}, allowed_names)
        return f"{expression} = {result}"
    except Exception as e:
        return f"Error calculating '{expression}': {e}"

calculator = Tool(
    name="Calculator",
    func=safe_calculator,
    description="Useful for performing mathematical calculations. Input should be a valid Python math expression. Returns the calculation and result."
)

prompt = """
You are a mathematical reasoning assistant. Break down the problem step by step and provide a clear solution path.

For the question below, identify all the steps needed and provide the final calculation:

Question: {question}

Think through this step by step:
1. Identify what you start with
2. Track what changes (eaten, given away, bought)
3. Calculate the final total

Provide a clear, logical solution:
"""

prompt_template = PromptTemplate(
    input_variables=["question"],
    template=prompt
)

# Combine all the tools into chain
chain = prompt_template | llm

def reasoning_func(question):
    response = chain.invoke({"question": question})
    return response.content if hasattr(response, 'content') else str(response)

reasoning_tool = Tool(
    name="Reasoning",
    func=reasoning_func,
    description="Useful for breaking down complex word problems step by step. Use this first to understand the problem structure."
)

# Initialize the agents
react_prompt = hub.pull("hwchase17/react")

# Create a custom system message to guide the agent
system_message = """
You are a helpful math assistant. When solving word problems:

1. First use the Reasoning tool to understand the problem and break it down
2. Then use the Calculator tool for any arithmetic needed
3. Provide a clear final answer

For multi-step problems, work through them systematically without repeating calculations.
"""

assistant_agent = create_react_agent(
    llm=llm,
    tools=[wiki_tool, calculator, reasoning_tool],
    prompt=react_prompt
)

agent_executor = AgentExecutor(
    agent=assistant_agent,
    tools=[wiki_tool, calculator, reasoning_tool],
    verbose=True,
    handle_parsing_errors=True,
    max_iterations=10,  # Limit iterations to prevent infinite loops
    early_stopping_method="generate"  # Stop when a final answer is generated
)

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hi, I am a math chatbot that can answer all your math questions."}
    ]


for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# Function to generate the response
def generate_response(question):
    try:
        response = agent_executor.invoke({
            "input": question})
        return response["output"]
    except Exception as e:
        st.error(f"Error: {e}")
        return f"Error: {e}"

# Start the interaction
question = st.text_area("Enter your question:","I have 5 bananas and 7 grapes. I eat 2 bananas and give away 3 grapes. Then I buy a dozen apples and 2 packs of blueberries. Each pack of blueberries contains 25 berries. How many total pieces of fruit do I have at the end?")

if st.button("Find My Answer"):
    if question:
        with st.spinner("Generate response.."):
            st.session_state.messages.append({"role":"user","content":question})
            st.chat_message("user").write(question)

            st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
            try:
                response = agent_executor.invoke(
                    {"input": question},
                    {"callbacks": [st_cb]}
                )
                answer = response["output"]
                st.session_state.messages.append({'role':'assistant',"content":answer})
                st.write('### Response:')
                st.success(answer)
            except Exception as e:
                error_msg = f"Error: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append({'role':'assistant',"content":error_msg})

    else:
        st.warning("Please enter the question")
