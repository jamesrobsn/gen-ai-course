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

# Set up the Streamlit app
st.set_page_config(page_title="Math GPT Improved", page_icon="🧮")
st.title("Math GPT - Improved Version")

# Create a comprehensive math solver that handles the problem in one go
def comprehensive_math_solver(question: str):
    """Solve math word problems comprehensively without loops."""
    prompt = f"""
    Solve this math word problem step by step:

    {question}

    Please:
    1. Break down the problem into clear steps
    2. Show all calculations
    3. Provide the final answer

    Important: As soon as you have calculated and stated the final answer, stop. Do not repeat calculations or continue with further thoughts.
    Format your response clearly with numbered steps and finish with 'Final Answer: ...'
    """
    
    try:
        response = llm.invoke(prompt)
        output = response.content if hasattr(response, 'content') else str(response)
        # Ensure output ends with 'Final Answer: ...'
        if 'Final Answer' not in output:
            # Try to extract the last number as the answer
            import re
            numbers = re.findall(r"\d+", output)
            if numbers:
                output = output.strip() + f"\nFinal Answer: {numbers[-1]}"
        return output
    except Exception as e:
        return f"Error: {e}"

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

# Initialize Wikipedia API wrapper
wiki = WikipediaAPIWrapper()
wiki_tool = Tool(
    name="Wikipedia",
    func=wiki.run,
    description="Useful for answering questions about general knowledge."
)

calculator = Tool(
    name="Calculator",
    func=safe_calculator,
    description="Use this for individual calculations after breaking down the problem. Input should be a single mathematical expression."
)

math_solver = Tool(
    name="MathSolver",
    func=comprehensive_math_solver,
    description="Use this to solve complete word problems. This tool will break down the problem and solve it comprehensively."
)

# Initialize the agents with better configuration
react_prompt = hub.pull("hwchase17/react")

assistant_agent = create_react_agent(
    llm=llm,
    tools=[math_solver, calculator, wiki_tool],
    prompt=react_prompt
)

agent_executor = AgentExecutor(
    agent=assistant_agent,
    tools=[math_solver, calculator, wiki_tool],
    verbose=True,
    handle_parsing_errors=True,
    max_iterations=20,
    early_stopping_method="force"  # Stop as soon as a tool returns an answer
)

# Session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hi! I'm an improved math chatbot. I can solve word problems step by step without getting stuck in loops."}
    ]

# Display chat history
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# Function to generate response with better error handling
def generate_response(question):
    try:
        # First try the comprehensive solver directly
        if any(word in question.lower() for word in ['have', 'buy', 'eat', 'give', 'total', 'how many']):
            return comprehensive_math_solver(question)
        else:
            # For other questions, use the agent
            response = agent_executor.invoke({"input": question})
            return response["output"]
    except Exception as e:
        st.error(f"Error: {e}")
        return f"I encountered an error: {e}. Let me try a direct approach."

# User interface
question = st.text_area(
    "Enter your math question:",
    "I have 5 bananas and 7 grapes. I eat 2 bananas and give away 3 grapes. Then I buy a dozen apples and 2 packs of blueberries. Each pack of blueberries contains 25 berries. How many total pieces of fruit do I have at the end?"
)

col1, col2 = st.columns(2)

with col1:
    if st.button("Solve with Agent"):
        if question:
            with st.spinner("Solving with agent.."):
                st.session_state.messages.append({"role": "user", "content": question})
                st.chat_message("user").write(question)

                st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
                try:
                    response = agent_executor.invoke(
                        {"input": question},
                        {"callbacks": [st_cb]}
                    )
                    answer = response["output"]
                    st.session_state.messages.append({'role': 'assistant', "content": answer})
                    st.write('### Agent Response:')
                    st.success(answer)
                except Exception as e:
                    error_msg = f"Agent Error: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({'role': 'assistant', "content": error_msg})
        else:
            st.warning("Please enter a question")

with col2:
    if st.button("Solve Directly"):
        if question:
            with st.spinner("Solving directly.."):
                st.session_state.messages.append({"role": "user", "content": question})
                st.chat_message("user").write(question)

                try:
                    answer = comprehensive_math_solver(question)
                    st.session_state.messages.append({'role': 'assistant', "content": answer})
                    st.write('### Direct Solution:')
                    st.success(answer)
                except Exception as e:
                    error_msg = f"Direct Solver Error: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({'role': 'assistant', "content": error_msg})
        else:
            st.warning("Please enter a question")

# Clear chat button
if st.button("Clear Chat"):
    st.session_state.messages = [
        {"role": "assistant", "content": "Hi! I'm an improved math chatbot. I can solve word problems step by step without getting stuck in loops."}
    ]
    st.rerun()
