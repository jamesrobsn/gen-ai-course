import streamlit as st
from pathlib import Path
from langchain_community.agent_toolkits.sql.base import create_sql_agent
from langchain_community.utilities import SQLDatabase
from langchain.agents.agent_types import AgentType
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from sqlalchemy import create_engine
import sqlite3
from langchain_groq import ChatGroq

import os
from dotenv import load_dotenv

load_dotenv()
os.environ["USER_AGENT"] = "gen-ai-course-app"
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

st.set_page_config(page_title="SQL Agent", page_icon=":robot:")
st.title("SQL Agent")

    # Debug: Show actual contents of students table
with st.expander("Show students table (debug)"):
    db_file_path = (Path(__file__).parent / "student.db").absolute()

INJECTION_WARNING = """
SQL agent can be vulnerable to SQL injection attacks.
Use a DB role with limited permissions to mitigate this risk.
"""

LOCALDB = "USE_LOCALDB"
MYSQL = "USE_MYSQL"

try:
    conn = sqlite3.connect(db_file_path)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM students;")
    rows = cursor.fetchall()
    st.write(f"Database path: {db_file_path}")
    st.write("Rows in students table:")
    st.dataframe(rows)
    cursor.close()
    conn.close()
except Exception as e:
    st.error(f"Error reading students table: {e}")

radio_opt = ["Use SQLite - student.db", "Use MySQL"]
db_choice = st.sidebar.radio("Select Database", radio_opt)

if radio_opt.index(db_choice) == 1:
    db_uri = MYSQL
    mysql_host = st.sidebar.text_input("MySQL Host", "localhost")
    mysql_user = st.sidebar.text_input("MySQL User", "root")
    mysql_password = st.sidebar.text_input("MySQL Password", "", type="password")
    mysql_db = st.sidebar.text_input("MySQL Database", "gen_ai_course")
    db_uri = f"mysql+pymysql://{mysql_user}:{mysql_password}@{mysql_host}/{mysql_db}"
else:
    db_uri = LOCALDB

if not db_uri:
    st.info("Please select a database.")

if not GROQ_API_KEY:
    api_key = st.sidebar.text_input("Groq API Key", type="password")

    if not api_key:
        st.info("Please enter your Groq API Key.")

llm = ChatGroq(
    model_name="Llama3-8b-8192",
    groq_api_key=GROQ_API_KEY,
    streaming=True
)

@st.cache_resource(ttl="2h")
def configure_db(db_uri, mysql_host=None, mysql_user=None, mysql_password=None, mysql_db=None):
    if db_uri == LOCALDB:
        db_file_path = (Path(__file__).parent / "student.db").absolute()
        print(db_file_path)
        creator = lambda: sqlite3.connect(f"file:{db_file_path}?mode=ro", uri=True)
        return SQLDatabase(create_engine("sqlite://", creator=creator))
    elif db_uri == MYSQL:
        if not mysql_host or not mysql_user or not mysql_password or not mysql_db:
            st.error("Please provide all MySQL connection details.")
            st.stop()
        mysql_uri = f"mysql+mysqlconnector://{mysql_user}:{mysql_password}@{mysql_host}/{mysql_db}"
        return SQLDatabase(create_engine(mysql_uri))

if db_uri == MYSQL:
    db = configure_db(db_uri, mysql_host, mysql_user, mysql_password, mysql_db)
else:
    db = configure_db(db_uri)

# Toolkit for the SQL database
toolkit = SQLDatabaseToolkit(db=db, llm=llm)

# Create the SQL agent
agent = create_sql_agent(
    llm=llm,
    toolkit=toolkit,
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    handle_parsing_errors=True,
    return_intermediate_steps=True
)

# Initialize chat history
if "messages" not in st.session_state or st.sidebar.button("Reset Chat"):
    st.session_state["messages"] = [
        {"role": "assistant", "content": "How can I help you?"}
    ]

# Display chat history
for msg in st.session_state["messages"]:
    st.chat_message(msg["role"]).write(msg["content"])

user_query = st.chat_input(placeholder="Ask a question from the database...")

if user_query:
    st.session_state["messages"].append({"role": "user", "content": user_query})
    st.chat_message("user").write(user_query)

    with st.chat_message("assistant"):
        st_callback = StreamlitCallbackHandler(st.container())
        try:
            response = agent.run(user_query, callbacks=[st_callback])
            # If return_intermediate_steps=True, response is a dict
            if isinstance(response, dict) and "output" in response:
                final_output = response["output"]
                st.session_state["messages"].append({"role": "assistant", "content": final_output})
                st.write(final_output)
                # Optionally show intermediate steps for debugging
                if "intermediate_steps" in response:
                    with st.expander("Show intermediate steps"):
                        st.write(response["intermediate_steps"])
            else:
                st.session_state["messages"].append({"role": "assistant", "content": response})
                st.write(response)
        except Exception as e:
            st.session_state["messages"].append({"role": "assistant", "content": f"Error: {e}"})
            st.error(f"Agent error: {e}")