import streamlit as st
import pandas as pd
import sqlite3
import openai
import cohere
import streamlit as st
import pandas as pd
import sqlite3
from datetime import datetime, timedelta
import pytz
import hashlib
import matplotlib.pyplot as plt
import plotly.express as px
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from openai import OpenAI

DB_NAME = "inventory.db"
lima_tz = pytz.timezone("America/Lima")

# ---------- DATABASE SETUP ----------
def create_tables():
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()

    c.execute("""
    CREATE TABLE IF NOT EXISTS users (
        username TEXT PRIMARY KEY,
        password TEXT NOT NULL
    )
    """)

    c.execute("""
    CREATE TABLE IF NOT EXISTS inventory (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp_in TEXT,
        timestamp_out TEXT,
        product_name TEXT,
        batch_id TEXT,
        stock_in INTEGER,
        stock_out INTEGER,
        total_stock INTEGER,
        unit_price REAL,
        quantity INTEGER,
        total_price REAL,
        total_units INTEGER,
        expiration_date TEXT,
        username TEXT
    )
    """)
    conn.commit()
    conn.close()

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def validate_login(username, password):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("SELECT password FROM users WHERE username = ?", (username,))
    row = c.fetchone()
    conn.close()
    return row and row[0] == hash_password(password)

def add_user(username, password):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, hash_password(password)))
    conn.commit()
    conn.close()

def insert_inventory(data):
    conn = sqlite3.connect(DB_NAME)
    df = pd.DataFrame([data])
    df.to_sql("inventory", conn, if_exists="append", index=False)
    conn.close()

def load_inventory(username):
    conn = sqlite3.connect(DB_NAME)
    df = pd.read_sql("SELECT * FROM inventory WHERE username = ?", conn, params=(username,))
    conn.close()
    return df

# ---------- NATURAL LANGUAGE TO SQL ----------
def generate_sql_query(user_question, username):
    schema = """
    Table: inventory
    Columns:
      - id (INTEGER)
      - timestamp_in (TEXT)
      - timestamp_out (TEXT)
      - product_name (TEXT)
      - batch_id (TEXT)
      - stock_in (INTEGER)
      - stock_out (INTEGER)
      - total_stock (INTEGER)
      - unit_price (REAL)
      - quantity (INTEGER)
      - total_price (REAL)
      - total_units (INTEGER)
      - expiration_date (TEXT)
      - username (TEXT)
    """
    prompt = f"""
    You are a SQL assistant. Translate the following user input into a valid SQL query for a SQLite database.

    {schema}

    Always include a WHERE clause filtering by username = '{username}'.
    If the input is already SQL, keep it as is but ensure it includes WHERE username = '{username}'.

    Input: {user_question}

    SQL:
    """
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        max_tokens=150,
        temperature=0
    )
    sql_query = response.choices[0].text.strip()
    return sql_query

def execute_sql_query(sql_query):
    conn = sqlite3.connect(DB_NAME)
    try:
        df = pd.read_sql_query(sql_query, conn)
    except Exception as e:
        df = pd.DataFrame()
    conn.close()
    return df

def generate_response(user_question, query_results):
    context = query_results.to_string(index=False)
    prompt = f"Based on the following data:\n{context}\n\nAnswer the question: {user_question}"
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        max_tokens=150,
        temperature=0
    )
    answer = response.choices[0].text.strip()
    return answer

# ---------- CHATBOT FOR INVENTORY QUESTIONS ----------
st.subheader("Ask Inventory Questions in Natural Language or SQL")
user_question = st.text_input("Ask a question or write a SQL query about your inventory:")

# Display product name suggestions to guide users
if 'username' in st.session_state:
    user_inventory = load_inventory(st.session_state['username'])
    if not user_inventory.empty:
        st.caption("Available product names in your inventory:")
        st.markdown(", ".join(sorted(user_inventory['product_name'].unique())))

if user_question and 'username' in st.session_state:
    sql_query = generate_sql_query(user_question, st.session_state['username'])
    st.code(sql_query, language='sql')  # Show the generated SQL for transparency
    query_results = execute_sql_query(sql_query)
    if not query_results.empty:
        st.success("✅ Query executed successfully")
        st.dataframe(query_results, use_container_width=True)
        answer = generate_response(user_question, query_results)
        st.info(f"💬 Assistant Response: {answer}")
    else:
        st.warning("⚠️ I'm sorry, I couldn't find any data matching your query.")
