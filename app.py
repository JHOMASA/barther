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

# ---------- CONFIGURATION ----------
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
    )""")
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
    )""")
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

def get_inventory_context(username):
    conn = sqlite3.connect(DB_NAME)
    df = pd.read_sql_query("SELECT * FROM inventory WHERE username = ?", conn, params=(username,))
    conn.close()
    if df.empty:
        return "There is no inventory data for this user."
    df['expiration_date'] = pd.to_datetime(df['expiration_date'], errors='coerce').dt.date
    summary = df.groupby('product_name').agg({
        'stock_in': 'sum',
        'stock_out': 'sum',
        'total_stock': 'last',
        'expiration_date': 'last'
    }).reset_index()
    context_lines = []
    for _, row in summary.iterrows():
        context_lines.append(
            f"Product: {row['product_name']}, Total In: {row['stock_in']}, "
            f"Out: {row['stock_out']}, Current Stock: {row['total_stock']}, "
            f"Last Expiry: {row['expiration_date']}"
        )
    return "\n".join(context_lines)

# ---------- APP ----------
create_tables()
st.set_page_config("📦 Inventory Tracker", layout="wide")
st.title("📦 Inventory Management System - Lima Time")

if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False

if not st.session_state.logged_in:
    tab1, tab2 = st.tabs(["🔑 Login", "🆕 Register"])
    with tab1:
        st.subheader("Login")
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        if st.button("Login"):
            if validate_login(username, password):
                st.session_state.logged_in = True
                st.session_state.username = username
                st.success(f"✅ Welcome, {username}!")
                st.experimental_rerun()
            else:
                st.error("❌ Invalid credentials")
    with tab2:
        st.subheader("Register")
        new_user = st.text_input("New Username")
        new_pass = st.text_input("New Password", type="password")
        if st.button("Register"):
            add_user(new_user, new_pass)
            st.success("✅ Account created! You can now login.")

else:
    st.sidebar.success(f"Logged in as {st.session_state.username}")
    df = load_inventory(st.session_state.username)

    st.subheader("➕ Add Inventory Movement")
    with st.form("add_form"):
        product_name = st.text_input("Product Name")
        batch_id = st.text_input("Batch ID")
        stock_in = st.number_input("Stock In", min_value=0, value=0)
        stock_out = st.number_input("Stock Out", min_value=0, value=0)
        unit_price = st.number_input("Unit Price", min_value=0.0, format="%.2f")
        quantity = st.number_input("Quantity", min_value=1, value=1)
        requires_expiration = st.radio("Does this product require an expiration date?", ("Yes", "No"))
        expiration_date = None
        if requires_expiration == "Yes":
            expiration_date = st.date_input("Expiration Date")
        else:
            st.info("🛈 No expiration date registered.")
        submitted = st.form_submit_button("✅ Record Entry")
        if submitted:
            now = datetime.now(lima_tz)
            total_units = stock_in - stock_out
            total_price = unit_price * quantity
            prev_stock = df[df["product_name"] == product_name]["total_stock"].iloc[-1] if product_name in df["product_name"].values else 0
            new_stock = prev_stock + total_units
            data = {
                "timestamp_in": now.strftime("%Y-%m-%d %H:%M:%S") if stock_in > 0 else None,
                "timestamp_out": now.strftime("%Y-%m-%d %H:%M:%S") if stock_out > 0 else None,
                "product_name": product_name,
                "batch_id": batch_id,
                "stock_in": stock_in,
                "stock_out": stock_out,
                "total_stock": new_stock,
                "unit_price": unit_price,
                "quantity": quantity,
                "total_price": total_price,
                "total_units": total_units,
                "expiration_date": expiration_date.strftime("%Y-%m-%d") if expiration_date else None,
                "username": st.session_state.username
            }
            insert_inventory(data)
            st.success(f"📦 Entry for **{product_name}** saved.")
            st.experimental_rerun()

    st.subheader("📊 Inventory Log")
    st.dataframe(df, use_container_width=True)

    # ---------- AutoGPT-style Assistant ----------
    st.subheader("🤖 AutoGPT-style Inventory Assistant")

    if 'openrouter_api_key' not in st.session_state:
        st.session_state['openrouter_api_key'] = ''
    if 'cohere_api_key' not in st.session_state:
        st.session_state['cohere_api_key'] = ''

    with st.expander("🔐 Enter Your API Keys"):
        openrouter_api_key = st.text_input("OpenRouter API Key", type="password")
        cohere_api_key = st.text_input("Cohere API Key", type="password")
        if openrouter_api_key:
            st.session_state['openrouter_api_key'] = openrouter_api_key
            st.success("OpenRouter API key saved successfully!")
        if cohere_api_key:
            st.session_state['cohere_api_key'] = cohere_api_key
            st.success("Cohere API key saved successfully!")

    if st.session_state['openrouter_api_key'] and st.session_state['cohere_api_key']:
        openrouter_client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=st.session_state['openrouter_api_key'])
        cohere_client = cohere.Client(st.session_state['cohere_api_key'])

        if 'auto_gpt_chat_history' not in st.session_state:
            st.session_state['auto_gpt_chat_history'] = []

        for message in st.session_state['auto_gpt_chat_history']:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        if prompt := st.chat_input("Describe your inventory task..."):
            st.session_state['auto_gpt_chat_history'].append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                full_response = ""
                try:
                    if len(prompt) >= 250:
                        summary = cohere_client.summarize(
                            text=prompt,
                            length='short',
                            format='plain',
                            model='summarize-xlarge',
                            additional_command='Summarize the task for Kimi AI.'
                        ).summary
                    else:
                        summary = prompt

                    context = get_inventory_context(st.session_state.username)
                    response = openrouter_client.chat.completions.create(
                        model="moonshotai/kimi-vl-a3b-thinking:free",
                        messages=[
                            {"role": "system", "content": f"You are an autonomous inventory assistant. Here is the inventory context:\n\n{context}"},
                            {"role": "user", "content": summary}
                        ]
                    )
                    full_response = response.choices[0].message.content
                    message_placeholder.markdown(full_response)
                except Exception as e:
                    message_placeholder.markdown(f"⚠️ Error: {e}")
                    full_response = f"⚠️ Error: {e}"

                st.session_state['auto_gpt_chat_history'].append({"role": "assistant", "content": full_response})
    else:
        st.warning("Please enter your API keys to use the assistant.")
