import streamlit as st
import pandas as pd
import sqlite3
import hashlib
from datetime import datetime, timedelta
import pytz
import plotly.express as px

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

# ---------- MAIN APP ----------
create_tables()
st.set_page_config(page_title="Inventory Tracker", layout="wide")
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

    with st.form("add_stock_form"):
        st.subheader("➕ Add Inventory Movement")
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
            prev_stock = (
                df[df["product_name"] == product_name]["total_stock"].iloc[-1]
                if product_name in df["product_name"].values else 0
            )
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

    if "expiration_date" in df.columns:
        df['expiration_date'] = pd.to_datetime(df['expiration_date'], errors='coerce')
        expired = df[df['expiration_date'] < datetime.now()]
        expiring_soon = df[(df['expiration_date'] >= datetime.now()) &
                           (df['expiration_date'] <= datetime.now() + timedelta(days=7))]

        if not expired.empty:
            st.warning("⚠️ Some products have **expired**:")
            st.dataframe(expired[["product_name", "batch_id", "expiration_date"]])

        if not expiring_soon.empty:
            st.info("🔔 Products **expiring soon** (within 7 days):")
            st.dataframe(expiring_soon[["product_name", "batch_id", "expiration_date"]])

    st.subheader("🗑️ Delete Specific Inventory Row")
    if not df.empty:
        row_to_delete = st.selectbox("Select Row ID to Delete", df['id'].astype(str))
        if st.button("Delete Selected Row"):
            conn = sqlite3.connect(DB_NAME)
            conn.execute("DELETE FROM inventory WHERE id = ? AND username = ?", (row_to_delete, st.session_state.username))
            conn.commit()
            conn.close()
            st.success(f"✅ Row with ID {row_to_delete} deleted.")
            st.experimental_rerun()

    st.download_button("⬇ Download CSV", df.to_csv(index=False).encode(), "inventory.csv", "text/csv")

    st.subheader("📈 Inventory Over Time (Line Graph by Product)")
    if not df.empty:
        df['timestamp_in'] = pd.to_datetime(df['timestamp_in'], errors='coerce')
        grouped = df.groupby(['product_name', pd.Grouper(key='timestamp_in', freq='D')])['total_stock'].max().reset_index()

        fig = px.line(
            grouped,
            x='timestamp_in',
            y='total_stock',
            color='product_name',
            title="📉 Stock Level per Product Over Time"
        )
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("🗃️ Full Inventory Table (From Database)")
    conn = sqlite3.connect(DB_NAME)
    full_df = pd.read_sql("SELECT * FROM inventory WHERE username = ?", conn, params=(st.session_state.username,))
    st.dataframe(full_df, use_container_width=True)
    conn.close()




    
