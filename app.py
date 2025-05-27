import streamlit as st
import pandas as pd
import sqlite3
import hashlib
from datetime import datetime, timedelta
import pytz
import plotly.express as px
from st_aggrid import AgGrid, GridOptionsBuilder
import plotly.graph_objects as go

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
        product_id TEXT,
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
            batch_df = df[(df["product_name"] == product_name) & (df["batch_id"] == batch_id)]
            prev_stock = batch_df["total_stock"].iloc[-1] if not batch_df.empty else 0
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
            st.success(f"📦 Entry for **{product_name}** (Batch {batch_id}) saved.")
            st.experimental_rerun()

    st.subheader("📦 Interactive Batch-Level Stock Summary")
    if not df.empty:
        summary = (
            df.groupby(['product_name', 'batch_id'])
            .agg(
                total_in=pd.NamedAgg(column='stock_in', aggfunc='sum'),
                total_out=pd.NamedAgg(column='stock_out', aggfunc='sum')
            )
            .reset_index()
        )
        summary['current_stock'] = summary['total_in'] - summary['total_out']

        gb = GridOptionsBuilder.from_dataframe(summary[['product_name', 'batch_id', 'current_stock']])
        gb.configure_pagination()
        gb.configure_side_bar()
        gb.configure_default_column(filterable=True, sortable=True, editable=False)
        grid_options = gb.build()

        AgGrid(summary[['product_name', 'batch_id', 'current_stock']], gridOptions=grid_options, height=350, theme="streamlit")

    st.subheader("📈 Stock Movement Over Time")
    if not df.empty:
        df['timestamp'] = pd.to_datetime(df['timestamp_in'].fillna(df['timestamp_out']), errors='coerce')
        time_filtered = df[['timestamp', 'product_name', 'batch_id', 'stock_in', 'stock_out']].dropna()

        selected_product = st.selectbox("Select Product to View Stock Movement", df['product_name'].unique())
        product_df = time_filtered[time_filtered['product_name'] == selected_product]

        selected_batch = st.selectbox("Select Batch ID (optional)", ['All'] + sorted(product_df['batch_id'].dropna().unique().tolist()))
        if selected_batch != 'All':
            product_df = product_df[product_df['batch_id'] == selected_batch]

        product_df = product_df.sort_values("timestamp")
        product_df['net_stock'] = product_df['stock_in'] - product_df['stock_out']
        product_df['cumulative_stock'] = product_df['net_stock'].cumsum()

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=product_df['timestamp'], y=product_df['stock_in'], mode='lines+markers', name='Stock In'))
        fig.add_trace(go.Scatter(x=product_df['timestamp'], y=product_df['stock_out'], mode='lines+markers', name='Stock Out'))
        fig.add_trace(go.Scatter(x=product_df['timestamp'], y=product_df['cumulative_stock'], mode='lines+markers', name='Net Stock'))

        fig.update_layout(title=f"Stock Movement for {selected_product} ({selected_batch})",
                          xaxis_title="Date",
                          yaxis_title="Units",
                          legend_title="Legend",
                          hovermode="x unified")
        st.plotly_chart(fig, use_container_width=True)

        csv_data = product_df[['timestamp', 'stock_in', 'stock_out', 'cumulative_stock']].to_csv(index=False).encode()
        st.download_button("⬇ Download Stock Movement CSV", csv_data, f"{selected_product}_{selected_batch}_stock_movement.csv", "text/csv")

    st.subheader("📊 Inventory Log")
    st.dataframe(df, use_container_width=True)

    st.subheader("🗃️ Internal Database Explorer")
    conn = sqlite3.connect(DB_NAME)
    table_choice = st.selectbox("Select Table to View", ["inventory", "users"])
    df_table = pd.read_sql(f"SELECT * FROM {table_choice}", conn)
    conn.close()
    st.dataframe(df_table, use_container_width=True)
    st.download_button(
        f"⬇ Download {table_choice}.csv",
        df_table.to_csv(index=False).encode(),
        f"{table_choice}.csv",
        "text/csv"
    )

