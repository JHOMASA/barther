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
def backfill_missing_product_ids():
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()

    # Ensure product_id column exists
    c.execute("PRAGMA table_info(inventory)")
    columns = [col[1] for col in c.fetchall()]
    if 'product_id' not in columns:
        c.execute("ALTER TABLE inventory ADD COLUMN product_id TEXT")

    df = pd.read_sql("SELECT * FROM inventory", conn)

    # Fill missing or empty product_id values
    missing_mask = df['product_id'].isna() | (df['product_id'].astype(str).str.strip() == '')
    df.loc[missing_mask, 'product_id'] = df.loc[missing_mask].apply(
        lambda row: f"AUTO_{row['product_name']}_{row['batch_id']}", axis=1
    )

    # Replace entire table with updated values
    df.to_sql("inventory", conn, if_exists="replace", index=False)
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

def generate_invoice_pdf(df, product_name, batch_id, username):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.set_font("Arial", style='B', size=16)
    pdf.cell(200, 10, txt="INVENTORY INVOICE", ln=True, align='C')
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt=f"User: {username} | Product: {product_name} | Batch: {batch_id}", ln=True, align='C')
    pdf.cell(200, 10, txt=f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True, align='C')
    pdf.ln(10)
    pdf.set_font("Arial", style='B', size=10)
    headers = ['Timestamp', 'Stock In', 'Stock Out', 'Total Units', 'Total Price']
    for header in headers:
        pdf.cell(38, 10, header, border=1)
    pdf.ln()
    pdf.set_font("Arial", size=10)
    for _, row in df.iterrows():
        timestamp = row.get("timestamp_in") or row.get("timestamp_out")
        pdf.cell(38, 10, str(timestamp)[:19], border=1)
        pdf.cell(38, 10, str(row["stock_in"]), border=1)
        pdf.cell(38, 10, str(row["stock_out"]), border=1)
        pdf.cell(38, 10, str(row["total_units"]), border=1)
        pdf.cell(38, 10, f"${row['total_price']:.2f}", border=1)
        pdf.ln()
    temp_dir = tempfile.gettempdir()
    file_path = os.path.join(temp_dir, f"invoice_{product_name}_{batch_id}.pdf")
    pdf.output(file_path)
    return file_path


def show_expiration_alerts(df):
    st.subheader("🔔 Expiration Alerts (Next 7 Days)")
    if "expiration_date" in df.columns and not df["expiration_date"].isna().all():
        df["expiration_date"] = pd.to_datetime(df["expiration_date"], errors="coerce")
        today = pd.Timestamp(datetime.date.today())
        soon_expiring = df[(df["expiration_date"].notna()) & (df["expiration_date"] <= today + pd.Timedelta(days=7))]
        if not soon_expiring.empty:
            st.warning("⚠️ The following items are expiring in the next 7 days:")
            st.dataframe(soon_expiring[["product_name", "batch_id", "expiration_date", "total_stock"]], use_container_width=True)
        else:
            st.success("✅ No items expiring in the next 7 days.")
    else:
        st.info("ℹ️ No expiration data available.")


# ---------- PDF & ALERT TEST PANEL ----------
st.sidebar.markdown("---")
st.sidebar.subheader("🧪 Test Utilities")
df = pd.read_sql("SELECT * FROM inventory", sqlite3.connect(DB_NAME))
if not df.empty:
    selected_product = st.sidebar.selectbox("Select Product for PDF", df["product_name"].unique())
    filtered_df = df[df["product_name"] == selected_product]
    selected_batch = st.sidebar.selectbox("Select Batch", filtered_df["batch_id"].unique())
    batch_df = filtered_df[filtered_df["batch_id"] == selected_batch]
    if st.sidebar.button("📄 Generate PDF Invoice"):
        path = generate_invoice_pdf(batch_df, selected_product, selected_batch, "TestUser")
        with open(path, "rb") as f:
            st.sidebar.download_button("⬇ Download Invoice PDF", data=f, file_name=os.path.basename(path), mime="application/pdf")

show_expiration_alerts(df)

# ---------- MAIN APP ----------
create_tables()
backfill_missing_product_ids()
    
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
        product_id = st.text_input("Product ID")
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
                "product_id": product_id,
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
        if 'product_id' in df.columns:
            summary = (
                df.groupby(['product_name', 'product_id', 'batch_id'])
                .agg(
                    total_in=pd.NamedAgg(column='stock_in', aggfunc='sum'),
                    total_out=pd.NamedAgg(column='stock_out', aggfunc='sum')
                )
                .reset_index()
            )
            summary['current_stock'] = summary['total_in'] - summary['total_out']

            gb = GridOptionsBuilder.from_dataframe(summary[['product_name', 'product_id', 'batch_id', 'current_stock']])
            gb.configure_pagination()
            gb.configure_side_bar()
            gb.configure_default_column(filterable=True, sortable=True, editable=False)
            grid_options = gb.build()

            AgGrid(summary[['product_name', 'product_id', 'batch_id', 'current_stock']], gridOptions=grid_options, height=350, theme="streamlit")
        else:
            st.warning("⚠️ Your inventory table does not include 'product_id'. New records will have it, but old entries won’t show here.")


    st.subheader("📈 Stock Movement Over Time")
    if not df.empty:
        df['timestamp'] = pd.to_datetime(df['timestamp_in'].fillna(df['timestamp_out']), errors='coerce')

        required_cols = ['timestamp', 'product_name', 'batch_id', 'stock_in', 'stock_out']
        if 'product_id' in df.columns:
            required_cols.insert(2, 'product_id')  # Insert product_id after product_name

        missing_cols = [col for col in required_cols if col not in df.columns]

        if missing_cols:
            st.warning(f"⚠️ Missing columns in your inventory: {', '.join(missing_cols)}. Please add new inventory entries with full data.")
            time_filtered = pd.DataFrame()  # Empty to prevent crash
        else:
            time_filtered = df[required_cols].dropna()

        selected_product = st.selectbox("Select Product to View Stock Movement", df['product_name'].unique())
        product_df = time_filtered[time_filtered['product_name'] == selected_product]
        if not product_df.empty and 'product_id' in product_df.columns:
            selected_product_id = product_df['product_id'].iloc[0]
        else:
            selected_product_id = "N/A"
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

        fig.update_layout(
            title=f"Stock Movement for {selected_product} [ID: {selected_product_id}] ({selected_batch})",
            xaxis_title="Date",
            yaxis_title="Units",
            legend_title="Legend",
            hovermode="x unified"
        )
        st.plotly_chart(fig, use_container_width=True)

        export_cols = ['timestamp', 'stock_in', 'stock_out', 'cumulative_stock']
        if 'product_id' in product_df.columns:
            export_cols.insert(1, 'product_id')  # place after timestamp

        missing_cols = [col for col in export_cols if col not in product_df.columns]
        if missing_cols:
            st.warning(f"⚠️ Cannot export CSV: missing columns: {', '.join(missing_cols)}")
        else:
            csv_data = product_df[export_cols].to_csv(index=False).encode()
            st.download_button(
                "⬇ Download Stock Movement CSV",
                csv_data,
                f"{selected_product}_{selected_batch}_stock_movement.csv",
                "text/csv"
            )

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
