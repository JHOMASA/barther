import streamlit as st
import pandas as pd
import sqlite3
import openai
import cohere

# Initialize session state for API keys
if 'openrouter_api_key' not in st.session_state:
    st.session_state['openrouter_api_key'] = ''
if 'cohere_api_key' not in st.session_state:
    st.session_state['cohere_api_key'] = ''

st.set_page_config(page_title="Inventory Management Assistant", layout="wide")
st.title("📦 Inventory Management Assistant")

# Prompt user for API keys
st.sidebar.header("🔐 Enter Your API Keys")
with st.sidebar.form("api_keys_form"):
    openrouter_api_key = st.text_input("OpenRouter API Key", type="password")
    cohere_api_key = st.text_input("Cohere API Key", type="password")
    submitted = st.form_submit_button("Save API Keys")
    if submitted:
        if openrouter_api_key and cohere_api_key:
            st.session_state['openrouter_api_key'] = openrouter_api_key
            st.session_state['cohere_api_key'] = cohere_api_key
            st.success("API keys saved successfully!")
        else:
            st.error("Please enter both API keys.")

# Proceed only if API keys are provided
if st.session_state['openrouter_api_key'] and st.session_state['cohere_api_key']:
    # Set API keys for OpenAI and Cohere
    openai.api_key = st.session_state['openrouter_api_key']
    openai.api_base = "https://openrouter.ai/api/v1"
    co = cohere.Client(st.session_state['cohere_api_key'])

    # Database setup
    DB_NAME = "inventory.db"

    def create_tables():
        conn = sqlite3.connect(DB_NAME)
        c = conn.cursor()
        c.execute("""
            CREATE TABLE IF NOT EXISTS inventory (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                product_name TEXT,
                quantity INTEGER
            )
        """)
        conn.commit()
        conn.close()

    def add_product(product_name, quantity):
        conn = sqlite3.connect(DB_NAME)
        c = conn.cursor()
        c.execute("SELECT quantity FROM inventory WHERE product_name = ?", (product_name,))
        result = c.fetchone()
        if result:
            new_quantity = result[0] + quantity
            c.execute("UPDATE inventory SET quantity = ? WHERE product_name = ?", (new_quantity, product_name))
        else:
            c.execute("INSERT INTO inventory (product_name, quantity) VALUES (?, ?)", (product_name, quantity))
        conn.commit()
        conn.close()

    def get_inventory():
        conn = sqlite3.connect(DB_NAME)
        df = pd.read_sql_query("SELECT * FROM inventory", conn)
        conn.close()
        return df

    def delete_product(product_name):
        conn = sqlite3.connect(DB_NAME)
        c = conn.cursor()
        c.execute("DELETE FROM inventory WHERE product_name = ?", (product_name,))
        conn.commit()
        conn.close()

    # AI Functions
    def get_kimi_plan(user_input):
        response = openai.ChatCompletion.create(
            model="moonshotai/kimi-vl-a3b-thinking:free",
            messages=[{"role": "user", "content": user_input}]
        )
        return response.choices[0].message["content"]

    def get_cohere_response(prompt):
        response = co.chat(message=prompt, model="command-r-plus")
        return response.text

    # Initialize database
    create_tables()

    # Sidebar for inventory operations
    st.sidebar.header("Inventory Operations")
    operation = st.sidebar.selectbox("Choose an operation", ["Add Product", "Delete Product", "View Inventory"])

    if operation == "Add Product":
        st.sidebar.subheader("Add a New Product")
        product_name = st.sidebar.text_input("Product Name")
        quantity = st.sidebar.number_input("Quantity", min_value=1, step=1)
        if st.sidebar.button("Add"):
            if product_name:
                add_product(product_name, quantity)
                st.sidebar.success(f"Added {quantity} of {product_name} to inventory.")
            else:
                st.sidebar.error("Please enter a product name.")

    elif operation == "Delete Product":
        st.sidebar.subheader("Delete a Product")
        inventory_df = get_inventory()
        product_list = inventory_df['product_name'].tolist()
        if product_list:
            product_to_delete = st.sidebar.selectbox("Select Product", product_list)
            if st.sidebar.button("Delete"):
                delete_product(product_to_delete)
                st.sidebar.success(f"Deleted {product_to_delete} from inventory.")
        else:
            st.sidebar.info("Inventory is empty.")

    elif operation == "View Inventory":
        st.subheader("Current Inventory")
        inventory_df = get_inventory()
        st.dataframe(inventory_df)

    # AI Assistant
    st.subheader("🤖 Inventory Assistant")
    user_query = st.text_input("Ask a question about your inventory:")

    if st.button("Get Response"):
        if user_query:
            with st.spinner("Generating plan with Kimi AI..."):
                plan = get_kimi_plan(user_query)
            st.success("Plan generated by Kimi AI:")
            st.write(plan)

            with st.spinner("Executing plan with Cohere..."):
                response = get_cohere_response(plan)
            st.success("Response from Cohere:")
            st.write(response)
        else:
            st.error("Please enter a question.")
else:
    st.warning("Please enter your OpenRouter and Cohere API keys to continue.")
