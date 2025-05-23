import streamlit as st
st.set_page_config("📦 Inventory Tracker", layout="wide")
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
from graphviz import Digraph

# ---------- CONFIGURATION ----------
DB_NAME = "inventory.db"
lima_tz = pytz.timezone("America/Lima")


# ---------- DATABASE SETUP ----------
def create_tables():
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute(\"\"\"
    CREATE TABLE IF NOT EXISTS users (
        username TEXT PRIMARY KEY,
        password TEXT NOT NULL
    )\"\"\")
    c.execute(\"\"\"
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
    )\"\"\")
    conn.commit()
    conn.close()

create_tables()

# ---------- AUTHENTICATION ----------
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

# ---------- MODEL SELECTION ----------
def select_model():
    return st.radio("Choose Reasoning Model:", ["Mistral 7B", "Cohere Command R+"], index=0, key="model_choice_unique")

model_choice = select_model()

with st.expander("🔐 Enter Your API Keys"):
    if model_choice == "Mistral 7B":
        openrouter_api_key = st.text_input("OpenRouter API Key", type="password", key="openrouter_key")
        if openrouter_api_key:
            st.session_state['openrouter_api_key'] = openrouter_api_key
            st.success("✅ OpenRouter API key saved!")
    elif model_choice == "Cohere Command R+":
        cohere_api_key = st.text_input("Cohere API Key", type="password", key="cohere_key")
        if cohere_api_key:
            st.session_state['cohere_api_key'] = cohere_api_key
            st.success("✅ Cohere API key saved!")

if model_choice == "Mistral 7B" and not st.session_state.get("openrouter_api_key"):
    st.warning("⚠️ Please enter your OpenRouter API key to use Mistral.")
elif model_choice == "Cohere Command R+" and not st.session_state.get("cohere_api_key"):
    st.warning("⚠️ Please enter your Cohere API key to use Command R+.")

# ---------- LOGIN SYSTEM ----------
st.title("📦 Inventory Management System - Lima Time")
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False

if not st.session_state.logged_in:
    tab1, tab2 = st.tabs(["🔑 Login", "🆕 Register"])
    with tab1:
        st.subheader("Login")
        username = st.text_input("Username", key="login_user")
        password = st.text_input("Password", type="password", key="login_pass")
        if st.button("Login", key="login_btn"):
            if validate_login(username, password):
                st.session_state.logged_in = True
                st.session_state.username = username
                st.experimental_rerun()
            else:
                st.error("❌ Invalid credentials")
    with tab2:
        st.subheader("Register")
        new_user = st.text_input("New Username", key="reg_user")
        new_pass = st.text_input("New Password", type="password", key="reg_pass")
        if st.button("Register", key="register_btn"):
            add_user(new_user, new_pass)
            st.success("✅ Account created! You can now login.")
else:
    st.sidebar.success(f"Logged in as: {st.session_state.username}")

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

    st.subheader("➕ Add Inventory Movement")
    with st.form("add_form"):
        product_name = st.text_input("Product Name", key="prod_name")
        batch_id = st.text_input("Batch ID", key="batch_id")
        stock_in = st.number_input("Stock In", min_value=0, value=0, key="stock_in")
        stock_out = st.number_input("Stock Out", min_value=0, value=0, key="stock_out")
        unit_price = st.number_input("Unit Price", min_value=0.0, format="%.2f", key="unit_price")
        quantity = st.number_input("Quantity", min_value=1, value=1, key="quantity")
        requires_expiration = st.radio("Does this product require an expiration date?", ("Yes", "No"), key="require_exp")
        expiration_date = st.date_input("Expiration Date", key="exp_date") if requires_expiration == "Yes" else None
        submitted = st.form_submit_button("✅ Record Entry", key="submit_inv")
        if submitted:
            now = datetime.now(lima_tz)
            df = load_inventory(st.session_state.username)
            prev_stock = df[df["product_name"] == product_name]["total_stock"].iloc[-1] if product_name in df["product_name"].values else 0
            total_units = stock_in - stock_out
            new_stock = prev_stock + total_units
            total_price = unit_price * quantity
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

    df = load_inventory(st.session_state.username)
    st.subheader("📊 Inventory Log")
    st.dataframe(df, use_container_width=True)
    
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

# ---------- GRAPH VISUALIZATION ----------
def visualize_reasoning_with_graph4nlp(steps):
    dot = Digraph()
    dot.attr(rankdir='LR')
    for i, step in enumerate(steps):
        dot.node(f"{i+1}", step['step'])
        if i > 0:
            dot.edge(f"{i}", f"{i+1}")
    return dot

# ---------- INVENTORY GRAPH ----------
def show_inventory_graph(username):
    conn = sqlite3.connect(DB_NAME)
    df = pd.read_sql_query("SELECT * FROM inventory WHERE username = ?", conn, params=(username,))
    conn.close()
    if df.empty:
        st.info("No inventory data available to display the graph.")
        return
    df['timestamp_in'] = pd.to_datetime(df['timestamp_in'], errors='coerce')
    df = df.dropna(subset=['timestamp_in'])
    grouped = df.groupby(['product_name', pd.Grouper(key='timestamp_in', freq='D')])['total_stock'].max().reset_index()
    fig = px.bar(
        grouped,
        x='timestamp_in',
        y='total_stock',
        color='product_name',
        title="📊 Daily Stock Trends by Product",
        barmode='group',
        labels={'timestamp_in': 'Date', 'total_stock': 'Stock Level'}
    )
    fig.update_layout(
        legend_title_text='Product Name',
        xaxis_title='Date',
        yaxis_title='Stock Quantity',
        hovermode='x unified',
        template='plotly_white',
        height=500,
        margin=dict(l=20, r=20, t=60, b=20),
        font=dict(size=14)
    )
    st.plotly_chart(fig, use_container_width=True)

# ---------- SQL EXPLORER ----------
def show_sql_explorer(username):
    st.subheader("🧠 SQL Explorer")
    st.markdown("You can write SQL queries below using the `inventory` table. Example:")
    st.code(f"SELECT product_name, SUM(total_stock) as total_stock FROM inventory WHERE username = '{username}' GROUP BY product_name", language="sql")
    query_input = st.text_area("Write your SQL query:", key="sql_query")
    if st.button("▶ Run SQL Query", key="run_sql"):
        conn = sqlite3.connect(DB_NAME)
        try:
            df_query = pd.read_sql_query(query_input, conn)
            st.success("✅ Query executed successfully")
            st.dataframe(df_query, use_container_width=True)
        except Exception as e:
            st.error(f"❌ SQL Error: {e}")
        finally:
            conn.close()

# ---------- USER MODEL CHOICE ----------
def select_model():
    return st.radio("Choose Reasoning Model:", ["Mistral 7B", "Cohere Command R+"], index=0, key="unique_model_choice")
    
# ---------- AUTO REASONING LOOP ----------
# ---------- AUTO REASONING LOOP ----------
def run_reasoning_loop(user_goal, inventory_context, openrouter_client, model_choice):
    loop_history = []

    if model_choice == "Cohere Command R+":
        if not st.session_state.get('cohere_api_key'):
            st.warning("Cohere API key required for this model.")
            return []
        cohere_client = cohere.Client(st.session_state['cohere_api_key'])
        plan = cohere_client.generate(
            model='command',
            prompt=f"Break down this goal into step-by-step reasoning tasks.\n\nGoal: {user_goal}",
            max_tokens=150,
            temperature=0.5
        )
        steps = plan.generations[0].text.strip().split("\n")
        steps = [s for s in steps if s.strip() and s[0].isdigit()]
    else:
        if not st.session_state.get('openrouter_api_key'):
            st.warning("OpenRouter API key required for Mistral 7B.")
            return []
        mistral_response = openrouter_client.chat.completions.create(
            model="mistralai/mistral-7b-instruct:free",
            messages=[
                {"role": "system", "content": "Break down user goals into actionable inventory steps."},
                {"role": "user", "content": f"Goal: {user_goal}"}
            ]
        )
        steps = mistral_response.choices[0].message.content.strip().split("\n")
        steps = [s for s in steps if s.strip() and s[0].isdigit()]

    for step in steps:
        current_step = step.split(".", 1)[-1].strip()
        kimi_response = openrouter_client.chat.completions.create(
            model="moonshotai/kimi-vl-a3b-thinking:free",
            messages=[
                {"role": "system", "content": f"You are an inventory reasoning assistant. Here is the inventory data:\n\n{inventory_context}"},
                {"role": "user", "content": f"Your current task is: {current_step}. Execute it and provide the result."}
            ]
        )
        result = kimi_response.choices[0].message.content
        loop_history.append({"step": current_step, "result": result})
    return loop_history



# ---------- GRAPH4NLP SIMULATION ----------
def visualize_reasoning_with_graph4nlp(steps):
    dot = Digraph()
    dot.attr(rankdir='LR')
    for i, step in enumerate(steps):
        dot.node(f"{i+1}", step['step'])
        if i > 0:
            dot.edge(f"{i}", f"{i+1}")
    return dot

# ---------- MINI AUTOGPT LOOP FUNCTION ----------
# ---------- MINI AUTOGPT LOOP FUNCTION ----------
def run_reasoning_loop(user_goal, inventory_context, cohere_client, openrouter_client):
    loop_history = []

    task_plan = cohere_client.generate(
        model='command',
        prompt=f"Break down this goal into step-by-step reasoning tasks.\n\nGoal: {user_goal}",
        max_tokens=150,
        temperature=0.5
    )
    steps = task_plan.generations[0].text.strip().split("\n")
    steps = [step for step in steps if step.strip() and step[0].isdigit()]

    for step in steps:
        current_step = step.split(".", 1)[-1].strip()

        if 'interruption' in st.session_state and st.session_state['interruption']:
            current_step += f" (Note: User previously asked: {st.session_state['interruption']})"

        response = openrouter_client.chat.completions.create(
            model="moonshotai/kimi-vl-a3b-thinking:free",
            messages=[
                {"role": "system", "content": f"You are an inventory reasoning assistant. Here is the inventory data:\n\n{inventory_context}"},
                {"role": "user", "content": f"Your current task is: {current_step}. Execute it and provide the result."}
            ]
        )
        result = response.choices[0].message.content if response and hasattr(response, "choices") and response.choices else "No result."

        loop_history.append({"step": current_step, "result": result})

        follow_up = st.text_input(f"💬 Ask a follow-up or clarify step '{current_step}' (optional)", key=current_step)
        if follow_up:
            st.session_state['interruption'] = follow_up
            break
        else:
            st.session_state['interruption'] = None

        reflection_prompt = f"""
        You just completed the step: {current_step}.
        Result: {result}
        Should we continue to the next step? If yes, say NEXT. If done, say DONE.
        """
        reflection = cohere_client.generate(
            model='command',
            prompt=reflection_prompt,
            max_tokens=30
        ).generations[0].text.strip().upper()

        if "DONE" in reflection:
            break

    return loop_history

# ---------- EMAIL SENDING FUNCTION ----------
def send_email_with_attachment(to_address, subject, body, attachment_name, attachment_data):
    import smtplib
    from email.message import EmailMessage

    sender_email = "your_email@example.com"
    sender_password = "your_email_password"

    msg = EmailMessage()
    msg['Subject'] = subject
    msg['From'] = sender_email
    msg['To'] = to_address
    msg.set_content(body)

    msg.add_attachment(attachment_data, maintype='application', subtype='octet-stream', filename=attachment_name)

    with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
        smtp.login(sender_email, sender_password)
        smtp.send_message(msg)

# ---------- CHAT ENTRY POINT ----------
if 'username' in st.session_state:
    st.markdown("### 📈 Inventory Visualization")
    show_inventory_graph(st.session_state.username)

    inventory_context = get_inventory_context(st.session_state.username)
    with st.expander("📦 Current Inventory Context", expanded=False):
        st.markdown(inventory_context)

    st.subheader("🧠 AutoGPT-like Inventory Reasoning")
    user_goal = st.text_input("What do you want to achieve with your inventory?", key="user_goal_input")
    email_address = st.text_input("Recipient email (optional)")
    permission_level = st.selectbox("Access level", ["admin", "owner", "reviewer"])

    if st.button("🔁 Start Reasoning") and user_goal:
        with st.spinner("Running reasoning loop..."):
            openrouter_client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=st.session_state['openrouter_api_key'])
            cohere_client = cohere.Client(st.session_state['cohere_api_key'])
            loop_results = run_reasoning_loop(user_goal, inventory_context, cohere_client, openrouter_client)

        if 'reasoning_logs' not in st.session_state:
            st.session_state['reasoning_logs'] = []
        st.session_state['reasoning_logs'].append({"goal": user_goal, "results": loop_results, "email": email_address, "access": permission_level})

        st.success("✅ Reasoning completed")
        for i, step in enumerate(loop_results):
            st.markdown(f"**Step {i+1}: {step['step']}**")
            st.markdown(f"Result: {step['result']}")

        export_df = pd.DataFrame(loop_results)
        csv = export_df.to_csv(index=False).encode('utf-8')
        json_data = export_df.to_json(orient='records', indent=2).encode('utf-8')

        st.download_button("⬇ Download CSV", csv, f"reasoning_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv", "text/csv")
        st.download_button("⬇ Download JSON", json_data, f"reasoning_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json", "application/json")

        if email_address and st.checkbox("📧 Email this report"):
            try:
                send_email_with_attachment(
                    to_address=email_address,
                    subject="Inventory Reasoning Report",
                    body=f"See the reasoning result for: {user_goal}",
                    attachment_name="reasoning.csv",
                    attachment_data=csv
                )
                st.success(f"📬 Sent to {email_address}")
            except Exception as e:
                st.error(f"❌ Email failed: {e}")

    show_sql_explorer(st.session_state.username)

    if 'reasoning_logs' in st.session_state and st.session_state['reasoning_logs']:
        latest_steps = st.session_state['reasoning_logs'][-1]['results']
        st.markdown("#### 🔗 Graph4NLP Reasoning Graph")
        st.graphviz_chart(visualize_reasoning_with_graph4nlp(latest_steps))
    else:
        st.markdown("#### 🔗 Graph4NLP Reasoning Graph")
        st.info("Run the reasoning loop to visualize reasoning steps as a graph.")
else:
    inventory_context = "No user logged in yet."
    with st.expander("📦 Current Inventory Context", expanded=False):
        st.markdown(inventory_context)

# ---------- APP ----------
create_tables()
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
    # ---------- USER MODEL CHOICE (SINGLE INSTANCE) ----------
    model_choice = st.radio("Choose Reasoning Model:", ["Mistral 7B", "Cohere Command R+"], index=0, key="model_choice")
    with st.expander("🔐 Enter Your API Keys"):
     model_choice = select_model()  # This must be called early

     openrouter_api_key = st.text_input("OpenRouter API Key", type="password")
     if openrouter_api_key:
        st.session_state['openrouter_api_key'] = openrouter_api_key
        st.success("✅ OpenRouter API key saved!")

     if model_choice == "Cohere Command R+":
        cohere_api_key = st.text_input("Cohere API Key", type="password")
        if cohere_api_key:
            st.session_state['cohere_api_key'] = cohere_api_key
            st.success("✅ Cohere API key saved!")
    # Warn only if needed
    if model_choice == "Mistral 7B" and not st.session_state.get("openrouter_api_key"):
        st.warning("Please enter your OpenRouter API key to use Mistral.")
    elif model_choice == "Cohere Command R+" and not st.session_state.get("cohere_api_key"):
        st.warning("Please enter your Cohere API key to use Command R+.")

    model_choice = select_model()
    if((model_choice == "Mistral 7B" and st.session_state.get('openrouter_api_key')) or(model_choice == "Cohere Command R+" and st.session_state.get('cohere_api_key'))):
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
