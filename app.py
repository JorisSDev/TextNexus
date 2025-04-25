import sqlite3
import torch
import pandas as pd

from flask import Flask, render_template, request, redirect, url_for

from database import init_db
from api import api
from llm_engine import (
    generate_with_gpt2,
    generate_with_bart,
    generate_with_bert,
    generate_with_deepseek,
    generate_with_gpt41,
    generate_with_bitnet,
)
app = Flask(__name__)

init_db()


def save_to_db(input_text, output_text, model, chat_session_id=None):
    conn = sqlite3.connect("textnexus.db")
    cursor = conn.cursor()
    if chat_session_id is not None:
        cursor.execute(
            "INSERT INTO text_records (input_text, output_text, model, chat_session_id) VALUES (?, ?, ?, ?)",
            (input_text, output_text, model, chat_session_id)
        )
    else:
        cursor.execute(
            "INSERT INTO text_records (input_text, output_text, model) VALUES (?, ?, ?)",
            (input_text, output_text, model)
        )
    conn.commit()
    conn.close()

@app.route("/", methods=["GET", "POST"])
def home():
    output = None
    selected_model = "gpt2"  # Default model
    input_text = ""

    if request.method == "POST":
        input_text = request.form.get("input_text", "").strip()
        selected_model = request.form.get("model", "gpt2")
        api_key = request.form.get("api_key", "").strip()

        if input_text:
            if selected_model == "gpt2":
                output = generate_with_gpt2(input_text)
            elif selected_model == "bart":
                output = generate_with_bart(input_text)
            elif selected_model == "bert":
                output = generate_with_bert(input_text)
            elif selected_model == "deepseek":
                output = generate_with_deepseek(input_text)
            elif selected_model == "gpt41":
                output = generate_with_gpt41(input_text, api_key)
            elif selected_model == "bitnet":
                output = generate_with_bitnet(input_text)
            elif selected_model == "llama32":
                output = generate_with_llama32(input_text)
            else:
                output = ["Invalid model selected."]

    return render_template("index.html", output=output, selected_model=selected_model, input_text=input_text)

def get_chat_messages(session_name):
    conn = sqlite3.connect("textnexus.db")
    cursor = conn.cursor()
    cursor.execute("SELECT role, content FROM chat_sessions WHERE session_name = ? ORDER BY id", (session_name,))
    rows = cursor.fetchall()
    conn.close()
    return [{"role": role, "content": content} for role, content in rows]

def add_chat_message(session_name, role, content, model=None):
    conn = sqlite3.connect("textnexus.db")
    cursor = conn.cursor()
    cursor.execute("INSERT INTO chat_sessions (session_name, role, content, model) VALUES (?, ?, ?, ?)",
                   (session_name, role, content, model))
    conn.commit()
    conn.close()

def get_all_session_names():
    conn = sqlite3.connect("textnexus.db")
    cursor = conn.cursor()
    cursor.execute("SELECT DISTINCT session_name FROM chat_sessions")
    sessions = [row[0] for row in cursor.fetchall()]
    conn.close()
    # If no sessions exist, provide some defaults
    if not sessions:
        return ["session1", "session2", "session3", "session4"]
    return sessions

def get_latest_session_id(session_name):
    conn = sqlite3.connect("textnexus.db")
    cursor = conn.cursor()
    cursor.execute("SELECT id FROM chat_sessions WHERE session_name = ? ORDER BY id DESC LIMIT 1", (session_name,))
    row = cursor.fetchone()
    conn.close()
    return row[0] if row else None

@app.route("/chatbot", methods=["GET", "POST"])
def chatbot():
    selected_session = request.args.get("session", "session1")
    user_input = ""
    selected_model = request.form.get("model")
    if not selected_model:
        conn = sqlite3.connect("textnexus.db")
        cursor = conn.cursor()
        cursor.execute("SELECT model FROM chat_sessions WHERE session_name = ? AND model IS NOT NULL ORDER BY id DESC LIMIT 1", (selected_session,))
        row = cursor.fetchone()
        conn.close()
        selected_model = row[0] if row else "gpt2"
    api_key = request.form.get("api_key", "").strip()

    if request.method == "POST":
        user_input = request.form.get("user_input", "")
        if user_input:
            add_chat_message(selected_session, "user", user_input, selected_model)

            session_id = get_latest_session_id(selected_session)

            messages = get_chat_messages(selected_session)
            prompt = ""
            for msg in messages:
                if msg["role"] == "user":
                    prompt += f"User: {msg['content']}\n"
                elif msg["role"] == "bot":
                    prompt += f"Bot: {msg['content']}\n"

            # Use shared generate_with_* functions for text generation
            if selected_model == "gpt2":
                response = generate_with_gpt2(prompt, chat_session_id=session_id)
                generated_text = response[0]
            elif selected_model == "deepseek":
                response = generate_with_deepseek(prompt, chat_session_id=session_id)
                generated_text = response[0]
            elif selected_model == "gpt41":
                response = generate_with_gpt41(prompt, api_key, chat_session_id=session_id)
                generated_text = response[0]
            elif selected_model == "llama32":
                response = generate_with_llama32(prompt, chat_session_id=session_id)
                generated_text = response[0]
            else:
                generated_text = "Selected model is not supported in chatbot."

            generated_part = generated_text[len(prompt):].strip() if generated_text.startswith(prompt) else generated_text
            add_chat_message(selected_session, "bot", generated_part, selected_model)

    return render_template("chatbot.html",
                          sessions=get_all_session_names(),
                          selected_session=selected_session,
                          messages=get_chat_messages(selected_session),
                          selected_model=selected_model)

@app.route("/create_session", methods=["POST"])
def create_session():
    new_session = request.form.get("new_session", "").strip()
    if new_session:
        # Create a dummy entry to ensure the session appears
        add_chat_message(new_session, "system", "New session created.")
    return redirect(url_for("chatbot", session=new_session))


@app.route("/delete_session/<session_name>", methods=["POST"])
def delete_session(session_name):
    conn = sqlite3.connect("textnexus.db")
    cursor = conn.cursor()
    cursor.execute("DELETE FROM chat_sessions WHERE session_name = ?", (session_name,))
    conn.commit()
    conn.close()
    return redirect(url_for("chatbot"))


# Settings page route
@app.route("/settings")
def settings():
    return render_template("settings.html")

@app.route("/users")
def users():
    return render_template("users.html")

@app.route("/login")
def login():
    return render_template("login.html")

@app.route("/register")
def register():
    return render_template("register.html")

@app.route("/info")
def info():
    return render_template("info.html")

@app.route("/model_configuration")
def model_configuration():
    return render_template("model_configuration.html")

@app.route("/dashboard")
def dashboard():
    conn = sqlite3.connect("textnexus.db")
    df = pd.read_sql_query("SELECT * FROM text_records", conn)
    conn.close()

    # Ensure all data are pure Python types
    model_usage = df['model'].value_counts()
    model_labels = list(map(str, model_usage.index))
    model_counts = list(map(int, model_usage.values))  # Convert from np.int64 to int

    df['timestamp'] = pd.to_datetime(df['timestamp'])
    daily_usage = df['timestamp'].dt.date.value_counts().sort_index()
    daily_dates = [str(date) for date in daily_usage.index]
    daily_counts = list(map(int, daily_usage.values))

    top_sessions_raw = df['chat_session_id'].value_counts().head(5)
    top_sessions = [(f"Session {int(sid)}", int(count)) for sid, count in top_sessions_raw.items()]

    df['output_length'] = df['output_text'].apply(lambda x: len(x.split()))
    avg_output_by_model = df.groupby('model')['output_length'].mean()
    output_models = list(map(str, avg_output_by_model.index))
    output_lengths = [round(float(val), 2) for val in avg_output_by_model.values]

    recent_logs = df.sort_values(by='timestamp', ascending=False).head(10).to_dict(orient='records')

    return render_template(
        "dashboard.html",
        model_labels=model_labels,
        model_counts=model_counts,
        daily_dates=daily_dates,
        daily_counts=daily_counts,
        top_sessions=top_sessions,
        output_models=output_models,
        output_lengths=output_lengths,
        recent_logs=recent_logs
    )

if __name__ == "__main__":
    app.register_blueprint(api)
    app.run(debug=True)