import sqlite3
import torch

from flask import Flask, render_template, request, redirect, url_for
from openai import OpenAI
from transformers import pipeline, set_seed, AutoModelForCausalLM, AutoTokenizer

app = Flask(__name__)

def init_db():
    conn = sqlite3.connect("textnexus.db")
    cursor = conn.cursor()
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS text_records (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        input_text TEXT NOT NULL,
        output_text TEXT NOT NULL,
        model TEXT NOT NULL,
        chat_session_id INTEGER,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY(chat_session_id) REFERENCES chat_sessions(id) ON DELETE SET NULL
    )
    """)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS chat_sessions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        session_name TEXT NOT NULL,
        role TEXT NOT NULL,
        content TEXT NOT NULL,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
    )
    """)
    # Check and add 'model' column if not exists
    cursor.execute("PRAGMA table_info(chat_sessions)")
    columns = [row[1] for row in cursor.fetchall()]
    if 'model' not in columns:
        cursor.execute("ALTER TABLE chat_sessions ADD COLUMN model TEXT")
    conn.commit()
    conn.close()


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


def load_gpt2():
    """Loads the GPT-2 model and sets a random seed for reproducibility."""
    set_seed(42)
    return pipeline("text-generation", model="gpt2", truncation=True)


def load_bart():
    return pipeline("summarization", model="facebook/bart-large-cnn")


def load_bert():
    return pipeline("fill-mask", model="bert-base-uncased")


def load_deepseek():
    return pipeline("text-generation", model="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", truncation=True)


# Load models once
generator_gpt2 = load_gpt2()
summarizer_bart = load_bart()
mask_filler_bert = load_bert()
generator_deepseek = load_deepseek()


def generate_with_gpt2(text, chat_session_id=None):
    results = generator_gpt2(text, max_new_tokens=50, num_return_sequences=3)
    output = [res["generated_text"] for res in results]
    save_to_db(text, " | ".join(output), "gpt2", chat_session_id)
    return output


def generate_with_bart(text, chat_session_id=None):
    summary = summarizer_bart(text, max_length=130, min_length=30, do_sample=False)
    output = [summary[0]["summary_text"]]
    save_to_db(text, output[0], "bart", chat_session_id)
    return output


def generate_with_bert(text, chat_session_id=None):
    """Fills in the [MASK] token in a sentence using BERT."""
    if "[MASK]" not in text:
        return ["Please include a [MASK] token in the input text."]

    predictions = mask_filler_bert(text)
    output = [f"{pred['sequence']} (Confidence: {pred['score']:.2%})" for pred in predictions[:5]]
    save_to_db(text, " | ".join(output), "bert", chat_session_id)
    return output


def generate_with_deepseek(text, chat_session_id=None):
    results = generator_deepseek(text, max_new_tokens=300, do_sample=True, temperature=0.6, top_p=0.95)
    output = [res["generated_text"] for res in results]
    save_to_db(text, " | ".join(output), "deepseek", chat_session_id)
    return output

def generate_with_gpt41(text, api_key, chat_session_id=None):
    try:
        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model="gpt-4.1",
            messages=[{"role": "user", "content": text}],
            max_tokens=300,
            temperature=0.7
        )
        output_text = response.choices[0].message.content
        save_to_db(text, output_text, "gpt41", chat_session_id)
        return [output_text]
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        if "401" in str(e) or "Unauthorized" in str(e):
            return ["Error calling GPT-4.1: Invalid API key."]
        elif "timed out" in str(e) or "ConnectionError" in str(e) or "Connection error" in str(e):
            return ["Error calling GPT-4.1: Connection issue, please try again later."]
        else:
            return [f"Unexpected error calling GPT-4.1:\n{str(e)}"]

def generate_with_bitnet(text, chat_session_id=None):
    model_id = "microsoft/bitnet-b1.58-2B-4T"

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    )

    messages = [
        {"role": "system", "content": "You are a helpful AI assistant."},
        {"role": "user", "content": text},
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    chat_input = tokenizer(prompt, return_tensors="pt").to(model.device)

    chat_outputs = model.generate(**chat_input, max_new_tokens=50)
    response = tokenizer.decode(chat_outputs[0][chat_input['input_ids'].shape[-1]:], skip_special_tokens=True)

    save_to_db(text, response, "bitnet", chat_session_id)
    return [response]

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

            if selected_model == "gpt2":
                response = generator_gpt2(prompt, max_new_tokens=100, num_return_sequences=1, truncation=True)
                generated_text = response[0]["generated_text"]
            elif selected_model == "deepseek":
                response = generator_deepseek(prompt, max_new_tokens=100, do_sample=True, temperature=0.6, top_p=0.95)
                generated_text = response[0]["generated_text"]
            elif selected_model == "gpt41":
                result = generate_with_gpt41(prompt, api_key, chat_session_id=session_id)
                generated_text = result[0]
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

if __name__ == "__main__":
    app.run(debug=True)