from flask import Flask, render_template, request
from transformers import pipeline, set_seed
import sqlite3

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
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
    )
    """)
    conn.commit()
    conn.close()


init_db()

def save_to_db(input_text, output_text, model):
    conn = sqlite3.connect("textnexus.db")
    cursor = conn.cursor()
    cursor.execute("INSERT INTO text_records (input_text, output_text, model) VALUES (?, ?, ?)",
                   (input_text, output_text, model))
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


def generate_with_gpt2(text):
    results = generator_gpt2(text, max_length=50, num_return_sequences=3, truncation=True)
    output = [res["generated_text"] for res in results]
    save_to_db(text, " | ".join(output), "gpt2")
    return output


def generate_with_bart(text):
    summary = summarizer_bart(text, max_length=130, min_length=30, do_sample=False)
    output = [summary[0]["summary_text"]]
    save_to_db(text, output[0], "bart")
    return output


def generate_with_bert(text):
    """Fills in the [MASK] token in a sentence using BERT."""
    if "[MASK]" not in text:
        return ["Please include a [MASK] token in the input text."]

    predictions = mask_filler_bert(text)
    output = [f"{pred['sequence']} (Confidence: {pred['score']:.2%})" for pred in predictions[:5]]
    save_to_db(text, " | ".join(output), "bert")
    return output


def generate_with_deepseek(text):
    results = generator_deepseek(text, max_length=300, do_sample=True, temperature=0.6, top_p=0.95)
    output = [res["generated_text"] for res in results]
    save_to_db(text, " | ".join(output), "deepseek")
    return output


@app.route("/", methods=["GET", "POST"])
def home():
    output = None
    selected_model = "gpt2"  # Default model
    input_text = ""

    if request.method == "POST":
        input_text = request.form.get("input_text", "").strip()
        selected_model = request.form.get("model", "gpt2")

        if input_text:
            if selected_model == "gpt2":
                output = generate_with_gpt2(input_text)
            elif selected_model == "bart":
                output = generate_with_bart(input_text)
            elif selected_model == "bert":
                output = generate_with_bert(input_text)
            elif selected_model == "deepseek":
                output = generate_with_deepseek(input_text)
            else:
                output = ["Invalid model selected."]

    return render_template("index.html", output=output, selected_model=selected_model, input_text=input_text)


if __name__ == "__main__":
    app.run(debug=True)