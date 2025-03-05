from flask import Flask, render_template, request
from transformers import pipeline, set_seed

app = Flask(__name__)


def load_gpt2():
    """Loads the GPT-2 model and sets a random seed for reproducibility."""
    set_seed(42)
    return pipeline("text-generation", model="gpt2", truncation=True)


def load_bart():
    """Loads the BART model for text summarization."""
    return pipeline("summarization", model="facebook/bart-large-cnn")


def load_bert():
    """Loads the BERT model for masked language modeling."""
    return pipeline("fill-mask", model="bert-base-uncased")


def load_deepseek():
    """Loads the DeepSeek-R1-Distill-Qwen-1.5B model for text generation."""
    return pipeline("text-generation", model="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", truncation=True)


# Load models once
generator_gpt2 = load_gpt2()
summarizer_bart = load_bart()
mask_filler_bert = load_bert()
generator_deepseek = load_deepseek()


def generate_with_gpt2(text):
    """Generates paraphrased text using GPT-2."""
    results = generator_gpt2(text, max_length=50, num_return_sequences=3, truncation=True)
    return [res["generated_text"] for res in results]


def generate_with_bart(text):
    """Summarizes text using BART."""
    summary = summarizer_bart(text, max_length=130, min_length=30, do_sample=False)
    return [summary[0]["summary_text"]]


def generate_with_bert(text):
    """Fills in the [MASK] token in a sentence using BERT."""
    if "[MASK]" not in text:
        return ["Please include a [MASK] token in the input text."]

    predictions = mask_filler_bert(text)
    return [f"{pred['sequence']} (Confidence: {pred['score']:.2%})" for pred in predictions[:5]]


def generate_with_deepseek(text):
    """Generates text using DeepSeek-R1-Distill-Qwen-1.5B."""
    results = generator_deepseek(text, max_length=300, do_sample=True, temperature=0.6, top_p=0.95)
    return [res["generated_text"] for res in results]


@app.route("/", methods=["GET", "POST"])
def home():
    """
    Renders the homepage with a text input form.
    Generates paraphrased text, summarized text, masked word predictions, or general text generation.
    """
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