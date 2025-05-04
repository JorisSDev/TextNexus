import torch
from transformers import pipeline, set_seed, AutoModelForCausalLM, AutoTokenizer
from openai import OpenAI
from database import init_db
import sqlite3

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
    set_seed(42)
    return pipeline("text-generation", model="gpt2", truncation=True)

def load_bart():
    return pipeline("summarization", model="facebook/bart-large-cnn")

def load_bert():
    return pipeline("fill-mask", model="bert-base-uncased")

def load_deepseek():
    return pipeline("text-generation", model="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", truncation=True)

generator_gpt2 = load_gpt2()
summarizer_bart = load_bart()
mask_filler_bert = load_bert()
generator_deepseek = load_deepseek()

def generate_with_gpt2(text, chat_session_id=None):
    results = generator_gpt2(text, max_new_tokens=150, num_return_sequences=3)
    output = [res["generated_text"] for res in results]
    save_to_db(text, " | ".join(output), "gpt2", chat_session_id)
    return output

def generate_with_bart(text, chat_session_id=None):
    summary = summarizer_bart(text, max_length=130, min_length=30, do_sample=False)
    output = [summary[0]["summary_text"]]
    save_to_db(text, output[0], "bart", chat_session_id)
    return output

def generate_with_bert(text, chat_session_id=None):
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
