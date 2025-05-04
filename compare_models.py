import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from math import pi
import matplotlib.cm as cm
from llm_engine import (
    generate_with_gpt2,
    generate_with_bart,
    generate_with_bert,
    generate_with_deepseek,
)
from sentence_transformers import SentenceTransformer, util
from collections import Counter
import os
os.makedirs("model_graphs", exist_ok=True)

# Define input prompts
inputs = {
    "gpt2": "Once upon a time in a faraway land,",
    "bart": (
        "Artificial Intelligence is transforming various industries by enabling automation, personalization, "
        "and predictive analytics. This shift is evident in fields ranging from healthcare to finance."
    ),
    "bert": "The capital of Germany is [MASK].",
    "deepseek": "Modern software systems must be scalable, reliable, and easy to maintain. To achieve this,"
}

# Run models and gather output lengths
results = {}
print("Running comparative output test...\n")
for model, prompt in inputs.items():
    if model == "gpt2":
        outputs = generate_with_gpt2(prompt)
    elif model == "bart":
        outputs = generate_with_bart(prompt)
    elif model == "bert":
        outputs = generate_with_bert(prompt)
    elif model == "deepseek":
        outputs = generate_with_deepseek(prompt)
    else:
        continue

    print(f"--- {model.upper()} ---")
    for i, out in enumerate(outputs[:3], 1):
        print(f"[{i}] {out.strip()}\n")

    total_tokens = sum(len(out.split()) for out in outputs)
    results[model.upper()] = total_tokens

# Build DataFrame for visualization
df = pd.DataFrame(list(results.items()), columns=["Model", "Token Count"])
df["Prompt Index"] = np.arange(1, len(df) + 1)

model = SentenceTransformer("all-MiniLM-L6-v2")
relevance_scores = []
diversity_scores = []

for model_name, prompt in inputs.items():
    output_text = " ".join(generate_with_gpt2(prompt) if model_name == "gpt2"
                           else generate_with_bart(prompt) if model_name == "bart"
                           else generate_with_bert(prompt) if model_name == "bert"
                           else generate_with_deepseek(prompt))

    # Relevance
    prompt_emb = model.encode(prompt, convert_to_tensor=True)
    response_emb = model.encode(output_text, convert_to_tensor=True)
    similarity = float(util.cos_sim(prompt_emb, response_emb))
    relevance_scores.append(similarity)

    # Lexical Diversity
    tokens = output_text.split()
    diversity = len(set(tokens)) / len(tokens) if tokens else 0
    diversity_scores.append(diversity)

df["Prompt Relevance"] = relevance_scores
df["Lexical Diversity"] = diversity_scores

# --- Additional metrics: Semantic Coherence, Repetition Rate, Avg Sentence Length ---
semantic_scores = np.random.uniform(0.6, 0.95, len(df))  # Placeholder
repetition_scores = []
avg_sentence_lengths = []

for model_name, prompt in inputs.items():
    output_text = " ".join(generate_with_gpt2(prompt) if model_name == "gpt2"
                           else generate_with_bart(prompt) if model_name == "bart"
                           else generate_with_bert(prompt) if model_name == "bert"
                           else generate_with_deepseek(prompt))

    tokens = output_text.split()
    counts = Counter(tokens)
    repetitions = sum(c for c in counts.values() if c > 1)
    repetition_scores.append(repetitions / len(tokens) if tokens else 0)

    sentences = output_text.split(".")
    sentence_lengths = [len(s.split()) for s in sentences if s.strip()]
    avg_len = np.mean(sentence_lengths) if sentence_lengths else 0
    avg_sentence_lengths.append(avg_len)

df["Semantic Coherence"] = semantic_scores
df["Repetition Rate"] = repetition_scores
df["Avg Sentence Length"] = avg_sentence_lengths

# Individual plots for each metric

# Token Count
fig, ax = plt.subplots()
ax.bar(df["Model"], df["Token Count"], color="cornflowerblue")
ax.set_title("Output Token Count")
ax.set_ylabel("Tokens")
plt.tight_layout()
plt.savefig("model_graphs/token_count.png")
plt.close(fig)

# Prompt Relevance
fig, ax = plt.subplots()
ax.bar(df["Model"], df["Prompt Relevance"], color="mediumpurple")
ax.set_title("Prompt Relevance")
ax.set_ylim(0, 1)
plt.tight_layout()
plt.savefig("model_graphs/prompt_relevance.png")
plt.close(fig)

# Lexical Diversity
fig, ax = plt.subplots()
ax.bar(df["Model"], df["Lexical Diversity"], color="goldenrod")
ax.set_title("Lexical Diversity")
ax.set_ylim(0, 1)
plt.tight_layout()
plt.savefig("model_graphs/lexical_diversity.png")
plt.close(fig)

# Semantic Coherence
fig, ax = plt.subplots()
ax.bar(df["Model"], df["Semantic Coherence"], color="seagreen")
ax.set_title("Semantic Coherence (Simulated)")
ax.set_ylim(0.5, 1.0)
plt.tight_layout()
plt.savefig("model_graphs/semantic_coherence.png")
plt.close(fig)

# Repetition Rate
fig, ax = plt.subplots()
ax.bar(df["Model"], df["Repetition Rate"], color="tomato")
ax.set_title("Repetition Rate")
ax.set_ylim(0, 1)
plt.tight_layout()
plt.savefig("model_graphs/repetition_rate.png")
plt.close(fig)

# Avg Sentence Length
fig, ax = plt.subplots()
ax.bar(df["Model"], df["Avg Sentence Length"], color="steelblue")
ax.set_title("Avg Sentence Length")
ax.set_ylabel("Words")
plt.tight_layout()
plt.savefig("model_graphs/avg_sentence_length.png")
plt.close(fig)

# --- Additional Visualizations: Radar, Boxplot, Stacked Bar ---

# Normalize metrics for radar chart
def normalize(series):
    return (series - series.min()) / (series.max() - series.min())

radar_data = df.set_index("Model")[["Token Count", "Prompt Relevance", "Lexical Diversity", "Semantic Coherence"]]
radar_data = radar_data.apply(normalize)

# Radar chart
labels = radar_data.columns
num_vars = len(labels)

angles = [n / float(num_vars) * 2 * pi for n in range(num_vars)]
angles += angles[:1]  # Complete the loop

fig = plt.figure(figsize=(8, 8))
ax = plt.subplot(111, polar=True)
colors = cm.tab10.colors

for i, (model, row) in enumerate(radar_data.iterrows()):
    values = row.tolist()
    values += values[:1]
    ax.plot(angles, values, label=model, color=colors[i % len(colors)])
    ax.fill(angles, values, alpha=0.1, color=colors[i % len(colors)])

ax.set_xticks(angles[:-1])
ax.set_xticklabels(labels)
ax.set_title("Model Comparison Radar Chart")
ax.legend(loc="upper right", bbox_to_anchor=(1.1, 1.1))
plt.tight_layout()
plt.savefig("model_graphs/radar_chart.png")
plt.close(fig)


# --- Collect multiple samples per model for sentence length distribution ---
all_sentence_lengths = {}  # for real boxplot data

for model_name, prompt in inputs.items():
    all_lengths = []
    outputs = (
        generate_with_gpt2(prompt) if model_name == "gpt2"
        else generate_with_bart(prompt) if model_name == "bart"
        else generate_with_bert(prompt) if model_name == "bert"
        else generate_with_deepseek(prompt)
    )
    for output in outputs[:5]:
        sentences = output.split(".")
        lengths = [len(s.split()) for s in sentences if s.strip()]
        all_lengths.extend(lengths)

    all_sentence_lengths[model_name.upper()] = all_lengths

# Boxplot for Avg Sentence Length (real distribution)
fig = plt.figure(figsize=(6, 5))
plt.boxplot(all_sentence_lengths.values(), labels=all_sentence_lengths.keys())
plt.title("Distribution of Avg Sentence Length per Model")
plt.ylabel("Words")
plt.grid(True)
plt.tight_layout()
plt.savefig("model_graphs/avg_sentence_length_boxplot.png")
plt.close(fig)

# Stacked horizontal bar for Repetition vs Unique tokens
fig, ax = plt.subplots(figsize=(8, 5))
repeat = df["Repetition Rate"]
unique = 1 - repeat
ax.barh(df["Model"], unique, color="mediumseagreen", label="Unique")
ax.barh(df["Model"], repeat, left=unique, color="salmon", label="Repeated")
ax.set_title("Token Uniqueness vs Repetition")
ax.set_xlabel("Proportion")
ax.legend()
plt.tight_layout()
plt.savefig("model_graphs/repetition_stackbar.png")
plt.close(fig)