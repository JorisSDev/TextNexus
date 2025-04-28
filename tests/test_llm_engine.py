from llm_engine import generate_with_gpt2, generate_with_bart, generate_with_bert, generate_with_deepseek

def test_generate_with_gpt2_returns_list():
    text = "Once upon a time"
    outputs = generate_with_gpt2(text)
    assert isinstance(outputs, list)
    assert len(outputs) > 0

def test_generate_with_bart_returns_summary():
    text = "Artificial Intelligence is a field of computer science that focuses on creating machines capable of performing tasks that typically require human intelligence."
    outputs = generate_with_bart(text)
    assert isinstance(outputs, list)
    assert len(outputs) == 1

def test_generate_with_bert_requires_mask():
    text = "The capital of France is [MASK]."
    outputs = generate_with_bert(text)
    assert isinstance(outputs, list)
    assert len(outputs) > 0

def test_generate_with_bert_without_mask():
    text = "The capital of France is Paris."
    outputs = generate_with_bert(text)
    assert outputs == ["Please include a [MASK] token in the input text."]

def test_generate_with_deepseek_returns_list():
    text = "Deep learning models are very powerful tools for"
    outputs = generate_with_deepseek(text)
    assert isinstance(outputs, list)
    assert len(outputs) > 0