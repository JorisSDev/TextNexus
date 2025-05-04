from flask import Blueprint, request, jsonify
from llm_engine import (
    generate_with_gpt2,
    generate_with_bart,
    generate_with_bert,
    generate_with_deepseek,
    generate_with_gpt41,
    generate_with_bitnet
)

api = Blueprint('api', __name__, url_prefix='/api')

@api.route('/generate/gpt2', methods=['POST'])
def api_gpt2():
    data = request.get_json()
    prompt = data.get("prompt")
    if not prompt:
        return jsonify({"error": "Missing 'prompt'"}), 400
    result = generate_with_gpt2(prompt)
    print("Generated result:", result)
    return jsonify(result)

@api.route('/summarize/bart', methods=['POST'])
def api_bart():
    data = request.get_json()
    text = data.get("text")
    if not text:
        return jsonify({"error": "Missing 'text'"}), 400
    result = generate_with_bart(text)
    return jsonify(result)

@api.route('/fillmask/bert', methods=['POST'])
def api_bert():
    data = request.get_json()
    text = data.get("text")
    if not text:
        return jsonify({"error": "Missing 'text'"}), 400
    result = generate_with_bert(text)
    return jsonify(result)

@api.route('/generate/deepseek', methods=['POST'])
def api_deepseek():
    data = request.get_json()
    prompt = data.get("prompt")
    if not prompt:
        return jsonify({"error": "Missing 'prompt'"}), 400
    result = generate_with_deepseek(prompt)
    return jsonify(result)

@api.route('/generate/gpt41', methods=['POST'])
def api_gpt41():
    data = request.get_json()
    prompt = data.get("prompt")
    api_key = data.get("api_key")
    if not prompt or not api_key:
        return jsonify({"error": "Missing 'prompt' or 'api_key'"}), 400
    result = generate_with_gpt41(prompt, api_key)
    return jsonify(result)

@api.route('/generate/bitnet', methods=['POST'])
def api_bitnet():
    data = request.get_json()
    prompt = data.get("prompt")
    if not prompt:
        return jsonify({"error": "Missing 'prompt'"}), 400
    result = generate_with_bitnet(prompt)
    return jsonify(result)