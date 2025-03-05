from flask import Flask, render_template, request

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def home():
    output = None
    if request.method == "POST":
        input_text = request.form["input_text"]
        output = {
            "summary": f"Summarized: {input_text[:50]}...",
            "rephrase": f"Rephrased using masking: {' '.join(reversed(input_text.split()))}",
            "extended": f"Extended: {input_text} with more explanation."
        }
    return render_template("index.html", output=output)

if __name__ == "__main__":
    app.run(debug=True)