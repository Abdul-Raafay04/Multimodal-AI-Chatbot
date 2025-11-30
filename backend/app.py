from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

from pipeline import answer_text_query, answer_image_query

app = Flask(__name__)
CORS(app)

limiter = Limiter(
    key_func=get_remote_address,
    app=app,
    default_limits=["10 per minute"],
)

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})

@app.route("/query/text", methods=["POST"])
@limiter.limit("10 per minute")
def query_text():
    data = request.get_json()
    if "question" not in data:
        return jsonify({"error": "Missing question"}), 400

    try:
        answer = answer_text_query(data["question"])
        return jsonify({"answer": answer})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/query/image", methods=["POST"])
@limiter.limit("5 per minute")
def query_image():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    question = request.form.get("question", "")

    try:
        answer = answer_image_query(file.read(), question)
        return jsonify({"answer": answer})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
