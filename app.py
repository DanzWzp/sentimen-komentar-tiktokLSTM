"""
Web interface sederhana untuk deteksi sentimen komentar (LSTM + Word2Vec).

Jalankan:  python app.py
Lalu buka: http://127.0.0.1:5000
"""

from flask import Flask, jsonify, render_template, request

import predict as engine

app = Flask(__name__)


@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    text = ""
    if request.method == "POST":
        text = (request.form.get("comment") or "").strip()
        if text:
            result = engine.predict(text)
    return render_template("index.html", result=result, text=text,
                           info=engine.model_info())


@app.route("/api/predict", methods=["POST"])
def api_predict():
    """Endpoint JSON: {"comment": "..."} -> hasil prediksi."""
    data = request.get_json(silent=True) or {}
    text = (data.get("comment") or "").strip()
    if not text:
        return jsonify({"error": "Field 'comment' wajib diisi."}), 400
    return jsonify(engine.predict(text))


if __name__ == "__main__":
    print("Memuat model sentimen ...")
    engine.load()
    print("Model siap:", engine.model_info())
    app.run(host="127.0.0.1", port=5000, debug=False)
