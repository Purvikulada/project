# app.py
from flask import Flask, request, jsonify, render_template
import joblib
import os
import re

MODEL_PATH = os.path.join("model", "sentiment_model.joblib")

app = Flask(__name__, static_folder="static", template_folder="templates")

def simple_clean(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# Load model at startup
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Please run train_model.py first.")
model = joblib.load(MODEL_PATH)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    if not data or "review" not in data:
        return jsonify({"error": "Missing 'review' in request"}), 400
    review = data["review"]
    cleaned = simple_clean(review)
    pred = model.predict([cleaned])[0]
    proba = None
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba([cleaned])[0].tolist()
        # pair with classes
        classes = model.classes_.tolist()
        probs = dict(zip(classes, proba))
    else:
        probs = {}
    return jsonify({"prediction": pred, "probabilities": probs})

if __name__ == "__main__":
    app.run(debug=True)
