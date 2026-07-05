"""
inference.py
-------------
Loads the saved model + vectorizer and predicts on new SMS text.
This file is for local use only (e.g. testing, or a small Flask API)
-- the deployed web app on GitHub Pages does NOT use this file, since
GitHub Pages cannot run Python. See web/js/script.js for the browser
version of this same logic.

Usage:
    python inference.py "Congratulations! You have won a free prize, call now!"
"""

import sys
import joblib
from preprocess import clean_text

MODEL_PATH = "model/model.joblib"
VECTORIZER_PATH = "model/vectorizer.joblib"


def load_artifacts():
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    return model, vectorizer


def predict(text: str, model=None, vectorizer=None) -> dict:
    if model is None or vectorizer is None:
        model, vectorizer = load_artifacts()

    cleaned = clean_text(text)
    vec = vectorizer.transform([cleaned])
    label_num = model.predict(vec)[0]
    proba = model.predict_proba(vec)[0]

    label = "spam" if label_num == 1 else "ham"
    confidence = float(proba[label_num])

    return {"label": label, "confidence": round(confidence, 4)}


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('Usage: python inference.py "your message here"')
        sys.exit(1)

    message = " ".join(sys.argv[1:])
    result = predict(message)
    print(f"Message : {message}")
    print(f"Label   : {result['label']}")
    print(f"Confidence: {result['confidence'] * 100:.1f}%")
