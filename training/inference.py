"""
inference.py
-------------
Loads the saved model + vectorizer for a given language and predicts
on new SMS text. Local/CLI use only -- the deployed web app does not
use this file (GitHub Pages can't run Python); see js/script.js for
the browser-side version of this same logic.

Usage:
    python inference.py --lang en "Congratulations! You have won a free prize, call now!"
    python inference.py --lang fa "..."
"""

import argparse
import joblib
from preprocess import clean_text

MODEL_DIR = "model"


def load_artifacts(lang: str):
    model = joblib.load(f"{MODEL_DIR}/model_{lang}.joblib")
    vectorizer = joblib.load(f"{MODEL_DIR}/vectorizer_{lang}.joblib")
    return model, vectorizer


def predict(text: str, lang: str = "en", model=None, vectorizer=None) -> dict:
    if model is None or vectorizer is None:
        model, vectorizer = load_artifacts(lang)

    cleaned = clean_text(text, lang)
    vec = vectorizer.transform([cleaned])
    label_num = model.predict(vec)[0]
    proba = model.predict_proba(vec)[0]

    label = "spam" if label_num == 1 else "ham"
    confidence = float(proba[label_num])

    return {"label": label, "confidence": round(confidence, 4)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang", choices=["en", "fa"], default="en")
    parser.add_argument("message", nargs="+")
    args = parser.parse_args()

    message = " ".join(args.message)
    result = predict(message, lang=args.lang)
    print(f"Language: {args.lang}")
    print(f"Message : {message}")
    print(f"Label   : {result['label']}")
    print(f"Confidence: {result['confidence'] * 100:.1f}%")
