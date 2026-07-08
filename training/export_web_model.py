"""
export_web_model.py
--------------------
Converts trained scikit-learn artifacts (model/model_<lang>.joblib,
model/vectorizer_<lang>.joblib) into JSON files the browser can load
and run entirely client-side.

Run once per language, after train.py:
    python export_web_model.py --lang en
    python export_web_model.py --lang fa

Writes to ../js/model_<lang>.json (i.e. the js/ folder at the repo
root, next to script.js) since that's what GitHub Pages actually serves.
"""

import argparse
import json
import os
import joblib

MODEL_DIR = "model"
OUTPUT_DIR = "../js"  # repo_root/js -- adjust if your layout differs


def export(lang: str):
    vectorizer_path = f"{MODEL_DIR}/vectorizer_{lang}.joblib"
    model_path = f"{MODEL_DIR}/model_{lang}.joblib"

    vectorizer = joblib.load(vectorizer_path)
    model = joblib.load(model_path)

    vocabulary = {term: int(idx) for term, idx in vectorizer.vocabulary_.items()}
    idf = vectorizer.idf_.tolist()
    coef = model.coef_[0].tolist()
    intercept = float(model.intercept_[0])

    web_model = {
        "lang": lang,
        "vocabulary": vocabulary,
        "idf": idf,
        "coef": coef,
        "intercept": intercept,
        "ngram_range": [1, 2],
        "labels": {"0": "ham", "1": "spam"},
    }

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = f"{OUTPUT_DIR}/model_{lang}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(web_model, f, ensure_ascii=False)

    size_kb = os.path.getsize(out_path) / 1024
    print(f"[{lang}] Exported {out_path} ({size_kb:.1f} KB, {len(vocabulary)} vocab terms)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang", choices=["en", "fa", "all"], default="all")
    args = parser.parse_args()

    langs = ["en", "fa"] if args.lang == "all" else [args.lang]
    for lang in langs:
        model_exists = os.path.exists(f"{MODEL_DIR}/model_{lang}.joblib")
        if not model_exists:
            print(f"[{lang}] Skipped -- no trained model found. Run `python train.py --lang {lang}` first.")
            continue
        export(lang)
