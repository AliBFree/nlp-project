"""
export_web_model.py
--------------------
Converts the trained scikit-learn artifacts (model/model.joblib,
model/vectorizer.joblib) into a single JSON file that a plain
JavaScript file can load and run entirely in the browser.

Why not TensorFlow.js? A TF-IDF + Logistic Regression model is just
"vocabulary + idf weights + a weight per word + an intercept". That is
plain data, so we can reimplement the (tiny) TF-IDF math directly in
JavaScript and skip an entire model-conversion toolchain. This keeps
the whole thing GitHub Pages friendly with zero backend.

Run after train.py:
    python export_web_model.py
"""

import json
import joblib

vectorizer = joblib.load("model/vectorizer.joblib")
model = joblib.load("model/model.joblib")

vocabulary = {term: int(idx) for term, idx in vectorizer.vocabulary_.items()}
idf = vectorizer.idf_.tolist()
coef = model.coef_[0].tolist()
intercept = float(model.intercept_[0])

web_model = {
    "vocabulary": vocabulary,   # term -> feature index
    "idf": idf,                 # idf weight per feature index
    "coef": coef,                # logistic regression weight per feature index
    "intercept": intercept,
    "ngram_range": [1, 2],
    "labels": {"0": "ham", "1": "spam"},
}

with open("web/js/model.json", "w") as f:
    json.dump(web_model, f)

size_kb = len(json.dumps(web_model)) / 1024
print(f"Exported web/js/model.json ({size_kb:.1f} KB, {len(vocabulary)} vocab terms)")
