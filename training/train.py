"""
train.py
--------
Trains a TF-IDF + Logistic Regression SMS spam classifier for a given
language, evaluates it, and exports:

  model/model_<lang>.joblib
  model/vectorizer_<lang>.joblib
  model/metrics_<lang>.json

Usage:
    python train.py --lang en
    python train.py --lang fa
"""

import argparse
import json
import time
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

from preprocess import get_train_test_split

MODEL_DIR = "model"


def train(lang: str):
    print(f"[{lang}] Loading and splitting data...")
    X_train, X_test, y_train, y_test = get_train_test_split(lang=lang)
    print(f"[{lang}] Train size: {len(X_train)} | Test size: {len(X_test)}")

    print(f"[{lang}] Vectorizing text with TF-IDF...")
    vectorizer = TfidfVectorizer(
        max_features=5000,   # keep the model small and fast
        ngram_range=(1, 2),  # unigrams + bigrams catch phrases like "free entry"
        min_df=2,
    )
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    print(f"[{lang}] Training Logistic Regression...")
    start = time.time()
    model = LogisticRegression(max_iter=1000, class_weight="balanced")
    model.fit(X_train_vec, y_train)
    train_time = time.time() - start
    print(f"[{lang}] Training finished in {train_time:.2f}s (CPU only)")

    print(f"\n[{lang}] Evaluating on held-out test set...")
    y_pred = model.predict(X_test_vec)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred),
    }
    cm = confusion_matrix(y_test, y_pred).tolist()

    print(json.dumps(metrics, indent=2))
    print(f"\n[{lang}] Confusion matrix [[TN, FP], [FN, TP]]:")
    print(cm)
    print(f"\n[{lang}] Full classification report:")
    print(classification_report(y_test, y_pred, target_names=["ham", "spam"]))

    print(f"\n[{lang}] Saving model artifacts to {MODEL_DIR}/ ...")
    joblib.dump(model, f"{MODEL_DIR}/model_{lang}.joblib")
    joblib.dump(vectorizer, f"{MODEL_DIR}/vectorizer_{lang}.joblib")

    with open(f"{MODEL_DIR}/metrics_{lang}.json", "w") as f:
        json.dump({"metrics": metrics, "confusion_matrix": cm}, f, indent=2)

    print(f"[{lang}] Done. Artifacts saved.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang", choices=["en", "fa"], default="en")
    args = parser.parse_args()
    train(args.lang)
