# Spam Sense — SMS Spam Classifier

A tiny, fully client-side SMS spam classifier. Type a text message into
the browser and it's classified as **spam** or **ham** (legitimate)
instantly — no server, no API call, no network request after the page
loads. The model itself (a few hundred KB of JSON) runs directly in
JavaScript.

**[Live demo →](https://alibfree.github.io/nlp-project/)**

![status](https://img.shields.io/badge/status-learning_project-blue)
![python](https://img.shields.io/badge/python-3.10%2B-blue)
![license](https://img.shields.io/badge/license-MIT-green)

---

## Overview

This project trains a classical machine learning model (TF-IDF +
Logistic Regression) to detect spam SMS messages, then re-implements
the trained model's math in plain JavaScript so the whole thing can be
hosted for free on GitHub Pages. There is no backend in production —
Python is only used offline, to train the model once.

- **Preprocessing → training → evaluation** in Python (`preprocess.py`, `train.py`)
- **Model export** to a portable JSON file (`export_web_model.py`)
- **Inference** re-implemented from scratch in the browser (`web/js/script.js`)
- **UI**: a phone-style message thread where each tested message becomes a message bubble, color-coded and confidence-scored

## Dataset

[SMS Spam Collection](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset)
(UCI Machine Learning Repository / Kaggle), by Tiago A. Almeida and José
María Gómez Hidalgo.

- 5,572 unique SMS messages (English) after de-duplication
- Two classes: `ham` (86.6%) and `spam` (13.4%)
- Licensed CC BY 4.0 on the UCI repository mirror
- Plain text, tab-separated, no missing values — almost no cleaning required

## Installation

```bash
git clone <your-repo-url>
cd sms-spam-classifier
pip install -r requirements.txt
```

Requires Python 3.10+. No GPU needed — training takes well under a
second on a laptop CPU.

## Usage

**1. Train the model:**
```bash
python train.py
```
This reads `data/SMSSpamCollection.txt`, trains a TF-IDF + Logistic
Regression pipeline, prints evaluation metrics, and saves
`model/model.joblib` and `model/vectorizer.joblib`.

**2. Export the model for the browser:**
```bash
python export_web_model.py
```
Writes `web/js/model.json` — the vocabulary, idf weights, and
logistic regression coefficients as plain JSON.

**3. Try it from the command line (optional):**
```bash
python inference.py "Congratulations! You have won a free prize, call now!"
```

**4. Open the web app:**
Open `web/index.html` directly in a browser, or serve it locally:
```bash
cd web && python3 -m http.server 8000
```
Then visit `http://localhost:8000`.

## Results

Evaluated on a held-out 20% test split (1,034 messages), stratified by class:

| Metric | Score |
|---|---|
| Accuracy | 98.5% |
| Precision (spam) | 94.6% |
| Recall (spam) | 93.1% |
| F1 (spam) | 93.8% |

Confusion matrix:

| | Predicted ham | Predicted spam |
|---|---|---|
| **Actual ham** | 896 | 7 |
| **Actual spam** | 9 | 122 |

## Screenshots

*(Add a screenshot of the message-thread UI here once deployed, e.g. `assets/screenshot.png`.)*

## Project structure

```
sms-spam-classifier/
├── data/
│   └── SMSSpamCollection.txt
├── model/
│   ├── model.joblib
│   ├── vectorizer.joblib
│   └── metrics.json
├── web/
│   ├── index.html
│   ├── css/style.css
│   └── js/
│       ├── script.js
│       └── model.json
├── preprocess.py
├── train.py
├── export_web_model.py
├── inference.py
├── requirements.txt
└── README.md
```

## Future improvements

1. Add a probability calibration step (Platt scaling) for better-calibrated confidence scores
2. Try character n-grams to catch obfuscated spam (e.g. "fr33 pr1ze")
3. Add a small held-out "hard examples" test set of borderline messages
4. Support batch classification (paste multiple messages at once)
5. Add multi-language support (the current model is English-only)
6. Track false positives/negatives the user flags, for active-learning style review
7. Add a "why" explanation showing top contributing words per prediction
8. Compare against a Naive Bayes baseline and let the user toggle between models
9. Add unit tests for `preprocess.py` and the JS tokenizer to keep them in sync
10. Package as a browser extension that scans incoming messages/notifications

## License

Code: MIT. Dataset: CC BY 4.0 (SMS Spam Collection, Almeida & Hidalgo, UCI Machine Learning Repository) — please cite the original authors if you reuse the data.
