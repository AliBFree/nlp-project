# Spam Sense — 3-Day Beginner NLP Project Plan

## 1. Project title

**Spam Sense** — an SMS spam classifier with a fully client-side web UI.

## 2. Why this project?

- **Genuinely a 3-day scope.** The dataset needs almost no cleaning, the model trains in under a second on a CPU, and the whole "web app" is a single HTML page plus two small JS/CSS files.
- **One clean binary decision** (spam vs. ham), which is much easier to reason about and present than multi-class problems like news-category classification.
- **Classic, well-documented dataset** — thousands of public notebooks exist if you get stuck, which matters a lot the first time you do this kind of project.
- **No GPU, no deep learning, no token limits, no API keys.** TF-IDF + Logistic Regression trains on a laptop CPU in well under a second.
- **Deploys with zero backend**, which sidesteps the hardest part of most "ML web app" tutorials (hosting a Python server) entirely.
- **Easy to explain in a presentation**: "we count word patterns, weight the unusual ones more, and draw a line between spam and ham" is a one-sentence explanation a professor or interviewer can follow immediately.

## 3. Dataset

- **Name:** SMS Spam Collection
- **Source:** [Kaggle mirror (uciml)](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset) / originally [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/228/sms+spam+collection)
- **Samples:** 5,574 raw messages (5,572 after removing duplicates) — well under the 100MB limit (it's ~500KB as plain text)
- **Classes:** `ham` (legitimate, 86.6%) and `spam` (13.4%) — imbalanced but not severely so; handled with `class_weight="balanced"` in training
- **License:** CC BY 4.0 on the UCI repository mirror
- **Why this dataset is a good choice:** it's plain tab-separated text with no missing values, no images, no PII cleanup needed, and it's small enough to commit directly into a GitHub repo — no download scripts or external storage needed at deploy time.

## 4. Project architecture

```
Dataset (SMSSpamCollection.txt)
        │
        ▼
Preprocessing (preprocess.py)
  lowercase → mask URLs/emails/numbers → strip punctuation → clean text
        │
        ▼
Training (train.py)
  TF-IDF vectorizer  +  Logistic Regression
        │
        ▼
Saved Model (model/model.joblib, model/vectorizer.joblib)
        │
        ▼
Export to JSON (export_web_model.py)
  vocabulary + idf weights + coefficients + intercept → web/js/model.json
        │
        ▼
Web UI (web/index.html + web/js/script.js)
  same TF-IDF math re-implemented in plain JavaScript
        │
        ▼
Prediction (in-browser, no server call)
```

## 5. Folder structure

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
│   ├── css/
│   │   └── style.css
│   └── js/
│       ├── script.js
│       └── model.json
├── preprocess.py
├── train.py
├── export_web_model.py
├── inference.py
├── requirements.txt
├── README.md
└── PROJECT_PLAN.md
```

## 6. Development timeline (3 days)

**Day 1 — Data & preprocessing**
- Download the dataset, load it into pandas, inspect class balance and a handful of examples
- Write and test `preprocess.py` (cleaning function + train/test split)
- Sanity-check cleaned text output on a few messages by eye

**Day 2 — Model & evaluation**
- Write `train.py`: TF-IDF vectorizer + Logistic Regression
- Evaluate with accuracy, precision, recall, F1, confusion matrix
- Try 1–2 quick variations (e.g. unigrams only vs. unigrams+bigrams, `class_weight="balanced"` on/off) and keep the best
- Save the model artifacts and export to `model.json` for the browser

**Day 3 — Web UI & deployment**
- Build `index.html` / `style.css` / `script.js`
- Re-implement the TF-IDF + logistic regression math in JavaScript and verify it matches the Python predictions on the same test sentences
- Push to GitHub, enable GitHub Pages, write the README, take screenshots

## 7. Data preprocessing — step by step

1. **Lowercase** everything, so "FREE" and "free" are treated the same.
2. **Mask URLs, emails, and long digit runs** (phone numbers / shortcodes) with placeholder tokens (`urltoken`, `emailtoken`, `numtoken`) instead of deleting them — spam messages are full of these, so *that a link or number existed* is itself a useful signal, even after the raw digits are gone.
3. **Strip punctuation.** TF-IDF works on words, and punctuation adds noise without much signal for this dataset.
4. **Collapse whitespace** left behind by the previous steps.
5. **Train/test split** (80/20), stratified by label so both the train and test sets keep the same ~87/13 ham/spam ratio.

We deliberately **skip** stopword removal, stemming, and lemmatization: TF-IDF already down-weights very common words (like "the", "and") automatically via the idf term, and for a dataset this size, stemming tends to save only a little vocabulary size at the cost of readability when debugging.

## 8. Model choice

**TF-IDF + Logistic Regression.**

- **Simple to explain:** each word (and word pair) gets a weight; a message is spam if its words' weights add up past a threshold.
- **Fast:** trains in a fraction of a second on this dataset, no GPU.
- **Small artifact:** the whole trained model (vocabulary + weights) is under 300KB as JSON — light enough to ship to a browser.
- **Interpretable:** you can inspect which words push a message toward "spam" (e.g. "free", "winner", "urgent", "call now"), which is great for a presentation.
- A DistilBERT model would likely score a few points higher on F1, but would require a JavaScript ML runtime (e.g. transformers.js) and a multi-hundred-MB model file — overkill for a 3-day beginner project and much harder to host for free.

## 9. Training code

See `train.py` in this repository for the complete, runnable training script (TF-IDF vectorization, Logistic Regression training, full evaluation, and artifact export).

## 10. Evaluation

Measured on a held-out 20% test split (1,034 messages):

| Metric | Score | What it means here |
|---|---|---|
| **Accuracy** | 98.5% | Percent of all messages classified correctly (both spam and ham) |
| **Precision (spam)** | 94.6% | Of everything flagged as spam, how much actually was spam — high precision means few legitimate texts get wrongly blocked |
| **Recall (spam)** | 93.1% | Of all the real spam, how much did the model catch — high recall means little spam slips through |
| **F1 (spam)** | 93.8% | The balance between precision and recall in a single number |

Confusion matrix (rows = actual, columns = predicted):

|  | Predicted ham | Predicted spam |
|---|---|---|
| **Actual ham** | 896 | 7 |
| **Actual spam** | 9 | 122 |

Only 7 legitimate messages were misclassified as spam, and only 9 spam messages slipped through as ham — a solid result for a same-day, CPU-only, classical ML model.

## 11. Web UI

See `web/index.html`, `web/css/style.css`, and `web/js/script.js`. Design summary:

- A **phone message-thread mockup** is the centerpiece — every tested message appears as a real message bubble, colored mint (ham) or coral (spam), with a "receipt" line showing the label and a confidence bar.
- A composer bar at the bottom lets you type a message and hit send (or press Enter).
- A side panel explains the color legend, offers one-tap sample messages, and shows the model's headline stats.
- Model loading state is shown in the phone's header ("loading model…" → "ready — runs on-device") so it's clear nothing is being sent to a server.

## 12. GitHub Pages deployment

Because there is **no backend**, deployment is simple:

1. Push this repository to GitHub.
2. In the repo settings, go to **Settings → Pages**.
3. Under "Build and deployment," set **Source: Deploy from a branch**, branch `main`, folder `/web` — *or* if GitHub Pages needs `/web` at the root, move the contents of `web/` to a `docs/` folder (GitHub Pages supports `/docs`) or set up a simple GitHub Action to copy `web/` to the Pages branch.
4. Save. Your site will be live at `https://<username>.github.io/<repo-name>/`.

**What can and cannot run on GitHub Pages:**
- ✅ `web/index.html`, `web/css/style.css`, `web/js/script.js`, `web/js/model.json` — all static files, fully supported.
- ❌ `train.py`, `preprocess.py`, `export_web_model.py`, `inference.py` — these are Python scripts. They only ever run **once, locally (or in a notebook / CI job), offline**, to produce `model.json`. They are not part of the deployed site and GitHub Pages never executes them.
- If you later want live retraining or a Python backend, that would need separate hosting (e.g. Render, Railway, PythonAnywhere, or a small Flask app) — not GitHub Pages. That's not needed for this project since the trained model is small enough to ship as static JSON.

## 13. README

See `README.md` in this repository for the full, deployment-ready professional README (overview, dataset, installation, usage, results, screenshots section, future improvements).

## 14. Possible improvements

1. Probability calibration (Platt scaling / isotonic regression) for more trustworthy confidence scores
2. Character n-grams to catch obfuscated spam like "fr33 c@sh"
3. A small hand-picked "hard examples" test set for extra scrutiny beyond the random split
4. Batch classification: paste in multiple messages at once
5. Multi-language support (current model is English-only)
6. A "why" panel showing the top words that pushed a prediction toward spam
7. A toggle to compare Logistic Regression vs. a Naive Bayes baseline
8. Unit tests to keep the Python and JavaScript preprocessing in sync automatically
9. A dark/light theme toggle for the UI
10. Package it as a browser extension that scans notification text in real time
11. Add per-word contribution highlighting directly inside the message bubble

## 15. Difficulty

| Area | Rating (1–10) |
|---|---|
| Programming difficulty | 3 |
| NLP difficulty | 3 |
| Deployment difficulty | 2 |

## 16. Final recommendation

Compared to other beginner options — fake news detection (messier, less-clean datasets), news category classification (multi-class, harder to explain confidence), or DistilBERT-based sentiment analysis (needs a much bigger, harder-to-host model) — **SMS spam classification hits the sweet spot**: a clean, tiny, well-known dataset; a model that trains in under a second and explains itself in one sentence; and a deployment story that needs nothing more than GitHub Pages. It's realistic to finish well within 3 days, and the phone-message UI gives it a polish that stands out in a portfolio without extra engineering effort.
