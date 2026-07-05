"""
preprocess.py
--------------
Loads the raw SMS Spam Collection file and turns it into a clean,
model-ready train/test split.

Dataset format: tab-separated file with two columns and no header:
    <label>\t<message>
    ham     Go until jurong point, crazy..
    spam    Free entry in 2 a wkly comp...

We keep preprocessing intentionally light. TF-IDF already handles a lot
of the heavy lifting (it down-weights very common words on its own), so
we only do the steps that clearly help a simple linear model:

1. Lowercase the text
2. Strip URLs, email addresses and phone numbers down to placeholder tokens
   (spam messages are full of these, so keeping a *signal* that one existed
   is more useful than deleting it outright)
3. Remove punctuation and digits noise (but keep the placeholder tokens)
4. Collapse extra whitespace
5. Train/test split, stratified by label so both classes keep their ratio
"""

import re
import string
import pandas as pd
from sklearn.model_selection import train_test_split

RAW_DATA_PATH = "data/SMSSpamCollection.txt"

URL_RE = re.compile(r"(https?://\S+|www\.\S+)")
EMAIL_RE = re.compile(r"\S+@\S+")
PHONE_RE = re.compile(r"\b\d{5,}\b")  # 5+ digit runs (phone numbers, shortcodes)
MULTI_SPACE_RE = re.compile(r"\s+")
PUNCT_TABLE = str.maketrans("", "", string.punctuation)


def clean_text(text: str) -> str:
    """Apply the full cleaning pipeline to a single message."""
    text = text.lower()
    text = URL_RE.sub(" urltoken ", text)
    text = EMAIL_RE.sub(" emailtoken ", text)
    text = PHONE_RE.sub(" numtoken ", text)
    text = text.translate(PUNCT_TABLE)
    text = MULTI_SPACE_RE.sub(" ", text).strip()
    return text


def load_dataset(path: str = RAW_DATA_PATH) -> pd.DataFrame:
    """Read the raw tab-separated file into a DataFrame with clean text."""
    df = pd.read_csv(path, sep="\t", header=None, names=["label", "text"])
    df = df.dropna(subset=["text"]).drop_duplicates(subset=["text"]).reset_index(drop=True)
    df["clean_text"] = df["text"].apply(clean_text)
    df["label_num"] = (df["label"] == "spam").astype(int)  # ham=0, spam=1
    return df


def get_train_test_split(path: str = RAW_DATA_PATH, test_size: float = 0.2, random_state: int = 42):
    """Return X_train, X_test, y_train, y_test ready for vectorization."""
    df = load_dataset(path)
    X_train, X_test, y_train, y_test = train_test_split(
        df["clean_text"],
        df["label_num"],
        test_size=test_size,
        random_state=random_state,
        stratify=df["label_num"],
    )
    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    df = load_dataset()
    print(f"Loaded {len(df)} messages after de-duplication")
    print(df["label"].value_counts())
    print("\nExample before/after cleaning:")
    print("RAW:  ", df["text"].iloc[2])
    print("CLEAN:", df["clean_text"].iloc[2])
