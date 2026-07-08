"""
preprocess.py
--------------
Loads a raw SMS spam file and turns it into a clean, model-ready
train/test split. Supports two languages:

    lang="en"  -> data/SMSSpamCollection.txt          (tab-separated, no header)
    lang="fa"  -> data/persian_sms_spam_collection.txt (same format: label<TAB>text)

Both files must follow the exact same format:
    <label>\t<message>
    ham     Go until jurong point, crazy..
    spam    Free entry in 2 a wkly comp...

English cleaning (unchanged from the original version):
1. Lowercase
2. Mask URLs / emails / long digit runs with placeholder tokens
3. Strip punctuation
4. Collapse whitespace
5. Train/test split, stratified by label

Persian cleaning needs a few extra normalization steps that English
doesn't, because raw Persian text is much less standardized than raw
English text at the character level:

1. Normalize Arabic-script character variants to their Persian forms
   (e.g. Arabic ي -> Persian ی, Arabic ك -> Persian ک). Without this,
   the same word can silently end up as two different vocabulary
   entries depending on which keyboard/source typed it.
2. Normalize digits: Persian (۰-۹) and Arabic-Indic (٠-٩) digits are
   both mapped to ASCII digits before the "mask long digit runs" step,
   so phone numbers/codes get caught regardless of which digit set
   was used.
3. Remove zero-width non-joiner (ZWNJ, U+200C) instead of treating it
   as a normal character. Persian uses ZWNJ inside compound words
   (e.g. می‌روم) but it's used inconsistently across sources, so the
   same word can appear with or without it. We normalize by removing
   it rather than keeping two versions of the same word.
4. Strip Persian/Arabic punctuation (، ؛ ؟ etc.) in addition to the
   standard ASCII punctuation set.
5. No stemming/lemmatization here either, for the same reason as
   English: TF-IDF's idf term already discounts very common words,
   and adding a Persian stemmer is a real dependency for a dataset
   this size where it's unlikely to move the needle much.
"""

import re
import string
import pandas as pd
from sklearn.model_selection import train_test_split

DATA_PATHS = {
    "en": "data/SMSSpamCollection.txt",
    "fa": "data/persian_sms_spam_collection.txt",
}

# ---------------------------------------------------------------------
# English cleaning (unchanged)
# ---------------------------------------------------------------------
URL_RE = re.compile(r"(https?://\S+|www\.\S+)")
EMAIL_RE = re.compile(r"\S+@\S+")
PHONE_RE_EN = re.compile(r"\b\d{5,}\b")  # 5+ digit runs (phone numbers, shortcodes)
MULTI_SPACE_RE = re.compile(r"\s+")
PUNCT_TABLE = str.maketrans("", "", string.punctuation)


def clean_text_en(text: str) -> str:
    """English cleaning pipeline (same as the original project)."""
    text = text.lower()
    text = URL_RE.sub(" urltoken ", text)
    text = EMAIL_RE.sub(" emailtoken ", text)
    text = PHONE_RE_EN.sub(" numtoken ", text)
    text = text.translate(PUNCT_TABLE)
    text = MULTI_SPACE_RE.sub(" ", text).strip()
    return text


# ---------------------------------------------------------------------
# Persian (Farsi) cleaning
# ---------------------------------------------------------------------

# Arabic-script characters that commonly appear in Persian text but
# should be normalized to their standard Persian equivalents.
ARABIC_TO_PERSIAN = {
    "ي": "ی",  # Arabic yeh -> Persian yeh
    "ك": "ک",  # Arabic kaf -> Persian kaf
    "ة": "ه",  # Arabic teh marbuta -> Persian heh
    "ۀ": "ه",
    "أ": "ا",
    "إ": "ا",
    "آ": "ا",  # keep madda-normalized alef simple for a small dataset
    "ؤ": "و",
    "ئ": "ی",
}

# Persian / Arabic-Indic digits -> ASCII digits
DIGIT_MAP = {
    "۰": "0", "۱": "1", "۲": "2", "۳": "3", "۴": "4",
    "۵": "5", "۶": "6", "۷": "7", "۸": "8", "۹": "9",
    "٠": "0", "١": "1", "٢": "2", "٣": "3", "٤": "4",
    "٥": "5", "٦": "6", "٧": "7", "٨": "8", "٩": "9",
}

ZWNJ = "\u200c"

PERSIAN_PUNCTUATION = "،؛؟«»٫٬ـ" + string.punctuation
PERSIAN_PUNCT_TABLE = str.maketrans("", "", PERSIAN_PUNCTUATION)

PHONE_RE_FA = re.compile(r"\b\d{5,}\b")  # applied AFTER digit normalization


def _normalize_persian_chars(text: str) -> str:
    for arabic_char, persian_char in ARABIC_TO_PERSIAN.items():
        text = text.replace(arabic_char, persian_char)
    for src_digit, ascii_digit in DIGIT_MAP.items():
        text = text.replace(src_digit, ascii_digit)
    text = text.replace(ZWNJ, "")  # drop ZWNJ rather than keep it inconsistently
    return text


def clean_text_fa(text: str) -> str:
    """Persian cleaning pipeline: character normalization + the same
    URL/email/number masking and whitespace collapsing idea as English."""
    text = text.lower()  # affects any Latin characters mixed into the text
    text = _normalize_persian_chars(text)
    text = URL_RE.sub(" urltoken ", text)
    text = EMAIL_RE.sub(" emailtoken ", text)
    text = PHONE_RE_FA.sub(" numtoken ", text)
    text = text.translate(PERSIAN_PUNCT_TABLE)
    text = MULTI_SPACE_RE.sub(" ", text).strip()
    return text


CLEANERS = {"en": clean_text_en, "fa": clean_text_fa}


def clean_text(text: str, lang: str = "en") -> str:
    return CLEANERS[lang](text)


# Strips ALL digit runs (any length, any digit script) to build a
# "template key" -- used only to detect near-duplicate messages that
# differ solely in an embedded number (e.g. an OTP code, a shortcode,
# an amount). This matters most for template-generated datasets, where
# the same message shell can appear many times with only the number
# changed. If two such variants end up on opposite sides of a
# train/test split, the model is effectively tested on something it
# has already memorized, which inflates evaluation metrics.
_ANY_DIGIT_RE = re.compile(r"[0-9\u06F0-\u06F9\u0660-\u0669]+")


def _template_key(clean: str) -> str:
    return _ANY_DIGIT_RE.sub("N", clean)


# ---------------------------------------------------------------------
# Shared loading / splitting logic
# ---------------------------------------------------------------------

def load_dataset(lang: str = "en", path: str = None, dedup_templates: bool = True) -> pd.DataFrame:
    """Read the raw tab-separated file into a DataFrame with clean text.

    dedup_templates: if True (default), collapse near-duplicate messages
    that differ only by an embedded number down to one representative
    example each, using _template_key(). This is a no-op for datasets
    without templated messages (e.g. the English SMS Spam Collection)
    but is important for template-generated datasets to get an honest
    train/test split.
    """
    if path is None:
        path = DATA_PATHS[lang]
    df = pd.read_csv(path, sep="\t", header=None, names=["label", "text"])
    df = df.dropna(subset=["text"]).drop_duplicates(subset=["text"]).reset_index(drop=True)
    df["clean_text"] = df["text"].apply(lambda t: clean_text(t, lang))

    if dedup_templates:
        df["_template_key"] = df["clean_text"].apply(_template_key)
        before = len(df)
        df = df.drop_duplicates(subset=["_template_key"]).reset_index(drop=True)
        removed = before - len(df)
        if removed:
            print(f"[{lang}] Removed {removed} near-duplicate templated messages "
                  f"(same message shell, different embedded number) to avoid train/test leakage.")
        df = df.drop(columns=["_template_key"])

    df["label_num"] = (df["label"] == "spam").astype(int)  # ham=0, spam=1
    return df


def get_train_test_split(lang: str = "en", path: str = None, test_size: float = 0.2, random_state: int = 42):
    """Return X_train, X_test, y_train, y_test ready for vectorization."""
    df = load_dataset(lang=lang, path=path)
    X_train, X_test, y_train, y_test = train_test_split(
        df["clean_text"],
        df["label_num"],
        test_size=test_size,
        random_state=random_state,
        stratify=df["label_num"],
    )
    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    import sys

    lang = sys.argv[1] if len(sys.argv) > 1 else "en"
    df = load_dataset(lang=lang)
    print(f"[{lang}] Loaded {len(df)} messages after de-duplication")
    print(df["label"].value_counts())
    print("\nExample before/after cleaning:")
    print("RAW:  ", df["text"].iloc[2])
    print("CLEAN:", df["clean_text"].iloc[2])
