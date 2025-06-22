import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
import re

def get_vectorizer():
    return TfidfVectorizer(max_features=3000, ngram_range=(1, 2))

def save_model(model, name):
    joblib.dump(model, f"models/{name}.pkl")

def load_model(name):
    return joblib.load(f"models/{name}.pkl")

def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text