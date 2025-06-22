import pandas as pd
from sklearn.model_selection import train_test_split
from utils.model_utils import get_vectorizer, save_model

df = pd.read_csv("data/cleaned_text_dataset.csv")
disease_counts = df["disease"].value_counts()
df = df[df["disease"].isin(disease_counts[disease_counts > 1].index)]

X = df["clean_text"]
y = df["disease"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, test_size=0.2, random_state=42
)

vectorizer = get_vectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

save_model(vectorizer, "tfidf_vectorizer")

import joblib
joblib.dump((X_train_vec, X_test_vec, y_train, y_test), "models/vectorized_data.pkl")