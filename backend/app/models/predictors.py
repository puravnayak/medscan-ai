import joblib
import numpy as np
from app.utils.cleaner import clean_text
from app.core.config import CONFIDENCE_THRESHOLD

vectorizer = joblib.load("models/tfidf_vectorizer.pkl")
log_model = joblib.load("models/logistic_regression.pkl")
svm_model = joblib.load("models/svm.pkl")
# rf_model = joblib.load("models/random_forest.pkl")
label_encoder = joblib.load("models/label_encoder.pkl")

weights = np.array([0.86, 0.86]) #0.83->rf
weights = weights / weights.sum()


def get_top_k_predictions_with_confidence(text: str, k=3):
    cleaned = clean_text(text)
    X = vectorizer.transform([cleaned])

    proba_log = log_model.predict_proba(X)
    proba_svm = svm_model.predict_proba(X)
    # proba_rf = rf_model.predict_proba(X)

    avg_proba = proba_log * weights[0] + proba_svm * weights[1] #+ proba_rf * weights[2]

    top_indices = np.argsort(avg_proba[0])[::-1][:k]
    top_diseases = label_encoder.inverse_transform(top_indices)
    top_probs = avg_proba[0][top_indices]

    result = [
        {"disease": disease, "probability": round(float(prob), 4)}
        for disease, prob in zip(top_diseases, top_probs)
    ]

    disclaimer = None
    if result[0]["probability"] < CONFIDENCE_THRESHOLD:
        disclaimer = "⚠️ Model confidence is low. Please consult a healthcare professional."

    return result, disclaimer
