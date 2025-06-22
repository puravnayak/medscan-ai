import joblib
import numpy as np
from utils.model_utils import clean_text
from sklearn.preprocessing import LabelEncoder

print("Loading models and vectorizer...")
vectorizer = joblib.load("models/tfidf_vectorizer.pkl")
log_model = joblib.load("models/logistic_regression.pkl")
svm_model = joblib.load("models/svm.pkl")
rf_model = joblib.load("models/random_forest.pkl")
label_encoder = joblib.load("models/label_encoder.pkl")

weights = np.array([0.86, 0.86, 0.83])
weights = weights / weights.sum()

CONFIDENCE_THRESHOLD = 0.30

def predict_top_k(symptom_text, k=3):
    X_user = vectorizer.transform([clean_text(symptom_text)])

    proba_log = log_model.predict_proba(X_user)
    proba_svm = svm_model.predict_proba(X_user)
    proba_rf  = rf_model.predict_proba(X_user)

    avg_proba = (proba_log * weights[0] + proba_svm * weights[1] +proba_rf  * weights[2])

    top_indices = np.argsort(avg_proba[0])[::-1][:k]
    top_diseases = label_encoder.inverse_transform(top_indices)
    top_probs = avg_proba[0][top_indices]

    return list(zip(top_diseases, top_probs))

if __name__ == "__main__":
    print("\nWelcome to MedScan.AI Chatbot!")
    print("Type your symptoms below. Type 'reset' to clear, 'done' to exit.\n")

    accumulated_symptoms = ""

    while True:
        user_input = input("Describe your symptoms:\n> ").strip()

        if user_input.lower() in ["exit", "quit", "done"]:
            print("Goodbye!")
            break
        elif user_input.lower() == "reset":
            accumulated_symptoms = ""
            print("Symptom memory reset. Start over.\n")
            continue

        accumulated_symptoms += " " + user_input
        top_predictions = predict_top_k(accumulated_symptoms)
        
        if top_predictions[0][1] < CONFIDENCE_THRESHOLD:
            print("\n⚠️ Our model is uncertain. We recommend consulting a medical professional.")

        print("\nTop 3 predicted diseases:\n")
        for i, (disease, prob) in enumerate(top_predictions, 1):
            print(f"{i}. {disease} — {prob * 100:.2f}%")

