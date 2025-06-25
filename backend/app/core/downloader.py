import os
import gdown

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) 
MODEL_DIR = os.path.abspath(os.path.join(BASE_DIR, "../models"))

file_ids = {
    "models/logistic_regression.pkl": "1FcoXQBXkOFXzKkwm5Z1tz_vcqdoCzvhe",
    "models/svm.pkl": "16POS1zHpfypNP57JePz-Bafrh-OYGxdd",
    "models/tfidf_vectorizer.pkl": "1P1gmI5BwD75ke2AqX9c12r7gLJf1auIT",
    "models/label_encoder.pkl": "1BdAouLr_TyLG7Ye3xd4ftJ0CEvnqxyXr"
}

def download_missing_models():
    os.makedirs(MODEL_DIR, exist_ok=True)

    for rel_path, file_id in file_ids.items():
        full_path = os.path.join(MODEL_DIR, rel_path)
        if not os.path.exists(full_path):
            print(f"Downloading {rel_path}...")
            gdown.download(id=file_id, output=full_path, quiet=False)
