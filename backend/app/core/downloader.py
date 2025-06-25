import os
import gdown

file_ids = {
    "models/logistic_regression.pkl": "1FcoXQBXkOFXzKkwm5Z1tz_vcqdoCzvhe",
    "models/svm.pkl": "16POS1zHpfypNP57JePz-Bafrh-OYGxdd",
    "models/tfidf_vectorizer.pkl": "1P1gmI5BwD75ke2AqX9c12r7gLJf1auIT",
    "models/label_encoder.pkl": "1BdAouLr_TyLG7Ye3xd4ftJ0CEvnqxyXr"
}

def download_missing_models():
    os.makedirs("models", exist_ok=True)
    for path, file_id in file_ids.items():
        if not os.path.exists(path):
            print(f"Downloading {path}...")
            gdown.download(id=file_id, output=path, quiet=False)
