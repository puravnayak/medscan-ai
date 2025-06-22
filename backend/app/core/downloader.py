import os
import gdown

file_links = {
    "models/logistic_regression.pkl": "https://drive.google.com/uc?id=1FcoXQBXkOFXzKkwm5Z1tz_vcqdoCzvhe",
    "models/svm.pkl": "https://drive.google.com/uc?id=16POS1zHpfypNP57JePz-Bafrh-OYGxdd",
    "models/tfidf_vectorizer.pkl": "https://drive.google.com/uc?id=1P1gmI5BwD75ke2AqX9c12r7gLJf1auIT",
    "models/label_encoder.pkl": "https://drive.google.com/uc?id=1BdAouLr_TyLG7Ye3xd4ftJ0CEvnqxyXr"
}

def download_missing_models():
    os.makedirs("models", exist_ok=True)
    for path, url in file_links.items():
        if not os.path.exists(path):
            print(f"Downloading {path}...")
            gdown.download(url, path, quiet=False)