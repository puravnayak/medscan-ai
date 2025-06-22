import re
import nltk
import pandas as pd
from tqdm import tqdm
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

nltk.download('punkt')
nltk.download('stopwords')

stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = word_tokenize(text)
    filtered = [word for word in tokens if word not in stop_words]
    return ' '.join(filtered)

def clean_dataset(input_csv, output_csv):
    df = pd.read_csv(input_csv)

    tqdm.pandas(desc="Cleaning symptom_text")
    df['clean_text'] = df['symptom_text'].progress_apply(clean_text)

    df[['clean_text', 'disease']].to_csv(output_csv, index=False)
    print(f"✅ Cleaned dataset saved to: {output_csv}")

if __name__ == "__main__":
    clean_dataset("data/synthetic_text_dataset.csv", "data/cleaned_text_dataset.csv")
