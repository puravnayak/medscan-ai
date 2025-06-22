import pandas as pd
import random

def generate_symptom_text(row, symptom_list):
    active_symptoms = [symptom for symptom, val in zip(symptom_list, row) if val == 1]
    if not active_symptoms:
        return "I feel fine."
    
    random.shuffle(active_symptoms)  
    sentence = "I have " + ", ".join(active_symptoms[:-1])
    if len(active_symptoms) > 1:
        sentence += " and " + active_symptoms[-1]
    else:
        sentence = "I have " + active_symptoms[0]
    return sentence + "."

def generate_dataset(input_csv, output_csv):
    df = pd.read_csv(input_csv)
    symptom_list = df.columns.tolist()[1:]  
    total = len(df)

    generated_data = []
    for idx, (_, row) in enumerate(df.iterrows()):
        disease = row['diseases']
        symptoms = row[symptom_list].tolist()
        text = generate_symptom_text(symptoms, symptom_list)
        generated_data.append([text, disease])

        
        if idx % 100 == 0 or idx == total - 1:
            progress = (idx + 1) / total * 100
            print(f"Progress: {progress:.2f}% ({idx + 1}/{total})")

    out_df = pd.DataFrame(generated_data, columns=["symptom_text", "disease"])
    out_df.to_csv(output_csv, index=False)
    print(f"✅ Generated dataset saved to: {output_csv}")

if __name__ == "__main__":
    generate_dataset("data/disease_symptom.csv", "data/synthetic_text_dataset.csv")
