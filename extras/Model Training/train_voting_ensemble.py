import joblib
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.metrics import accuracy_score, classification_report

_, X_test_vec, _, y_test = joblib.load("models/vectorized_data.pkl")
label_encoder = joblib.load("models/label_encoder.pkl")

log_model = joblib.load("models/logistic_regression.pkl")
svm_model = joblib.load("models/svm.pkl")
rf_model = joblib.load("models/random_forest.pkl")

log_preds = log_model.predict(X_test_vec)
svm_preds = svm_model.predict(X_test_vec)
rf_preds = rf_model.predict(X_test_vec)

def encode_preds(preds):
    return label_encoder.transform([str(x).strip().lower() for x in preds])

log_preds_enc = encode_preds(log_preds)
svm_preds_enc = encode_preds(svm_preds)
rf_preds_enc = encode_preds(rf_preds)
y_test_enc = encode_preds(y_test)

acc_log = accuracy_score(y_test_enc, log_preds_enc)
acc_svm = accuracy_score(y_test_enc, svm_preds_enc)
acc_rf = accuracy_score(y_test_enc, rf_preds_enc)
weights = np.array([acc_log, acc_svm, acc_rf])

all_preds = np.vstack([log_preds_enc, svm_preds_enc, rf_preds_enc])

ensemble_preds = []
confidences = []
agreements = []

def weighted_vote_with_meta(preds, weights):
    count = Counter()
    for pred, weight in zip(preds, weights):
        count[pred] += weight

    max_weight = max(count.values())
    tied_classes = [cls for cls, wt in count.items() if wt == max_weight]

    agreement = sum([1 for p in preds if p == tied_classes[0]])  
    confidence = max_weight / sum(weights)

    if len(tied_classes) == 1:
        return tied_classes[0], confidence, agreement

    model_preds = [preds[0], preds[1], preds[2]]
    model_priority = [1, 0, 2]
    for idx in model_priority:
        if model_preds[idx] in tied_classes:
            return model_preds[idx], confidence, agreement

    return tied_classes[0], confidence, agreement

for i in range(all_preds.shape[1]):
    voted_label, conf, agree = weighted_vote_with_meta(all_preds[:, i], weights)
    ensemble_preds.append(voted_label)
    confidences.append(conf)
    agreements.append(agree)

weighted_preds = label_encoder.inverse_transform(ensemble_preds)
y_test_decoded = label_encoder.inverse_transform(y_test_enc)

df_result = pd.DataFrame({
    "True_Label": y_test_decoded,
    "Logistic_Pred": label_encoder.inverse_transform(log_preds_enc),
    "SVM_Pred": label_encoder.inverse_transform(svm_preds_enc),
    "RandomForest_Pred": label_encoder.inverse_transform(rf_preds_enc),
    "Ensemble_Pred": weighted_preds,
    "Confidence": confidences,
    "Agreement": agreements
})
df_result.to_csv("results/ensemble_predictions.csv", index=False)

ensemble_accuracy = accuracy_score(y_test_decoded, weighted_preds)
ensemble_report = classification_report(y_test_decoded, weighted_preds)

with open("results/voting_ensemble_report.txt", "w", encoding="utf-8") as f:
    f.write(f"Weighted Ensemble Accuracy: {ensemble_accuracy:.4f}\n\n")
    f.write(ensemble_report)

print(f"Ensemble Accuracy: {ensemble_accuracy:.4f}")
