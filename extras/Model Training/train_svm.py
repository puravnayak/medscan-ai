import joblib
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import classification_report
from utils.model_utils import save_model

X_train_vec, X_test_vec, y_train, y_test = joblib.load("models/vectorized_data.pkl")

print("\nTraining base LinearSVC...")
base_svm = LinearSVC(max_iter=1000, verbose=1)
base_svm.fit(X_train_vec, y_train)

print("\nCalibrating probabilities using sigmoid (cv='prefit')...")
model = CalibratedClassifierCV(base_svm, method='sigmoid', cv='prefit')
model.fit(X_train_vec, y_train)  

preds = model.predict(X_test_vec)
report = classification_report(y_test, preds)

print("\nClassification Report:\n")
print(report)

save_model(model, "svm")
print("Model saved: models/svm.pkl")

with open("results/svm_report.txt", "w", encoding="utf-8") as f:
    f.write(report)
print("Report saved: results/svm_report.txt")
