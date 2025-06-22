import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from utils.model_utils import save_model

X_train_vec, X_test_vec, y_train, y_test = joblib.load("models/vectorized_data.pkl")

print("\nTraining Logistic Regression...")
model = LogisticRegression(max_iter=1000,solver='saga', verbose=1, n_jobs=-1)
model.fit(X_train_vec, y_train)

preds = model.predict(X_test_vec)
report = classification_report(y_test, preds)

print("\nClassification Report:\n")
print(report)

save_model(model, "logistic_regression")
print("Model saved: models/logistic_regression.pkl")

with open("results/logistic_regression_report.txt", "w", encoding="utf-8") as f:
    f.write(report)
print("Report saved: results/logistic_regression_report.txt")
