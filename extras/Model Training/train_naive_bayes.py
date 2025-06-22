import joblib
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
from utils.model_utils import save_model

X_train_vec, X_test_vec, y_train, y_test = joblib.load("models/vectorized_data.pkl")

print("\nTraining Naive Bayes...")
model = MultinomialNB()
model.fit(X_train_vec, y_train)

preds = model.predict(X_test_vec)
report = classification_report(y_test, preds)

print("\nClassification Report:\n")
print(report)

save_model(model, "naive_bayes")
print("Model saved: models/naive_bayes.pkl")

with open("results/naive_bayes_report.txt", "w", encoding="utf-8") as f:
    f.write(report)
print("Report saved: results/naive_bayes_report.txt")
