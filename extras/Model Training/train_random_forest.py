import joblib
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from utils.model_utils import save_model

os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)

X_train_vec, X_test_vec, y_train, y_test = joblib.load("models/vectorized_data.pkl")

X_train_vec = X_train_vec.astype('float32')
X_test_vec = X_test_vec.astype('float32')

print("\nTraining Random Forest")
model = RandomForestClassifier(
    n_estimators=5,         
    # max_depth=25,            
    class_weight='balanced', 
    n_jobs=-1,                
    random_state=42,
    verbose=1
)
model.fit(X_train_vec, y_train)

preds = model.predict(X_test_vec)
report = classification_report(y_test, preds)

print("\nClassification Report:\n")
print(report)

save_model(model, "random_forest1")
print("Model saved: models/random_forest1.pkl")

with open("results/random_forest_report1.txt", "w", encoding="utf-8") as f:
    f.write(report)
print("Report saved: results/random_forest_report1.txt")
