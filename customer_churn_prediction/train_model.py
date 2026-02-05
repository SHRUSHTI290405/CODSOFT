import pandas as pd
import numpy as np
import joblib
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report

print("📂 Loading dataset...")
df = pd.read_csv("data/Churn_Modelling.csv")

# Drop irrelevant columns
df.drop(columns=["RowNumber", "CustomerId", "Surname"], inplace=True)

# Encode categorical variables
le_geo = LabelEncoder()
le_gender = LabelEncoder()

df["Geography"] = le_geo.fit_transform(df["Geography"])
df["Gender"] = le_gender.fit_transform(df["Gender"])

# Split features and target
X = df.drop("Exited", axis=1)
y = df["Exited"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scale numeric features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Model
print("🤖 Training Gradient Boosting model...")
model = GradientBoostingClassifier(n_estimators=150, learning_rate=0.1, random_state=42)
model.fit(X_train_scaled, y_train)

# Evaluation
preds = model.predict(X_test_scaled)
probs = model.predict_proba(X_test_scaled)[:, 1]

accuracy = accuracy_score(y_test, preds)
roc_auc = roc_auc_score(y_test, probs)
report = classification_report(y_test, preds, output_dict=True)

print(f"Accuracy: {accuracy:.4f}")
print(f"ROC-AUC: {roc_auc:.4f}")

# Save everything
joblib.dump(model, "model/churn_model.pkl")
joblib.dump(scaler, "model/scaler.pkl")
joblib.dump(le_geo, "model/geo_encoder.pkl")
joblib.dump(le_gender, "model/gender_encoder.pkl")

with open("model/metrics.json", "w") as f:
    json.dump({"accuracy": accuracy, "roc_auc": roc_auc, "report": report}, f)

print("✅ Churn Model Trained & Saved!")
