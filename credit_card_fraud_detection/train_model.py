import pandas as pd
import numpy as np
import joblib
import json
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score

print("📂 Loading data...")
train_df = pd.read_csv("data/fraudTrain.csv")
test_df = pd.read_csv("data/fraudTest.csv")

# -----------------------------
# 🚀 SPEED BOOST: Sample Data
# -----------------------------
train_df = train_df.sample(100000, random_state=42)   # use 100k rows for fast training

# -----------------------------
# 🕒 Convert Datetime
# -----------------------------
for df in [train_df, test_df]:
    df["trans_date_trans_time"] = pd.to_datetime(df["trans_date_trans_time"])
    df["trans_hour"] = df["trans_date_trans_time"].dt.hour
    df["trans_day"] = df["trans_date_trans_time"].dt.day
    df["trans_month"] = df["trans_date_trans_time"].dt.month
    df.drop(columns=["trans_date_trans_time"], inplace=True)

# -----------------------------
# 🗑 Drop High Cardinal Columns
# -----------------------------
drop_cols = ["trans_num", "first", "last", "street", "dob"]
train_df.drop(columns=[c for c in drop_cols if c in train_df.columns], inplace=True)
test_df.drop(columns=[c for c in drop_cols if c in test_df.columns], inplace=True)

# -----------------------------
# 🔤 Encode Categoricals Safely
# -----------------------------
cat_cols = train_df.select_dtypes(include="object").columns
encoders = {}

for col in cat_cols:
    le = LabelEncoder()
    train_df[col] = le.fit_transform(train_df[col].astype(str))
    
    test_df[col] = test_df[col].astype(str).map(lambda s: s if s in le.classes_ else "UNKNOWN")
    le.classes_ = np.append(le.classes_, "UNKNOWN")
    test_df[col] = le.transform(test_df[col])
    
    encoders[col] = le

# -----------------------------
# 🎯 Split
# -----------------------------
X_train = train_df.drop("is_fraud", axis=1)
y_train = train_df["is_fraud"]

X_test = test_df.drop("is_fraud", axis=1)
y_test = test_df["is_fraud"]

# -----------------------------
# ⚖ Scale
# -----------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# -----------------------------
# 🌲 Faster Random Forest
# -----------------------------
print("🤖 Training Random Forest...")
model = RandomForestClassifier(
    n_estimators=80,          # reduced trees
    max_depth=18,             # limit tree depth
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

model.fit(X_train_scaled, y_train)

# -----------------------------
# 📊 Evaluation
# -----------------------------
print("📊 Evaluating...")
preds = model.predict(X_test_scaled)
probs = model.predict_proba(X_test_scaled)[:, 1]

accuracy = accuracy_score(y_test, preds)
roc_auc = roc_auc_score(y_test, probs)
report = classification_report(y_test, preds, output_dict=True)

# -----------------------------
# 💾 Save
# -----------------------------
joblib.dump(model, "model/fraud_model.pkl")
joblib.dump(scaler, "model/scaler.pkl")
joblib.dump(encoders, "model/encoders.pkl")

with open("model/metrics.json", "w") as f:
    json.dump({"accuracy": accuracy, "roc_auc": roc_auc, "report": report}, f)

print("\n✅ Training Complete!")
print(f"Accuracy: {accuracy:.4f}")
print(f"ROC-AUC: {roc_auc:.4f}")
