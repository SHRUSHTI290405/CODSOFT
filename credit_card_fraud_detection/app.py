import streamlit as st
import pandas as pd
import joblib
import json
import numpy as np

# -----------------------------
# Load saved assets
# -----------------------------
model = joblib.load("model/fraud_model.pkl")
scaler = joblib.load("model/scaler.pkl")
encoders = joblib.load("model/encoders.pkl")

with open("model/metrics.json") as f:
    metrics = json.load(f)

st.set_page_config(page_title="Fraud Detection AI", layout="wide")

st.title("💳 Credit Card Fraud Detection System")
st.markdown("AI-powered system for identifying fraudulent transactions in financial systems.")

# -----------------------------
# Sidebar — Model Insights
# -----------------------------
st.sidebar.header("📊 Model Performance")
st.sidebar.metric("Accuracy", f"{metrics['accuracy']*100:.2f}%")
st.sidebar.metric("ROC-AUC Score", f"{metrics['roc_auc']:.3f}")

st.sidebar.header("🧠 Model Details")
st.sidebar.write("Algorithm: Random Forest")
st.sidebar.write("Training Strategy: Balanced Class Weights")
st.sidebar.write("Feature Scaling: StandardScaler")

# -----------------------------
# Upload CSV
# -----------------------------
uploaded_file = st.file_uploader("📂 Upload Transaction Data (CSV)", type=["csv"])

if uploaded_file:
    data = pd.read_csv(uploaded_file)
    st.subheader("📄 Uploaded Transactions")
    st.dataframe(data.head())

    # -----------------------------
    # 🔁 Preprocessing (MATCH TRAINING)
    # -----------------------------

    # Convert datetime
    data["trans_date_trans_time"] = pd.to_datetime(data["trans_date_trans_time"])
    data["trans_hour"] = data["trans_date_trans_time"].dt.hour
    data["trans_day"] = data["trans_date_trans_time"].dt.day
    data["trans_month"] = data["trans_date_trans_time"].dt.month
    data.drop(columns=["trans_date_trans_time"], inplace=True)

    # Drop unused columns
    drop_cols = ["trans_num", "first", "last", "street", "dob"]
    data.drop(columns=[c for c in drop_cols if c in data.columns], inplace=True)

    # ❗ Remove label column if present
    if "is_fraud" in data.columns:
        data.drop(columns=["is_fraud"], inplace=True)

    # Encode categoricals safely
    for col, le in encoders.items():
        if col in data.columns:
            data[col] = data[col].astype(str).map(lambda s: s if s in le.classes_ else "UNKNOWN")
            data[col] = le.transform(data[col])

    # Ensure column order matches training
    data = data[scaler.feature_names_in_]

    # Scale
    scaled = scaler.transform(data)

    # -----------------------------
    # 🤖 Predictions
    # -----------------------------
    preds = model.predict(scaled)
    probs = model.predict_proba(scaled)[:, 1]

    data["Fraud Prediction"] = preds
    data["Fraud Probability"] = probs

    # -----------------------------
    # 📊 Results Section
    # -----------------------------
    st.subheader("🚨 Fraud Detection Results")
    st.dataframe(data.head())

    fraud_count = int(data["Fraud Prediction"].sum())
    legit_count = len(data) - fraud_count

    col1, col2 = st.columns(2)
    col1.metric("⚠️ Fraudulent Transactions", fraud_count)
    col2.metric("✅ Legitimate Transactions", legit_count)

    # Fraud vs Legit Chart
    st.subheader("📊 Fraud vs Legitimate Transactions")
    st.bar_chart(data["Fraud Prediction"].value_counts())

    # Fraud by Hour
    st.subheader("⏰ Fraud Activity by Hour")
    fraud_by_hour = data[data["Fraud Prediction"] == 1]["trans_hour"].value_counts().sort_index()
    st.line_chart(fraud_by_hour)

    # High Risk Transactions
    st.subheader("🔥 High-Risk Transactions (Probability > 0.8)")
    high_risk = data[data["Fraud Probability"] > 0.8]
    if len(high_risk) > 0:
        st.dataframe(high_risk)
    else:
        st.success("No extremely high-risk transactions detected.")

    # Feature Importance
    st.subheader("📈 Top Features Influencing Fraud Detection")
    feature_importances = model.feature_importances_
    feature_names = scaler.feature_names_in_

    fi_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": feature_importances
    }).sort_values(by="Importance", ascending=False).head(10)

    st.bar_chart(fi_df.set_index("Feature"))

# -----------------------------
# 💼 Business Insight
# -----------------------------
st.subheader("💼 Business Impact")
st.info("""
This AI-driven fraud detection system helps financial institutions detect suspicious transactions
in real time. By reducing false negatives (missed fraud) and controlling false positives,
the system protects customers while minimizing financial losses.
""")
