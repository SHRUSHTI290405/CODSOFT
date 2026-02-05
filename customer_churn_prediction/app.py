import streamlit as st
import pandas as pd
import joblib
import json
import numpy as np

# Load models
model = joblib.load("model/churn_model.pkl")
scaler = joblib.load("model/scaler.pkl")
geo_encoder = joblib.load("model/geo_encoder.pkl")
gender_encoder = joblib.load("model/gender_encoder.pkl")

with open("model/metrics.json") as f:
    metrics = json.load(f)

st.set_page_config(page_title="Customer Churn Prediction", layout="wide")

st.title("📉 Customer Churn Prediction System")
st.markdown("AI system to predict which customers are likely to leave a subscription service.")

# Sidebar insights
st.sidebar.header("📊 Model Performance")
st.sidebar.metric("Accuracy", f"{metrics['accuracy']*100:.2f}%")
st.sidebar.metric("ROC-AUC Score", f"{metrics['roc_auc']:.3f}")

st.sidebar.header("🧠 Model Details")
st.sidebar.write("Algorithm: Gradient Boosting")
st.sidebar.write("Use Case: Customer Retention Analytics")

# -----------------------------
# Manual Prediction Form
# -----------------------------
st.subheader("🔍 Predict Customer Churn")

credit_score = st.number_input("Credit Score", 300, 900, 600)
geography = st.selectbox("Geography", geo_encoder.classes_)
gender = st.selectbox("Gender", gender_encoder.classes_)
age = st.slider("Age", 18, 100, 35)
tenure = st.slider("Tenure (Years with Company)", 0, 10, 3)
balance = st.number_input("Account Balance", 0.0, 300000.0, 60000.0)
products = st.slider("Number of Products", 1, 4, 1)
has_card = st.selectbox("Has Credit Card", [0, 1])
is_active = st.selectbox("Is Active Member", [0, 1])
salary = st.number_input("Estimated Salary", 0.0, 200000.0, 50000.0)

if st.button("Predict Churn"):
    input_data = pd.DataFrame([{
        "CreditScore": credit_score,
        "Geography": geo_encoder.transform([geography])[0],
        "Gender": gender_encoder.transform([gender])[0],
        "Age": age,
        "Tenure": tenure,
        "Balance": balance,
        "NumOfProducts": products,
        "HasCrCard": has_card,
        "IsActiveMember": is_active,
        "EstimatedSalary": salary
    }])

    scaled = scaler.transform(input_data)
    pred = model.predict(scaled)[0]
    prob = model.predict_proba(scaled)[0][1]

    if pred == 1:
        st.error(f"⚠️ Customer Likely to Churn (Risk: {prob*100:.1f}%)")
    else:
        st.success(f"✅ Customer Likely to Stay (Churn Risk: {prob*100:.1f}%)")

    st.progress(int(prob * 100))

# -----------------------------
# Business Insight
# -----------------------------
st.subheader("💼 Business Insight")
st.info("""
This AI model helps businesses identify high-risk customers before they leave.
Companies can use these predictions to offer discounts, loyalty rewards, or
personalized engagement strategies to improve retention and revenue.
""")
