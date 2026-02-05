import streamlit as st
import pandas as pd
import joblib
import json
import re

# Load trained model
model = joblib.load("model/spam_model.pkl")
with open("model/metrics.json") as f:
    metrics = json.load(f)

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text

st.set_page_config(page_title="Spam SMS Detection System", layout="wide")

st.title("📩 AI Spam SMS Detection & Analytics")
st.markdown("Analyze SMS datasets and detect spam messages using Machine Learning.")

# -----------------------------
# Sidebar Model Info
# -----------------------------
st.sidebar.header("🤖 Model Performance")
st.sidebar.metric("Accuracy", f"{metrics['accuracy']*100:.2f}%")
st.sidebar.metric("ROC-AUC", f"{metrics['roc_auc']:.3f}")

st.sidebar.header("🧠 Model Details")
st.sidebar.write("Algorithm: Multinomial Naive Bayes")
st.sidebar.write("Vectorization: TF-IDF")
st.sidebar.write("Task: Binary Text Classification")

# -----------------------------
# DATASET ANALYTICS MODE
# -----------------------------
st.header("📊 Spam Dataset Insights")

uploaded_file = st.file_uploader("Upload SMS Dataset (CSV)", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file, encoding="latin-1")[['v1','v2']]
    df.columns = ["label", "message"]
    df["label_num"] = df["label"].map({"ham": 0, "spam": 1})

    st.subheader("📄 Dataset Preview")
    st.dataframe(df.head())

    total_msgs = len(df)
    spam_count = df["label_num"].sum()
    ham_count = total_msgs - spam_count

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Messages", total_msgs)
    col2.metric("Spam Messages", spam_count)
    col3.metric("Legitimate Messages", ham_count)

    st.subheader("📊 Spam vs Legitimate Distribution")
    st.bar_chart(df["label"].value_counts())

    # Word count insight
    df["word_count"] = df["message"].apply(lambda x: len(str(x).split()))
    st.subheader("📝 Average Message Length")
    st.write("Spam Avg Words:", int(df[df["label"]=="spam"]["word_count"].mean()))
    st.write("Ham Avg Words:", int(df[df["label"]=="ham"]["word_count"].mean()))

    # Run model on dataset
    st.subheader("🤖 Model Predictions on Uploaded Data")
    df["cleaned"] = df["message"].apply(clean_text)
    preds = model.predict(df["cleaned"])
    df["Predicted"] = preds

    accuracy_dataset = (df["Predicted"] == df["label_num"]).mean()
    st.metric("Model Accuracy on Uploaded Dataset", f"{accuracy_dataset*100:.2f}%")

    st.dataframe(df[["message","label","Predicted"]].head())

# -----------------------------
# LIVE SMS DETECTOR
# -----------------------------
st.header("📨 Live SMS Spam Detector")

user_msg = st.text_area("Enter SMS Message")

if st.button("Detect Spam"):
    cleaned = clean_text(user_msg)
    pred = model.predict([cleaned])[0]
    prob = model.predict_proba([cleaned])[0][1]

    if pred == 1:
        st.error(f"🚫 Spam Detected (Confidence: {prob*100:.1f}%)")
    else:
        st.success(f"✅ Legitimate Message (Spam Risk: {prob*100:.1f}%)")

    st.progress(int(prob * 100))

# -----------------------------
# BUSINESS INSIGHT
# -----------------------------
st.subheader("💼 Business Insight")
st.info("""
This AI-powered spam detection system helps messaging platforms and telecom providers
filter unwanted promotional and phishing messages. Dataset analytics reveal spam trends,
while live detection protects users in real time.
""")
