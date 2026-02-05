import streamlit as st
import joblib
import json
import re
import pandas as pd

# Load assets
model = joblib.load("model/genre_model.pkl")
with open("model/metrics.json") as f:
    metrics = json.load(f)

columns = ["ID", "TITLE", "GENRE", "DESCRIPTION"]
train_df = pd.read_csv("data/train_data.txt", sep=":::", names=columns, engine='python')

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text

# Page config
st.set_page_config(page_title="AI Movie Genre Classifier", layout="wide")

# Header
st.title("🎬 AI Movie Genre Classification System")
st.markdown("An NLP-powered system that predicts movie genres from storyline descriptions.")

# Sidebar — Dataset Insights
st.sidebar.header("📂 Dataset Insights")
st.sidebar.write(f"Total Training Samples: {len(train_df)}")
st.sidebar.write(f"Unique Genres: {train_df['GENRE'].nunique()}")

st.sidebar.subheader("🎭 Top Genres")
st.sidebar.bar_chart(train_df["GENRE"].value_counts().head(5))

# Sidebar — Model Info
st.sidebar.header("🤖 Model Details")
st.sidebar.write("Algorithm: Logistic Regression")
st.sidebar.write("Vectorizer: TF-IDF (5000 features)")
st.sidebar.write("Text Cleaning: Regex + Stopword Removal")

# Sidebar — Accuracy
st.sidebar.header("📈 Model Accuracy")
st.sidebar.metric("Accuracy", f"{metrics['accuracy']*100:.2f}%")

# Main input
user_input = st.text_area("📝 Enter Movie Plot Summary")

if st.button("Predict Genre"):
    cleaned = clean_text(user_input)
    prediction = model.predict([cleaned])[0]
    probs = model.predict_proba([cleaned])[0]

    st.success(f"🎯 Predicted Genre: **{prediction}**")

    # Confidence chart
    st.subheader("🔍 Prediction Confidence")
    prob_df = pd.DataFrame({
        "Genre": model.classes_,
        "Confidence": probs
    }).sort_values(by="Confidence", ascending=False)

    st.bar_chart(prob_df.set_index("Genre"))

    # Word count insight
    word_count = len(user_input.split())
    st.write(f"📝 Word Count: {word_count}")
    if word_count < 20:
        st.warning("Short descriptions may reduce prediction accuracy.")

    # Influential words
    st.subheader("🔑 Influential Keywords")
    vectorizer = model.named_steps['tfidf']
    feature_names = vectorizer.get_feature_names_out()
    coefs = model.named_steps['clf'].coef_
    class_index = list(model.classes_).index(prediction)

    top_features = sorted(zip(coefs[class_index], feature_names), reverse=True)[:10]
    keywords = [word for _, word in top_features]

    st.write(", ".join(keywords))

# Business Insight
st.subheader("💼 Business Insight")
st.info("""
This AI solution can help streaming platforms and production companies automatically tag
movies by genre, improving recommendation engines, search accuracy, and content discovery.
""")
