import pandas as pd
import re
import nltk
import joblib
import json
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score

nltk.download('stopwords')

print("📂 Loading dataset...")
df = pd.read_csv("data/spam.csv", encoding='latin-1')[['v1', 'v2']]
df.columns = ["label", "message"]

# Convert labels to binary
df["label"] = df["label"].map({"ham": 0, "spam": 1})

# Text cleaning function
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text

df["message"] = df["message"].apply(clean_text)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    df["message"], df["label"], test_size=0.2, random_state=42, stratify=df["label"]
)

# Pipeline
pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(stop_words=stopwords.words('english'), max_features=4000)),
    ("clf", MultinomialNB())
])

print("🤖 Training Spam Detection Model...")
pipeline.fit(X_train, y_train)

# Evaluation
preds = pipeline.predict(X_test)
probs = pipeline.predict_proba(X_test)[:, 1]

accuracy = accuracy_score(y_test, preds)
roc_auc = roc_auc_score(y_test, probs)
report = classification_report(y_test, preds, output_dict=True)

print(f"Accuracy: {accuracy:.4f}")
print(f"ROC-AUC: {roc_auc:.4f}")

# Save model
joblib.dump(pipeline, "model/spam_model.pkl")

with open("model/metrics.json", "w") as f:
    json.dump({"accuracy": accuracy, "roc_auc": roc_auc, "report": report}, f)

print("✅ Spam Model Trained & Saved!")