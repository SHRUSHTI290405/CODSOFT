import pandas as pd
import numpy as np
import joblib
import re
import nltk
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score

nltk.download('stopwords')
from nltk.corpus import stopwords

columns = ["ID", "TITLE", "GENRE", "DESCRIPTION"]

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text

# Load training data
train_df = pd.read_csv("data/train_data.txt", sep=":::", names=columns, engine='python')
train_df["DESCRIPTION"] = train_df["DESCRIPTION"].apply(clean_text)

X_train = train_df["DESCRIPTION"]
y_train = train_df["GENRE"]

pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(stop_words=stopwords.words('english'), max_features=5000)),
    ("clf", LogisticRegression(max_iter=200))
])

pipeline.fit(X_train, y_train)

# Save model
joblib.dump(pipeline, "model/genre_model.pkl")

# Evaluation
test_df = pd.read_csv("data/test_data_solution.txt", sep=":::", names=columns, engine='python')
test_df["DESCRIPTION"] = test_df["DESCRIPTION"].apply(clean_text)

X_test = test_df["DESCRIPTION"]
y_test = test_df["GENRE"]

predictions = pipeline.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
report = classification_report(y_test, predictions, output_dict=True)

# Save metrics
with open("model/metrics.json", "w") as f:
    json.dump({"accuracy": accuracy, "report": report}, f)

print("✅ Model trained and metrics saved!")
