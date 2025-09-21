# train_model.py
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib
import os
import re

DATA_PATH = "reviews.csv"

#DATA_PATH = r"C:\Users\LENOVO\OneDrive\Desktop\project\review-analyzer\reviews.csv"
MODEL_DIR = "model"
MODEL_FILE = os.path.join(MODEL_DIR, "sentiment_model.joblib")
VECT_FILE = os.path.join(MODEL_DIR, "vectorizer.joblib")

def simple_clean(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def main():
    df = pd.read_csv(DATA_PATH)
    df['review'] = df['review'].astype(str).apply(simple_clean)
    X = df['review']
    y = df['label']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    vect = TfidfVectorizer(ngram_range=(1,2), max_features=5000)
    clf = MultinomialNB()

    pipeline = make_pipeline(vect, clf)
    pipeline.fit(X_train, y_train)

    preds = pipeline.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, preds))
    print(classification_report(y_test, preds))

    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(pipeline, MODEL_FILE)
    print(f"Saved trained pipeline to {MODEL_FILE}")

if __name__ == "__main__":
    main()
