
import pandas as pd
import joblib
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer

BASE_DIR = Path(__file__).resolve().parent

# Load preprocessed dataset
df = pd.read_csv(BASE_DIR / "ted_talks_preprocessed.csv")

# Handle any missing text
texts = df["combined_text"].fillna("").astype(str)

# Convert text into numerical vectors
vectorizer = TfidfVectorizer(
    stop_words="english",
    max_features=10000
)

tfidf_matrix = vectorizer.fit_transform(texts)

# Save the fitted vectorizer and feature matrix
joblib.dump(
    vectorizer,
    BASE_DIR / "tfidf_vectorizer.joblib"
)

joblib.dump(
    tfidf_matrix,
    BASE_DIR / "tfidf_matrix.joblib"
)

print("TF-IDF vectorization completed!")
print("Number of talks:", tfidf_matrix.shape[0])
print("Number of features:", tfidf_matrix.shape[1])
print("Matrix type:", type(tfidf_matrix))
print("Vectorizer and matrix saved successfully.")
