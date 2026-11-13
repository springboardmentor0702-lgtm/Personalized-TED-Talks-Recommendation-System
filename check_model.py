
import pandas as pd
import joblib
from sklearn.metrics.pairwise import cosine_similarity

print("Loading TED Talks dataset...")

df = pd.read_csv("data/ted_model_dataset.csv")

print("Loading TF-IDF matrix...")
tfidf_matrix = joblib.load("models/ted_tfidf_matrix.joblib")

print("Loading TF-IDF vectorizer...")
tfidf_vectorizer = joblib.load("models/ted_tfidf_vectorizer.joblib")

print("\n========== MODEL VALIDATION ==========")

print("Dataset shape:", df.shape)
print("TF-IDF matrix shape:", tfidf_matrix.shape)
print("Vocabulary size:", len(tfidf_vectorizer.vocabulary_))

print("\nChecking row compatibility...")

if len(df) == tfidf_matrix.shape[0]:
    print("SUCCESS: Dataset and TF-IDF matrix rows match.")
else:
    print("ERROR: Dataset and TF-IDF matrix rows do not match.")

print("\nTesting cosine similarity...")

similarity = cosine_similarity(
    tfidf_matrix[0],
    tfidf_matrix[1]
)[0][0]

print("Sample cosine similarity:", round(float(similarity), 4))

print("\nFirst TED Talk:", df.iloc[0]["title"])
print("Second TED Talk:", df.iloc[1]["title"])

print("\nMODEL VALIDATION COMPLETED!")
