
import pandas as pd
import joblib
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

df = pd.read_csv(BASE_DIR / "ted_talks_preprocessed.csv")
vectorizer = joblib.load(BASE_DIR / "tfidf_vectorizer.joblib")
matrix = joblib.load(BASE_DIR / "tfidf_matrix.joblib")

words = vectorizer.get_feature_names_out()

# Select features that occur in the first 10 talks
sample_matrix = matrix[:10]
word_scores = sample_matrix.sum(axis=0).A1
top_indices = word_scores.argsort()[-20:][::-1]

selected_words = words[top_indices]
values = matrix[:10, top_indices].toarray()

result = pd.DataFrame(values, columns=selected_words)
result.insert(0, "Title", df["title"].iloc[:10].values)

result.to_excel(BASE_DIR / "tfidf_values.xlsx", index=False)

print("Matrix shape:", matrix.shape)
print("\nTF-IDF values for selected words:")
print(result.to_string(index=False))
print("\nSaved to tfidf_values.xlsx")
