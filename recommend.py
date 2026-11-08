
import pandas as pd
import joblib
import re
import math
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

BASE_DIR = Path(__file__).resolve().parent

df = pd.read_csv(BASE_DIR / "ted_talks_preprocessed.csv")
df["title"] = df["title"].fillna("").astype(str)
df["details"] = df["details"].fillna("").astype(str)
df["url"] = df["url"].fillna("").astype(str)
df["main_speaker"] = df["main_speaker"].fillna("Unknown").astype(str)

# Build separate representations for titles and descriptions
title_vectorizer = TfidfVectorizer(
    stop_words="english",
    ngram_range=(1, 2),
    sublinear_tf=True
)

details_vectorizer = TfidfVectorizer(
    stop_words="english",
    ngram_range=(1, 2),
    max_features=20000,
    sublinear_tf=True
)

title_matrix = title_vectorizer.fit_transform(df["title"])
details_matrix = details_vectorizer.fit_transform(df["details"])

# Exclude common words when checking query coverage
stop_words = title_vectorizer.get_stop_words()


def normalize(text):
    return re.sub(r"[^a-z0-9]+", " ", str(text).lower()).strip()


def recommend_talks(topic, top_n=5):
    topic = topic.strip()
    if not topic:
        print("Please enter a topic.")
        return

    query = normalize(topic)
    terms = query.split()
    important_terms = [w for w in terms if w not in stop_words]

    if not important_terms:
        print("Please enter a more specific topic.")
        return

    # Calculate similarity separately for titles and descriptions
    title_query = title_vectorizer.transform([topic])
    details_query = details_vectorizer.transform([topic])

    title_scores = cosine_similarity(
        title_query, title_matrix
    ).flatten()

    details_scores = cosine_similarity(
        details_query, details_matrix
    ).flatten()

    results = []

    for i, row in df.iterrows():
        title = normalize(row["title"])
        details = normalize(row["details"])
        combined = title + " " + details

        # Require enough of the important query words to appear
        matched_terms = sum(
            1 for word in set(important_terms)
            if re.search(r"\b" + re.escape(word) + r"\b", combined)
        )

        required_terms = max(
            1, math.ceil(0.6 * len(set(important_terms)))
        )

        if matched_terms < required_terms:
            continue

        # Titles matter more than descriptions
        score = 0.65 * title_scores[i] + 0.35 * details_scores[i]

        # Bonus when the complete phrase appears
        if re.search(r"\b" + re.escape(query) + r"\b", title):
            score += 0.20
        elif re.search(r"\b" + re.escape(query) + r"\b", details):
            score += 0.10

        if score > 0:
            results.append((i, score))

    results.sort(key=lambda item: item[1], reverse=True)

    # Avoid displaying weak matches
    results = [item for item in results if item[1] >= 0.04]
    results = results[:top_n]

    print(f"\nRecommendations for: {topic}\n")

    if not results:
        print("No sufficiently relevant talks found in this dataset.")
        print("Try a broader or related topic.")
        return

    for rank, (idx, score) in enumerate(results, start=1):
        row = df.iloc[idx]

        print(f"{rank}. {row['title'].title()}")
        print(f"   Speaker: {row['main_speaker']}")
        print(f"   Description: {row['details']}")
        print(f"   Relevance score: {score:.3f}")
        print(f"   URL: {row['url']}\n")


if __name__ == "__main__":
    while True:
        topic = input(
            "Enter a topic (or type 'exit' to quit): "
        )

        if topic.strip().lower() == "exit":
            break

        recommend_talks(topic)
