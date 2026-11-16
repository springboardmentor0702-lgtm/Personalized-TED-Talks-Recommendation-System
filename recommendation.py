import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def load_data():
    return pd.read_csv("data/final_dataset.csv")


def build_model(df):
    df = df.copy()

    df["combined_text"] = (
        (df["title"].fillna("") + " ") * 5
        + (df["topics"].fillna("") + " ") * 5
        + (df["description"].fillna("") + " ") * 3
        + (df["transcript"].fillna("") + " ")
    )

    vectorizer = TfidfVectorizer(
        max_features=30000,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        sublinear_tf=True,
        stop_words="english",
        lowercase=True
    )

    tfidf_matrix = vectorizer.fit_transform(df["combined_text"])

    similarity_matrix = cosine_similarity(tfidf_matrix)

    return similarity_matrix


def recommend_talks(df, similarity_matrix, talk_title, top_n=5):
    matches = df.index[df["title"] == talk_title].tolist()

    if not matches:
        return pd.DataFrame()

    index = matches[0]

    scores = list(enumerate(similarity_matrix[index]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)

    recommendations = []

    for idx, score in scores:
        if idx == index:
            continue

        recommendations.append({
            "talk_id": df.iloc[idx]["talk_id"],
            "title": df.iloc[idx]["title"],
            "speaker_1": df.iloc[idx]["speaker_1"],
            "year": df.iloc[idx]["year"],
            "url": df.iloc[idx]["url"],
            "similarity_score": score
        })

        if len(recommendations) == top_n:
            break

    return pd.DataFrame(recommendations)


def get_popular_talks(df, n=10):
    return df.sort_values("views", ascending=False).head(n)