
import pandas as pd
import joblib
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity

# Project folder
BASE_DIR = Path(__file__).resolve().parent

# Load the dataset and saved TF-IDF objects
df = pd.read_csv(BASE_DIR / "ted_talks_preprocessed.csv")
vectorizer = joblib.load(BASE_DIR / "tfidf_vectorizer.joblib")
tfidf_matrix = joblib.load(BASE_DIR / "tfidf_matrix.joblib")


def recommend_talks(title, top_n=5):
    # Find the selected talk
    matches = df[
        df["title"].str.casefold() == title.casefold()
    ]

    if matches.empty:
        print("Talk not found. Please enter an exact title.")
        return pd.DataFrame()

    # Get its row position in the matrix
    talk_index = matches.index[0]

    # Compare the selected talk against all talks
    scores = cosine_similarity(
        tfidf_matrix[talk_index],
        tfidf_matrix
    ).flatten()

    # Sort talks by similarity, highest first
    ranked_indices = scores.argsort()[::-1]

    # Exclude the selected talk itself
    ranked_indices = [
        i for i in ranked_indices if i != talk_index
    ]

    # Return the top recommendations
    result = df.iloc[ranked_indices[:top_n]][
        ["title", "main_speaker", "details", "url"]
    ].copy()

    result["similarity_score"] = [
        round(float(scores[i]), 3)
        for i in ranked_indices[:top_n]
    ]

    return result


# Test the recommendation engine
if __name__ == "__main__":
    print("Available TED Talks:")
    print(df["title"].head(10).to_string(index=False))

    selected_title = input(
        "\nEnter an exact TED Talk title: "
    )

    recommendations = recommend_talks(selected_title)

    print("\nRecommended TED Talks:")
    print(recommendations.to_string(index=False))
