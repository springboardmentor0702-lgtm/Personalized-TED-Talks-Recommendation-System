
import pandas as pd
import joblib
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity

BASE_DIR = Path(__file__).resolve().parent

# Load dataset and saved TF-IDF objects
df = pd.read_csv(BASE_DIR / "ted_talks_preprocessed.csv")
vectorizer = joblib.load(BASE_DIR / "tfidf_vectorizer.joblib")
tfidf_matrix = joblib.load(BASE_DIR / "tfidf_matrix.joblib")


def recommend_talks(topic, top_n=5):
    topic = topic.strip().lower()

    if not topic:
        print("Please enter a topic.")
        return

    # Convert the entered topic into a TF-IDF vector
    query_vector = vectorizer.transform([topic])

    # Check whether the topic contains known words
    if query_vector.nnz == 0:
        print("\nNo matching words found in the TF-IDF vocabulary.")
        print("Try a related topic or a different phrase.")
        return

    # Compare the topic with all TED Talks
    scores = cosine_similarity(query_vector, tfidf_matrix).flatten()

    # Rank talks by similarity
    ranked_indices = scores.argsort()[::-1]

    print(f"\nRecommended TED Talks about: {topic}\n")

    count = 0

    for idx in ranked_indices:
        if scores[idx] <= 0:
            break

        talk = df.iloc[idx]

        print(f"{count + 1}. {talk['title'].title()}")
        print(f"   Speaker: {talk['main_speaker']}")
        print(f"   Description: {talk['details']}")
        print(f"   Similarity score: {scores[idx]:.4f}")
        print(f"   URL: {talk['url']}\n")

        count += 1

        if count == top_n:
            break

    if count == 0:
        print("No related talks found in the dataset.")


if __name__ == "__main__":
    print("TED Talks Topic Recommendation System")
    print("-------------------------------------")

    while True:
        topic = input("\nEnter a topic (or type 'exit' to quit): ")

        if topic.strip().lower() == "exit":
            print("Goodbye!")
            break

        recommend_talks(topic)
