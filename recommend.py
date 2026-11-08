"""
Module 4: Interactive TED Talks Recommender CLI
Personalized TED Talks Recommendation System
"""

import sys
import joblib
import pandas as pd
from pathlib import Path
from recommender_engine import ContentBasedRecommender, CollaborativeFilteringRecommender, HybridRecommender

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
DATA_FILE = BASE_DIR / "ted_talks_preprocessed.csv"

# Load preprocessed metadata
df = pd.read_csv(DATA_FILE)

# Load trained models
print("Loading recommendation models...")
content_rec = joblib.load(MODELS_DIR / "content_recommender.joblib")
hybrid_rec = joblib.load(MODELS_DIR / "hybrid_recommender.joblib")
print("Models loaded successfully!\n")


def display_recommendations(results, mode_name):
    if not results:
        print("\nNo matching talks found. Try a different query or topic.")
        return

    print(f"\n{'='*70}")
    print(f"Top {len(results)} Recommendations ({mode_name})")
    print(f"{'='*70}")

    for rank, (idx, score) in enumerate(results, start=1):
        talk = df.iloc[idx]
        title = str(talk["title"]).title()
        speaker = talk["main_speaker"]
        details = str(talk["details"])
        url = talk["url"]

        # Truncate details for readable CLI presentation
        short_details = details if len(details) <= 160 else details[:157] + "..."

        print(f"\n{rank}. {title}")
        print(f"   Speaker:     {speaker}")
        print(f"   Relevance:   {score:.4f}")
        print(f"   Description: {short_details}")
        if pd.notna(url) and str(url).strip():
            print(f"   URL:         {url}")
    print(f"\n{'='*70}\n")


def main():
    print("=" * 70)
    print("  PERSONALIZED TED TALKS RECOMMENDATION SYSTEM (CLI)")
    print("=" * 70)
    print("Modes:")
    print("  1. Search by Topic / Natural Language Query")
    print("  2. Recommend Similar Talks (Item-to-Item by Title or Index)")
    print("  Type 'exit' or 'quit' at any prompt to exit.\n")

    while True:
        try:
            choice = input("Select mode (1 for Topic Search, 2 for Similar Talks, 'exit' to quit): ").strip()
            if choice.lower() in ["exit", "quit", "q"]:
                print("Exiting TED Talks Recommender. Goodbye!")
                break

            if choice == "1":
                query = input("\nEnter topic / keywords (e.g. 'artificial intelligence', 'climate change'): ").strip()
                if query.lower() in ["exit", "quit"]:
                    break
                if not query:
                    print("Please enter a non-empty query.")
                    continue
                results = content_rec.recommend_for_query(query, top_n=5)
                display_recommendations(results, f"Topic: '{query}'")

            elif choice == "2":
                lookup = input("\nEnter Talk Index (0-4466) or partial title: ").strip()
                if lookup.lower() in ["exit", "quit"]:
                    break
                if not lookup:
                    continue

                target_idx = None
                if lookup.isdigit():
                    idx_val = int(lookup)
                    if 0 <= idx_val < len(df):
                        target_idx = idx_val
                    else:
                        print(f"Index out of range (0 to {len(df)-1}).")
                        continue
                else:
                    # Search matching titles
                    matches = df[df["title"].str.contains(lookup.lower(), na=False)]
                    if matches.empty:
                        print(f"No talk found containing '{lookup}'.")
                        continue
                    target_idx = matches.index[0]
                    print(f"Matched talk: '{df.iloc[target_idx]['title'].title()}' by {df.iloc[target_idx]['main_speaker']}")

                results = content_rec.recommend_similar_talks(target_idx, top_n=5)
                seed_title = df.iloc[target_idx]["title"].title()
                display_recommendations(results, f"Similar to: '{seed_title}'")

            else:
                print("Invalid selection. Please enter 1, 2, or 'exit'.")

        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break


if __name__ == "__main__":
    main()
