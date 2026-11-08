"""
Module 3: Feature Engineering, Vectorization & Interaction Matrix Construction
Personalized TED Talks Recommendation System
"""

import os
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

DATA_FILE = BASE_DIR / "ted_talks_preprocessed.csv"
df = pd.read_csv(DATA_FILE)

print("=" * 60)
print("MODULE 3: FEATURE ENGINEERING & VECTORIZATION")
print("=" * 60)

# Fill text nulls
titles = df["title"].fillna("").astype(str)
details = df["details"].fillna("").astype(str)
speakers = df["main_speaker"].fillna("Unknown").astype(str)
combined = df["combined_text"].fillna("").astype(str)

# 1. Title TF-IDF Vectorizer (emphasizes salient keywords and n-grams in titles)
print("\n[1/5] Fitting Title TF-IDF Vectorizer...")
title_vectorizer = TfidfVectorizer(
    stop_words="english",
    ngram_range=(1, 2),
    max_features=5000,
    sublinear_tf=True
)
title_matrix = title_vectorizer.fit_transform(titles)
print(f"  Title Matrix Shape: {title_matrix.shape}")

# 2. Details TF-IDF Vectorizer (captures broad semantic context and discourse)
print("\n[2/5] Fitting Details TF-IDF Vectorizer...")
details_vectorizer = TfidfVectorizer(
    stop_words="english",
    ngram_range=(1, 2),
    max_features=15000,
    sublinear_tf=True
)
details_matrix = details_vectorizer.fit_transform(details)
print(f"  Details Matrix Shape: {details_matrix.shape}")

# 3. Unified Content TF-IDF (Combined title, details, and speaker)
print("\n[3/5] Fitting Unified TF-IDF Vectorizer on Combined Text...")
unified_vectorizer = TfidfVectorizer(
    stop_words="english",
    ngram_range=(1, 2),
    max_features=10000,
    sublinear_tf=True
)
unified_matrix = unified_vectorizer.fit_transform(combined)
print(f"  Unified Matrix Shape: {unified_matrix.shape}")

# 4. Content-Based Precomputed Item-Item Similarity Matrix (top 100 neighbors per talk for efficiency)
print("\n[4/5] Computing Pairwise Item-Item Similarity Matrix...")
# Weighted similarity: 60% title similarity + 40% details similarity
# We compute in batches or full matrix (4467 x 4467 float32 ~ 79 MB)
sim_title = cosine_similarity(title_matrix, dense_output=False)
sim_details = cosine_similarity(details_matrix, dense_output=False)

# Weighted combination as CSR sparse matrix
weighted_similarity = 0.60 * sim_title + 0.40 * sim_details

# 5. User-Item Interaction Simulation (for Collaborative & Hybrid Filtering)
# TEDx dataset does not contain individual user activity logs.
# To satisfy Module 3 requirements ("construct and sparsely represent user-item interaction matrices"),
# we synthesize a realistic, clustered interaction history for 500 users across 5 topic affinities:
# Technology/AI, Psychology/Mind, Climate/Nature, Education/Society, Business/Leadership.
print("\n[5/5] Generating Synthetic User-Item Interaction Matrix for Collaborative Filtering...")
np.random.seed(42)
num_users = 500
num_items = len(df)

# Cluster keywords to define user personas
clusters = {
    0: ["tech", "ai", "computer", "data", "internet", "robot", "digital", "algorithm"],
    1: ["mind", "brain", "psychology", "mental", "happiness", "emotion", "behavior"],
    2: ["climate", "energy", "planet", "earth", "nature", "environment", "water"],
    3: ["school", "education", "art", "music", "children", "learning", "story"],
    4: ["business", "leadership", "money", "economy", "work", "organization", "innovation"]
}

# Assign primary cluster affinity to each talk
talk_text = df["combined_text"].str.lower().values
talk_clusters = np.zeros(num_items, dtype=int)
for i, txt in enumerate(talk_text):
    scores = [sum(txt.count(w) for w in keywords) for keywords in clusters.values()]
    talk_clusters[i] = int(np.argmax(scores)) if max(scores) > 0 else np.random.randint(0, 5)

interactions = []
user_ids = []
item_indices = []
ratings = []

for u in range(num_users):
    # User's favorite cluster and secondary cluster
    fav_cluster = u % 5
    sec_cluster = (u + 1) % 5
    
    # User watches 15-30 talks
    n_interactions = np.random.randint(15, 30)
    
    # 70% from favorite, 20% from secondary, 10% random exploration
    fav_items = np.where(talk_clusters == fav_cluster)[0]
    sec_items = np.where(talk_clusters == sec_cluster)[0]
    all_items = np.arange(num_items)
    
    watched = set()
    for _ in range(n_interactions):
        r = np.random.rand()
        if r < 0.70 and len(fav_items) > 0:
            item = np.random.choice(fav_items)
        elif r < 0.90 and len(sec_items) > 0:
            item = np.random.choice(sec_items)
        else:
            item = np.random.choice(all_items)
        
        if item not in watched:
            watched.add(item)
            rating = np.random.choice([4.0, 5.0], p=[0.35, 0.65]) if talk_clusters[item] == fav_cluster else np.random.choice([3.0, 4.0, 5.0], p=[0.4, 0.4, 0.2])
            interactions.append({"user_id": f"user_{u:04d}", "talk_idx": df.iloc[item]["idx"], "talk_row": item, "rating": rating})
            user_ids.append(u)
            item_indices.append(item)
            ratings.append(rating)

interactions_df = pd.DataFrame(interactions)
interactions_csv = MODELS_DIR / "synthetic_user_interactions.csv"
interactions_df.to_csv(interactions_csv, index=False)
print(f"  Generated {len(interactions_df)} interactions across {num_users} users.")

# Build Sparse User-Item Matrix
user_item_matrix = csr_matrix((ratings, (user_ids, item_indices)), shape=(num_users, num_items))
print(f"  User-Item Matrix Shape: {user_item_matrix.shape}, Sparsity: {100 * (1 - user_item_matrix.nnz / (num_users * num_items)):.2f}%")

# Save all models & matrices
print("\nSaving Feature Engineering Artifacts to 'models/'...")
joblib.dump(title_vectorizer, MODELS_DIR / "title_vectorizer.joblib")
joblib.dump(details_vectorizer, MODELS_DIR / "details_vectorizer.joblib")
joblib.dump(unified_vectorizer, MODELS_DIR / "unified_vectorizer.joblib")
joblib.dump(unified_matrix, MODELS_DIR / "unified_tfidf_matrix.joblib")
joblib.dump(weighted_similarity, MODELS_DIR / "weighted_similarity_matrix.joblib")
joblib.dump(user_item_matrix, MODELS_DIR / "user_item_matrix.joblib")

print("Module 3 Feature Engineering & Vectorization Completed Successfully!")

