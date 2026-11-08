"""
Module 4: Recommendation Engine Implementation
Personalized TED Talks Recommendation System

Implements:
1. Content-Based Recommender (Item-to-Item, Profile-based, Query-based)
2. Collaborative Filtering Recommender (Matrix Factorization / TruncatedSVD)
3. Hybrid Recommender (Combines Content Similarity + Collaborative Scores)
"""

import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
DATA_FILE = BASE_DIR / "ted_talks_preprocessed.csv"


class ContentBasedRecommender:
    """
    Content-Based Recommendation Engine using TF-IDF text features & cosine similarity.
    Supports item-to-item, user preference profile, and query search recommendations.
    """
    def __init__(self, df, unified_vectorizer, unified_matrix, weighted_similarity=None):
        self.df = df.reset_index(drop=True)
        self.vectorizer = unified_vectorizer
        self.matrix = unified_matrix
        self.weighted_similarity = weighted_similarity
        # Index map for quick title/idx lookup
        self.title_to_idx = {t.lower().strip(): i for i, t in enumerate(self.df["title"])}
        self.id_to_idx = {str(uid): i for i, uid in enumerate(self.df["idx"])}

    def recommend_similar_talks(self, talk_idx, top_n=5, exclude_self=True):
        """Recommend talks similar to a given talk index."""
        if self.weighted_similarity is not None:
            # Sparse or dense row lookup
            if hasattr(self.weighted_similarity, "toarray"):
                scores = self.weighted_similarity[talk_idx].toarray().flatten()
            else:
                scores = np.array(self.weighted_similarity[talk_idx]).flatten()
        else:
            talk_vec = self.matrix[talk_idx]
            scores = cosine_similarity(talk_vec, self.matrix).flatten()

        ranked = scores.argsort()[::-1]
        results = []
        for idx in ranked:
            if exclude_self and idx == talk_idx:
                continue
            results.append((int(idx), float(scores[idx])))
            if len(results) >= top_n:
                break
        return results

    def recommend_for_profile(self, liked_talk_indices, top_n=5):
        """
        Generate recommendations given a list of talk indices that a user liked/watched.
        Computes a centroid preference vector in TF-IDF feature space.
        """
        if not liked_talk_indices:
            return []
        
        # User profile vector is mean vector of liked talks
        user_vector = self.matrix[liked_talk_indices].mean(axis=0)
        # Convert to numpy array if matrix is sparse matrix or matrix type
        user_vector = np.asarray(user_vector)
        
        scores = cosine_similarity(user_vector, self.matrix).flatten()
        ranked = scores.argsort()[::-1]
        
        results = []
        liked_set = set(liked_talk_indices)
        for idx in ranked:
            if idx in liked_set:
                continue
            results.append((int(idx), float(scores[idx])))
            if len(results) >= top_n:
                break
        return results

    def recommend_for_query(self, query_text, top_n=5):
        """Search talks matching an arbitrary natural language topic query."""
        q_vec = self.vectorizer.transform([query_text])
        if q_vec.nnz == 0:
            return []
        scores = cosine_similarity(q_vec, self.matrix).flatten()
        ranked = scores.argsort()[::-1]
        results = [(int(idx), float(scores[idx])) for idx in ranked[:top_n] if scores[idx] > 0]
        return results


class CollaborativeFilteringRecommender:
    """
    Collaborative Filtering Engine using TruncatedSVD (Matrix Factorization).
    Decomposes user-item interaction matrix into latent factors: R ~ U * Sigma * V^T.
    """
    def __init__(self, n_components=20, random_state=42):
        self.n_components = n_components
        self.random_state = random_state
        self.svd = TruncatedSVD(n_components=n_components, random_state=random_state)
        self.user_factors = None
        self.item_factors = None
        self.user_item_matrix = None

    def fit(self, user_item_matrix):
        """Fit latent factor model on sparse user-item interaction matrix."""
        self.user_item_matrix = user_item_matrix
        # Decompose matrix
        self.user_factors = self.svd.fit_transform(user_item_matrix)
        self.item_factors = self.svd.components_  # shape: (n_components, n_items)
        return self

    def predict_user_scores(self, user_idx):
        """Predict preference scores for all items for a given user."""
        u_vector = self.user_factors[user_idx]  # shape: (n_components,)
        predicted_scores = np.dot(u_vector, self.item_factors)  # shape: (n_items,)
        return predicted_scores

    def recommend_for_user(self, user_idx, top_n=5, exclude_watched=True):
        """Recommend items for an existing user in the interaction matrix."""
        scores = self.predict_user_scores(user_idx)
        
        if exclude_watched and self.user_item_matrix is not None:
            watched_items = self.user_item_matrix[user_idx].indices
            scores[watched_items] = -np.inf
            
        ranked = scores.argsort()[::-1]
        results = [(int(idx), float(scores[idx])) for idx in ranked[:top_n] if scores[idx] > -np.inf]
        return results


class HybridRecommender:
    """
    Hybrid Recommendation Engine:
    Combines Content-Based similarity signals with Collaborative Filtering latent signals.
    Score = alpha * Normalized(ContentScore) + (1 - alpha) * Normalized(CollabScore)
    """
    def __init__(self, content_engine, collab_engine, alpha=0.65):
        self.content_engine = content_engine
        self.collab_engine = collab_engine
        self.alpha = alpha

    def recommend_for_user(self, user_idx, liked_talk_indices, top_n=5):
        """
        Produce hybrid recommendations blending user profile content similarity
        and collaborative filtering latent ratings.
        """
        num_items = len(self.content_engine.df)
        
        # 1. Content scores from profile
        user_vector = np.asarray(self.content_engine.matrix[liked_talk_indices].mean(axis=0))
        content_scores = cosine_similarity(user_vector, self.content_engine.matrix).flatten()
        # Min-max scale content scores
        c_min, c_max = content_scores.min(), content_scores.max()
        c_norm = (content_scores - c_min) / (c_max - c_min + 1e-9)

        # 2. Collaborative scores
        collab_scores = self.collab_engine.predict_user_scores(user_idx)
        cf_min, cf_max = collab_scores.min(), collab_scores.max()
        cf_norm = (collab_scores - cf_min) / (cf_max - cf_min + 1e-9)

        # 3. Hybrid blended score
        hybrid_scores = self.alpha * c_norm + (1 - self.alpha) * cf_norm
        
        # Mask already watched items
        for item in liked_talk_indices:
            hybrid_scores[item] = -1.0
            
        ranked = hybrid_scores.argsort()[::-1]
        results = [(int(idx), float(hybrid_scores[idx])) for idx in ranked[:top_n] if hybrid_scores[idx] > 0]
        return results


def train_and_save_recommender():
    """Train all recommendation models and persist them for deployment."""
    print("=" * 60)
    print("MODULE 4: TRAINING & SAVING RECOMMENDATION ENGINES")
    print("=" * 60)
    
    df = pd.read_csv(DATA_FILE)
    unified_vec = joblib.load(MODELS_DIR / "unified_vectorizer.joblib")
    unified_mat = joblib.load(MODELS_DIR / "unified_tfidf_matrix.joblib")
    weighted_sim = joblib.load(MODELS_DIR / "weighted_similarity_matrix.joblib")
    user_item_mat = joblib.load(MODELS_DIR / "user_item_matrix.joblib")

    print("\n[1/3] Initializing Content-Based Engine...")
    content_rec = ContentBasedRecommender(df, unified_vec, unified_mat, weighted_sim)

    print("\n[2/3] Training Collaborative Filtering SVD Engine...")
    cf_rec = CollaborativeFilteringRecommender(n_components=25, random_state=42)
    cf_rec.fit(user_item_mat)
    print(f"  TruncatedSVD Explained Variance Ratio Sum: {cf_rec.svd.explained_variance_ratio_.sum():.4f}")

    print("\n[3/3] Initializing Hybrid Engine (alpha=0.65 Content, 0.35 Collaborative)...")
    hybrid_rec = HybridRecommender(content_rec, cf_rec, alpha=0.65)

    # Save trained models
    print("\nSaving trained models to 'models/'...")
    joblib.dump(content_rec, MODELS_DIR / "content_recommender.joblib")
    joblib.dump(cf_rec, MODELS_DIR / "cf_recommender.joblib")
    joblib.dump(hybrid_rec, MODELS_DIR / "hybrid_recommender.joblib")
    
    # Quick sanity check test
    print("\n--- Sanity Check: Recommendations for talk index 0 ---")
    test_title = df.iloc[0]["title"]
    print(f"Seed Talk: '{test_title}' by {df.iloc[0]['main_speaker']}")
    recs = content_rec.recommend_similar_talks(0, top_n=3)
    for rank, (idx, score) in enumerate(recs, 1):
        row = df.iloc[idx]
        print(f"  {rank}. {row['title'].title()} (Score: {score:.3f}) by {row['main_speaker']}")

    print("\n--- Sanity Check: Recommendations for User 0 (Hybrid) ---")
    user0_watched = list(user_item_mat[0].indices)
    hybrid_recs = hybrid_rec.recommend_for_user(0, user0_watched, top_n=3)
    for rank, (idx, score) in enumerate(hybrid_recs, 1):
        row = df.iloc[idx]
        print(f"  {rank}. {row['title'].title()} (Hybrid Score: {score:.3f}) by {row['main_speaker']}")

    print("\nModule 4: Recommendation Engine Built & Persisted Successfully!")
    return content_rec, cf_rec, hybrid_rec


if __name__ == "__main__":
    train_and_save_recommender()

