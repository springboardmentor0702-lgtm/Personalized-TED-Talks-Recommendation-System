"""
TED Talks Recommendation System - Streamlit Application
========================================================
Clean, simple, and modern interface for TED Talk recommendations.
No emojis, clean layout, native Streamlit UI components.
"""

import sys
from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.model_loader import load_artifacts, ArtifactBundle
from src.recommender import RecommenderService
from src.recommendation_utils import parse_tags_list, format_views, format_duration

# ==============================================================================
# Page Configuration
# ==============================================================================
st.set_page_config(
    page_title="TED Talks Recommendation System",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ==============================================================================
# Model & Resource Caching
# ==============================================================================
@st.cache_resource(show_spinner="Loading recommendation models and catalog...")
def get_service() -> RecommenderService:
    bundle = load_artifacts(PROJECT_ROOT)
    return RecommenderService(bundle)

try:
    service = get_service()
    bundle = service.bundle
    catalog = service.catalog
except Exception as e:
    st.error(f"Error loading recommendation artifacts: {e}")
    st.info("Please ensure model files exist in the models/ directory.")
    st.stop()

# ==============================================================================
# Sidebar Navigation
# ==============================================================================
with st.sidebar:
    st.title("TED Recommender")
    st.caption("Content-Based Recommendation Engine")
    st.divider()
    
    navigation = st.radio(
        "Navigation",
        options=[
            "Talk-to-Talk Recommendations",
            "Natural Language Search",
            "Topic Tags Filtering",
            "User Preferences",
            "Collaborative Filtering Telemetry",
            "Evaluation Metrics",
            "System Information",
        ],
        index=0,
    )
    
    st.divider()
    st.subheader("Catalog Overview")
    st.text(f"Total Talks: {len(catalog):,}")
    st.text("Embeddings: 384-dim (Sentence-BERT)")
    st.text("Vocabulary: 10,000 TF-IDF features")
    st.text("Topics: 464 taxonomy tags")

# ==============================================================================
# Helper Renderer for Recommendation Cards using Native Containers
# ==============================================================================
def render_talk_card(rec: dict, is_seed: bool = False, rank: int = 1):
    with st.container(border=True):
        header_col1, header_col2 = st.columns([3, 1])
        with header_col1:
            if is_seed:
                st.caption("SELECTED SEED TALK")
            else:
                st.caption(f"RECOMMENDATION #{rank}")
            st.subheader(rec['title'])
            st.write(f"**Presenter:** {rec['speaker']} | **Year:** {rec['year']} | **Event:** {rec['event']}")
        with header_col2:
            if not is_seed and "score" in rec:
                pct = max(0.0, min(100.0, rec["score"] * 100))
                st.metric(label="Match Score", value=f"{pct:.1f}%")

        m1, m2, m3 = st.columns(3)
        m1.text(f"Duration: {rec['duration_str']}")
        m2.text(f"Views: {rec['views_str']}")
        m3.text(f"Talk ID: {rec['talk_id']}")

        tags = rec.get("tags", [])
        if tags:
            st.caption("Topics: " + ", ".join(tags[:6]))

        explanation = rec.get('explanation')
        if explanation:
            st.info(explanation)

        col_desc, col_btn = st.columns([4, 1])
        with col_desc:
            with st.expander("Talk Description"):
                st.write(rec.get("description", "No description available."))
        with col_btn:
            st.link_button("View on TED", rec.get("url", "https://www.ted.com"), use_container_width=True)

# ==============================================================================
# 1. Talk-to-Talk Recommendations
# ==============================================================================
if navigation == "Talk-to-Talk Recommendations":
    st.title("Talk-to-Talk Recommendations")
    st.write("Select a talk from the catalog to find similar talks based on speech content, themes, and taxonomy.")
    
    # Pre-formatted titles
    options_display = [
        f"{row['title']} - {row['speaker_1']} ({row['published_year']})"
        for _, row in catalog.iterrows()
    ]
    
    col_input, col_ctrl = st.columns([3, 1])
    with col_input:
        default_idx = 66 if len(catalog) > 66 else 0
        selected_display = st.selectbox(
            "Select a TED Talk:",
            options=options_display,
            index=default_idx,
            help="Type to search any title or speaker across the catalog.",
        )
        selected_idx = options_display.index(selected_display)
        selected_talk_id = int(catalog.iloc[selected_idx]["talk_id"])
        
    with col_ctrl:
        available_methods = bundle.get_available_methods()
        method_key = st.selectbox(
            "Recommendation Engine:",
            options=list(available_methods.keys()),
            format_func=lambda k: available_methods[k].split(" (")[0],
            index=0,
        )
        top_k = st.slider("Number of Recommendations:", min_value=3, max_value=20, value=5, step=1)
        
    if st.button("Generate Recommendations", type="primary", use_container_width=True):
        with st.spinner("Calculating recommendations..."):
            try:
                seed_talk, recommendations = service.recommend_from_talk(
                    selected_talk_id,
                    top_k=top_k,
                    method=method_key,
                )
                
                st.divider()
                st.subheader("Seed Talk")
                render_talk_card(seed_talk, is_seed=True)
                
                st.divider()
                st.subheader(f"Recommended Talks (Top {len(recommendations)})")
                for rank, rec in enumerate(recommendations, 1):
                    render_talk_card(rec, is_seed=False, rank=rank)
            except Exception as e:
                st.error(f"Error generating recommendations: {e}")

# ==============================================================================
# 2. Natural Language Search
# ==============================================================================
elif navigation == "Natural Language Search":
    st.title("Natural Language Semantic Search")
    st.write("Enter any query, question, or concept to retrieve matching talks using dense vector embeddings.")
    
    example_prompts = [
        "How can I overcome procrastination, build better habits, and stay focused?",
        "What are the ethical implications and future trajectory of Artificial Intelligence?",
        "How does human memory work, and can we reverse neurodegenerative diseases?",
        "Inspiring stories about resilience, overcoming failure, and discovering purpose",
    ]
    
    selected_example = st.selectbox("Example prompts:", ["-- Enter Custom Query --"] + example_prompts)
    default_text = "" if selected_example == "-- Enter Custom Query --" else selected_example
    user_query = st.text_area("Search query:", value=default_text, height=90)
    
    col_q1, col_q2 = st.columns([2, 1])
    with col_q1:
        search_engine = st.radio(
            "Retrieval Method:",
            options=["semantic", "keyword"],
            format_func=lambda x: "Sentence-BERT Contextual Embeddings" if x == "semantic" else "TF-IDF Lexical Keyword Matching",
            horizontal=True,
        )
    with col_q2:
        top_k_query = st.slider("Results count:", min_value=3, max_value=20, value=5)
        
    if st.button("Search", type="primary", use_container_width=True):
        if not user_query.strip():
            st.warning("Please enter a query.")
        else:
            with st.spinner("Searching catalog..."):
                results = service.recommend_by_query(user_query, top_k=top_k_query, engine=search_engine)
                if not results:
                    st.info("No matching talks found.")
                else:
                    st.divider()
                    st.subheader(f"Search Results ({len(results)} matches)")
                    for rank, rec in enumerate(results, 1):
                        render_talk_card(rec, is_seed=False, rank=rank)

# ==============================================================================
# 3. Topic Tags Filtering
# ==============================================================================
elif navigation == "Topic Tags Filtering":
    st.title("Topic Tags Filtering")
    st.write("Select one or more topics from the curated TED taxonomy to rank matching talks.")
    
    all_tags = sorted(list(set([t for tags in catalog["tags_text"].dropna() for t in tags.split()])))
    clean_tags_map = {t: t.replace("_", " ").title() for t in all_tags}
    
    col_tags, col_tags_ctrl = st.columns([3, 1])
    with col_tags:
        selected_raw_tags = st.multiselect(
            "Select topics:",
            options=all_tags,
            default=["education", "creativity"] if "education" in all_tags else all_tags[:2],
            format_func=lambda t: clean_tags_map.get(t, t),
        )
    with col_tags_ctrl:
        top_k_tags = st.slider("Result Count:", min_value=3, max_value=25, value=5)
        blend_pop = st.checkbox("Include Viewership Factor", value=True)
        
    if st.button("Find Talks", type="primary", use_container_width=True):
        if not selected_raw_tags:
            st.warning("Please select at least one topic.")
        else:
            with st.spinner("Filtering talks..."):
                recs = service.recommend_by_tags(
                    selected_tags=selected_raw_tags,
                    top_k=top_k_tags,
                    blend_popularity=blend_pop,
                )
                if not recs:
                    st.info("No talks found matching the selected topics.")
                else:
                    st.divider()
                    st.subheader(f"Filtered Talks ({len(recs)} matches)")
                    for rank, rec in enumerate(recs, 1):
                        render_talk_card(rec, is_seed=False, rank=rank)

# ==============================================================================
# 4. User Preferences
# ==============================================================================
elif navigation == "User Preferences":
    st.title("Personalized Recommendations")
    st.write("Select your preferred topics and previously watched talks to build your personalized recommendation feed.")
    
    col_p1, col_p2 = st.columns(2)
    with col_p1:
        st.subheader("Topics of Interest")
        popular_topics = ["science", "technology", "culture", "design", "business", "global issues", "health", "psychology", "education", "creativity"]
        user_topics = st.multiselect("Select topics:", options=popular_topics, default=["science", "technology"])
        
    with col_p2:
        st.subheader("Previously Liked Talks")
        liked_samples = catalog.sort_values(by="views", ascending=False).head(25)
        liked_options = {int(r["talk_id"]): f"{r['title']} - {r['speaker_1']}" for _, r in liked_samples.iterrows()}
        liked_tids = st.multiselect(
            "Select talks you enjoyed:",
            options=list(liked_options.keys()),
            format_func=lambda tid: liked_options[tid],
            default=[66] if 66 in liked_options else [],
        )
        
    st.subheader("Discovery Preference")
    niche_balance = st.slider(
        "Bias toward niche talks vs. popular talks:",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.1,
        help="0.0 emphasizes focused niche talks, while 1.0 emphasizes high-viewership talks.",
    )
    
    if st.button("Generate Recommendations", type="primary", use_container_width=True):
        with st.spinner("Generating personalized feed..."):
            profile_recs = service.recommend_by_profile(
                preferred_topics=user_topics,
                liked_talk_ids=liked_tids,
                top_k=6,
                niche_vs_popular=niche_balance,
            )
            st.divider()
            st.subheader("Recommended for You")
            for rank, rec in enumerate(profile_recs, 1):
                render_talk_card(rec, is_seed=False, rank=rank)

# ==============================================================================
# 5. Collaborative Filtering Telemetry
# ==============================================================================
elif navigation == "Collaborative Filtering Telemetry":
    st.title("Collaborative Filtering Telemetry Explorer")
    st.write("Explore matrix factorization predictions on the benchmark user interaction telemetry dataset.")
    
    st.info(
        "Note: The public TED Talks dataset does not contain individual user login sessions. "
        "This collaborative filtering model was trained on 36,557 simulated user interaction records "
        "across 1,000 persona cohorts as an architectural benchmark."
    )
    
    col_u1, col_u2 = st.columns([1, 2])
    with col_u1:
        user_id_input = st.number_input("User ID (0 to 999):", min_value=0, max_value=999, value=0, step=1)
        top_k_cf = st.slider("Result Count:", min_value=3, max_value=15, value=5)
        
    if st.button("Query Matrix Factorization Model", type="primary"):
        try:
            meta, cf_recs = service.get_cf_benchmark_user_recommendations(user_id=int(user_id_input), top_k=top_k_cf)
            with col_u2:
                st.write(f"**Persona:** {meta['persona']}")
                st.write(f"**Historical Talks Watched:** {meta['watched_count']}")
                if meta["history_sample"]:
                    st.caption("Sample history: " + ", ".join([h["title"] for h in meta["history_sample"][:3]]))
                    
            st.divider()
            st.subheader(f"Recommendations for User #{user_id_input}")
            for rank, rec in enumerate(cf_recs, 1):
                render_talk_card(rec, is_seed=False, rank=rank)
        except Exception as e:
            st.error(f"Error running model: {e}")

# ==============================================================================
# 6. Evaluation Metrics
# ==============================================================================
elif navigation == "Evaluation Metrics":
    st.title("Model Evaluation and Benchmark Metrics")
    st.write("Quantitative performance comparison across candidate recommendation architectures.")
    
    benchmark_df = pd.DataFrame({
        "Model Architecture": [
            "Multi-Signal Hybrid Engine (Leader)",
            "TF-IDF Content Baseline",
            "Sentence-BERT (all-MiniLM-L6-v2)",
            "Topic Tags Baseline",
            "Latent Semantic Analysis (TruncatedSVD)",
            "Word2Vec (TF-IDF Weighted)",
        ],
        "Precision@5": [0.1848, 0.1664, 0.1416, 0.1288, 0.1240, 0.0152],
        "Recall@5": [0.1551, 0.1389, 0.1192, 0.1082, 0.1040, 0.0127],
        "MRR": [0.4341, 0.3871, 0.3764, 0.2779, 0.2863, 0.0332],
        "NDCG@5": [0.2111, 0.1903, 0.1702, 0.1387, 0.1393, 0.0148],
        "Retrieval Latency": ["< 4 ms", "< 1 ms", "< 2 ms", "< 1 ms", "< 1 ms", "< 1 ms"],
    })
    
    st.dataframe(benchmark_df, use_container_width=True)
    
    vis_dir = PROJECT_ROOT / "assets" / "visualizations"
    col_img1, col_img2 = st.columns(2)
    if (vis_dir / "benchmark_comparison.png").exists():
        with col_img1:
            st.image(str(vis_dir / "benchmark_comparison.png"), caption="Model Comparison across Precision, Recall, MRR, and NDCG")
    if (vis_dir / "hybrid_weights.png").exists():
        with col_img2:
            st.image(str(vis_dir / "hybrid_weights.png"), caption="Multi-Signal Hybrid Engine Weight Distribution")

# ==============================================================================
# 7. System Information
# ==============================================================================
elif navigation == "System Information":
    st.title("System Specifications")
    st.write("Technical summary of models, representations, and evaluation methodology.")
    
    st.subheader("1. Models and Representations")
    st.write("- **Sentence-BERT:** Pretrained `sentence-transformers/all-MiniLM-L6-v2` dense vectors (384 dimensions).")
    st.write("- **TF-IDF Content:** 10,000 sublinear unigram and bigram features across transcripts, descriptions, and titles.")
    st.write("- **TF-IDF Tags:** 464 discrete topic categories from the official TED taxonomy.")
    st.write("- **Hybrid Model:** Weighted combination of semantic similarity (50%), topic tag overlap (25%), speaker continuity (15%), and log-viewership popularity prior (10%).")
    st.write("- **Matrix Factorization:** TruncatedSVD with 30 latent dimensions for collaborative filtering telemetry.")

    st.subheader("2. Evaluation Strategy")
    st.write("- **Editorial Ground Truth:** Human-curated `related_talks` editorial relationships.")
    st.write("- **Ranking Metrics:** Precision@5, Recall@5, Mean Reciprocal Rank (MRR), and NDCG@5.")
    st.write("- **Performance Summary:** The hybrid architecture achieves a +11.1% Precision gain and +12.1% MRR gain over individual baseline representations.")
