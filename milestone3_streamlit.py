"""
TED Talks Recommendation System - Interactive Streamlit Application
===================================================================
Production deployment interface integrating all verified Milestone 1 & 2 models:
- Multi-Signal Hybrid Engine (Leader: P@5=0.1848, MRR=0.4341)
- Sentence-BERT Semantic Recommender (all-MiniLM-L6-v2)
- TF-IDF Lexical Content Recommender (10k Unigrams/Bigrams)
- Topic Tags Curated Recommender (464 Taxonomy Features)
- Latent Semantic Analysis (TruncatedSVD 100-dim)
- Word2Vec TF-IDF-Weighted Distributed Embeddings
- Natural-Language Free-Text Query Retrieval
- Personalized User Profile Recommendations
- Collaborative Filtering SVD Telemetry Benchmark
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

#from src.model_loader import load_artifacts, ArtifactBundle
from src.recommender import RecommenderService
from src.recommendation_utils import parse_tags_list, format_views, format_duration

# ==============================================================================
# Page Configuration & Styling
# ==============================================================================
st.set_page_config(
    page_title="TED Talks Recommendation System",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Theme CSS
st.markdown(
    """
    <style>
    /* TED Theme Palette */
    :root {
        --ted-red: #E62B1E;
        --ted-dark-red: #B81D13;
        --ted-black: #111111;
        --ted-gray: #2B2D42;
        --ted-light: #F4F5F7;
    }
    
    .main-title {
        font-size: 2.3rem;
        font-weight: 800;
        color: #E62B1E;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.1rem;
        color: #8D99AE;
        margin-bottom: 1.5rem;
    }
    
    /* Talk Card Styling */
    .talk-card {
        background-color: #1A1D24;
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 18px;
        border-left: 5px solid #E62B1E;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.25);
    }
    .talk-card-seed {
        background-color: #20242D;
        border-radius: 10px;
        padding: 22px;
        margin-bottom: 25px;
        border-left: 6px solid #4CAF50;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.3);
    }
    .talk-title {
        font-size: 1.25rem;
        font-weight: 700;
        color: #FFFFFF;
        margin-bottom: 4px;
    }
    .talk-speaker {
        font-size: 1.0rem;
        font-weight: 600;
        color: #FF6F59;
        margin-bottom: 8px;
    }
    .metric-badge {
        display: inline-block;
        background-color: #2B2D42;
        color: #EDF2F4;
        font-size: 0.8rem;
        padding: 3px 8px;
        border-radius: 4px;
        margin-right: 6px;
        margin-bottom: 6px;
    }
    .score-badge {
        display: inline-block;
        background-color: #E62B1E;
        color: white;
        font-size: 0.85rem;
        font-weight: 700;
        padding: 4px 10px;
        border-radius: 5px;
        float: right;
    }
    .tag-badge {
        display: inline-block;
        background-color: #383B46;
        color: #D1D5DB;
        font-size: 0.75rem;
        padding: 2px 7px;
        border-radius: 3px;
        margin-right: 4px;
        margin-top: 5px;
    }
    .explanation-box {
        background-color: #121418;
        border-left: 3px solid #8D99AE;
        padding: 10px 14px;
        border-radius: 4px;
        font-size: 0.88rem;
        color: #C0C5CE;
        margin-top: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ==============================================================================
# Model & Resource Caching
# ==============================================================================
@st.cache_resource(show_spinner="Loading verified TED recommendation artifacts...")
def get_service() -> RecommenderService:
    """Load and cache artifacts and recommender service once per process."""
    bundle = load_artifacts(PROJECT_ROOT)
    return RecommenderService(bundle)


try:
    service = get_service()
    bundle = service.bundle
    catalog = service.catalog
except Exception as e:
    st.error(f"Failed to load recommendation artifacts: {e}")
    st.info("Please verify that models exist in the `models/` directory.")
    st.stop()


# ==============================================================================
# Sidebar Navigation & Diagnostics
# ==============================================================================
with st.sidebar:
    st.image(
        "https://upload.wikimedia.org/wikipedia/commons/thumb/0/09/TED_three_letter_logo.svg/320px-TED_three_letter_logo.svg.png",
        width=130,
    )
    st.markdown("### Recommendation Engine")
    st.caption("Milestone 4 Deployment • Springboard ML")
    
    st.markdown("---")
    navigation = st.radio(
        "Select Interaction Mode",
        options=[
            "🎯 Talk-to-Talk Recommendations",
            "🔍 Semantic Natural-Language Search",
            "🏷️ Topic Tags Discovery",
            "👤 Personalized User Profile",
            "📊 Collaborative Filtering Benchmark",
            "📈 Architecture Benchmarks & Metrics",
            "ℹ️ Model Card & Telemetry Info",
        ],
        index=0,
    )
    
    st.markdown("---")
    st.markdown("#### System Health & Telemetry")
    st.markdown(f"- **Indexed Talks:** `{len(catalog):,}`")
    st.markdown(f"- **Embeddings:** `384-dim (MiniLM-L6)`")
    st.markdown(f"- **Vocabulary:** `10,000 terms (TF-IDF)`")
    st.markdown(f"- **Taxonomy:** `464 curated tags`")
    st.markdown(f"- **Hybrid Leader:** `Precision@5 = 18.5%`")
    
    st.markdown("---")
    st.caption("Developed by Senior ML / NLP Engineer • Fall 2026")


# ==============================================================================
# Helper UI Renderer for Recommendation Cards
# ==============================================================================
def render_talk_card(rec: dict, is_seed: bool = False, rank: int = 1):
    card_class = "talk-card-seed" if is_seed else "talk-card"
    header_prefix = "SEED TALK" if is_seed else f"RANK #{rank}"
    
    score_html = ""
    if not is_seed and "score" in rec:
        score_pct = max(0.0, min(100.0, rec["score"] * 100))
        score_html = f"<span class='score-badge'>Match: {score_pct:.1f}%</span>"
        
    tag_pills = "".join([f"<span class='tag-badge'>{t}</span>" for t in rec["tags"][:6]])
    
    st.markdown(
        f"""
        <div class="{card_class}">
            {score_html}
            <div style="font-size: 0.8rem; font-weight: bold; color: {'#4CAF50' if is_seed else '#E62B1E'}; text-transform: uppercase; margin-bottom: 4px;">
                {header_prefix}
            </div>
            <div class="talk-title">{rec['title']}</div>
            <div class="talk-speaker">{rec['speaker']} &bull; {rec['year']} &bull; {rec['event']}</div>
            <div>
                <span class="metric-badge">⏱️ {rec['duration_str']}</span>
                <span class="metric-badge">👁️ {rec['views_str']} views</span>
                <span class="metric-badge">🆔 ID: {rec['talk_id']}</span>
            </div>
            <div style="margin-top: 8px;">
                {tag_pills}
            </div>
            <div class="explanation-box">
                {rec.get('explanation', 'Recommended based on content similarity.')}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    
    # Description expander & Watch Button
    col1, col2 = st.columns([4, 1])
    with col1:
        with st.expander("📖 Read Talk Description / Abstract"):
            st.write(rec["description"])
    with col2:
        st.link_button("▶️ Watch on TED", rec["url"], use_container_width=True)


# ==============================================================================
# MODE 1: Talk-to-Talk Recommendations
# ==============================================================================
if navigation == "🎯 Talk-to-Talk Recommendations":
    st.markdown("<div class='main-title'>🎙️ TED Talk-to-Talk Recommendations</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='sub-title'>Select any of the 4,005 TED Talks to discover mathematically similar talks using verified Milestone 2 models.</div>",
        unsafe_allow_html=True,
    )
    
    # Searchable talk picker
    all_titles = catalog["title"].tolist()
    # Format options nicely with speaker and year
    options_display = [f"{row['title']} — {row['speaker_1']} ({row['published_year']})" for _, row in catalog.iterrows()]
    
    col_input, col_ctrl = st.columns([3, 1])
    
    with col_input:
        default_index = 66 if len(catalog) > 66 else 0  # Ken Robinson "Do schools kill creativity?"
        selected_display = st.selectbox(
            "Select or Search a TED Talk:",
            options=options_display,
            index=default_index,
            help="Type to search any title or speaker across the 4,005 catalog talks.",
        )
        selected_idx = options_display.index(selected_display)
        selected_talk_id = int(catalog.iloc[selected_idx]["talk_id"])
        
    with col_ctrl:
        available_methods = bundle.get_available_methods()
        method_key = st.selectbox(
            "Recommendation Engine:",
            options=list(available_methods.keys()),
            format_func=lambda k: available_methods[k],
            index=0,
            help="Choose between the Multi-Signal Hybrid engine or specific baseline representations.",
        )
        top_k = st.slider("Recommendations (k):", min_value=3, max_value=20, value=5, step=1)

    if st.button("🚀 Generate Recommendations", type="primary", use_container_width=True):
        with st.spinner("Retrieving top candidates and computing cosine affinities..."):
            try:
                seed_talk, recommendations = service.recommend_from_talk(
                    selected_talk_id,
                    top_k=top_k,
                    method=method_key,
                )
                
                st.markdown("### 📌 Selected Seed Talk")
                render_talk_card(seed_talk, is_seed=True)
                
                st.markdown(f"### 🎯 Top {len(recommendations)} Recommendations ({available_methods[method_key]})")
                for rank, rec in enumerate(recommendations, 1):
                    render_talk_card(rec, is_seed=False, rank=rank)
                    
            except Exception as e:
                st.error(f"Error computing recommendations: {e}")


# ==============================================================================
# MODE 2: Topic Tags Discovery
# ==============================================================================
elif navigation == "🏷️ Topic Tags Discovery":
    st.markdown("<div class='main-title'>🏷️ Discover Talks by Topic Tags</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='sub-title'>Explore curated TED subjects by selecting multi-dimensional tag filters.</div>",
        unsafe_allow_html=True,
    )
    
    # Extract unique sorted tags
    all_tags = sorted(list(set([t for tags in catalog["tags_text"].dropna() for t in tags.split()])))
    # Clean tags for display
    clean_tags_map = {t: t.replace("_", " ").title() for t in all_tags}
    
    col_tags, col_tags_ctrl = st.columns([3, 1])
    with col_tags:
        selected_raw_tags = st.multiselect(
            "Select one or more topic tags:",
            options=all_tags,
            default=["education", "creativity"] if "education" in all_tags else all_tags[:2],
            format_func=lambda t: clean_tags_map.get(t, t),
        )
    with col_tags_ctrl:
        top_k_tags = st.slider("Number of Talks:", min_value=3, max_value=25, value=5)
        blend_pop = st.checkbox("Prioritize High-Engagement Talks", value=True)
        
    if st.button("🔍 Filter & Rank by Tags", type="primary", use_container_width=True):
        if not selected_raw_tags:
            st.warning("Please select at least one topic tag.")
        else:
            with st.spinner("Filtering candidate space and computing tag overlap..."):
                recs = service.recommend_by_tags(
                    selected_tags=selected_raw_tags,
                    top_k=top_k_tags,
                    blend_popularity=blend_pop,
                )
                if not recs:
                    st.info("No talks found matching the selected tag combinations.")
                else:
                    st.success(f"Retrieved {len(recs)} talks matching: {', '.join([clean_tags_map[t] for t in selected_raw_tags])}")
                    for rank, rec in enumerate(recs, 1):
                        render_talk_card(rec, is_seed=False, rank=rank)


# ==============================================================================
# MODE 3: Semantic Natural-Language Search
# ==============================================================================
elif navigation == "🔍 Semantic Natural-Language Search":
    st.markdown("<div class='main-title'>🔍 Semantic Natural-Language Search</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='sub-title'>Type any inquiry, curiosity, or problem description in plain English. Dense Sentence-BERT vectors retrieve conceptually aligned talks.</div>",
        unsafe_allow_html=True,
    )
    
    # Example prompts
    example_prompts = [
        "How can I overcome procrastination, build better habits, and stay focused?",
        "What are the ethical implications and future trajectory of Artificial Intelligence?",
        "How does human memory work, and can we reverse neurodegenerative diseases?",
        "Inspiring stories about resilience, overcoming failure, and discovering purpose",
    ]
    
    selected_example = st.selectbox("Or choose an example prompt:", ["-- Custom Input --"] + example_prompts)
    
    default_text = "" if selected_example == "-- Custom Input --" else selected_example
    user_query = st.text_area("Your Question / Area of Interest:", value=default_text, height=100)
    
    col_q1, col_q2 = st.columns([2, 1])
    with col_q1:
        search_engine = st.radio(
            "Search Engine:",
            options=["semantic", "keyword"],
            format_func=lambda x: "Sentence-BERT (Contextual Embedding Proximity)" if x == "semantic" else "TF-IDF (Lexical Keyword Matching)",
            horizontal=True,
        )
    with col_q2:
        top_k_query = st.slider("Result Count:", min_value=3, max_value=20, value=5)
        
    if st.button("🔎 Search Catalog", type="primary", use_container_width=True):
        if not user_query.strip():
            st.warning("Please enter a search query.")
        else:
            with st.spinner("Encoding query vector and scanning candidate manifold..."):
                results = service.recommend_by_query(user_query, top_k=top_k_query, engine=search_engine)
                if not results:
                    st.info("No matching talks found for your search terms.")
                else:
                    st.markdown(f"### 🌟 Top {len(results)} Semantic Matches")
                    for rank, rec in enumerate(results, 1):
                        render_talk_card(rec, is_seed=False, rank=rank)


# ==============================================================================
# MODE 4: Personalized User Profile
# ==============================================================================
elif navigation == "👤 Personalized User Profile":
    st.markdown("<div class='main-title'>👤 Personalized User Preference Profile</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='sub-title'>Build an interactive interest profile by declaring favorite topics, liked talks, and discovery preferences.</div>",
        unsafe_allow_html=True,
    )
    
    col_p1, col_p2 = st.columns(2)
    with col_p1:
        st.markdown("#### 1. Topics of Interest")
        popular_topics = ["science", "technology", "culture", "design", "business", "global issues", "health", "psychology", "education", "creativity"]
        user_topics = st.multiselect("Select themes you enjoy:", options=popular_topics, default=["science", "technology"])
        
    with col_p2:
        st.markdown("#### 2. Talks You Loved")
        liked_samples = catalog.sort_values(by="views", ascending=False).head(25)
        liked_options = {int(r["talk_id"]): f"{r['title']} — {r['speaker_1']}" for _, r in liked_samples.iterrows()}
        liked_tids = st.multiselect(
            "Select talks you previously watched & enjoyed:",
            options=list(liked_options.keys()),
            format_func=lambda tid: liked_options[tid],
            default=[66] if 66 in liked_options else [],
        )
        
    st.markdown("#### 3. Exploration Balance")
    niche_balance = st.slider(
        "Catalog Reach Bias (Niche Gems vs. Viral Talks):",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.1,
        help="0.0 favors focused, under-the-radar talks; 1.0 favors global viral flagship talks.",
    )
    
    if st.button("✨ Generate My Personalized Feed", type="primary", use_container_width=True):
        with st.spinner("Synthesizing user preference centroid vector..."):
            profile_recs = service.recommend_by_profile(
                preferred_topics=user_topics,
                liked_talk_ids=liked_tids,
                top_k=6,
                niche_vs_popular=niche_balance,
            )
            st.markdown("### 🎁 Your Personalized Recommendations")
            for rank, rec in enumerate(profile_recs, 1):
                render_talk_card(rec, is_seed=False, rank=rank)


# ==============================================================================
# MODE 5: Collaborative Filtering Benchmark Telemetry
# ==============================================================================
elif navigation == "📊 Collaborative Filtering Benchmark":
    st.markdown("<div class='main-title'>📊 Collaborative Filtering Telemetry Explorer</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='sub-title'>Explore TruncatedSVD Matrix Factorization recommendations for synthetic user telemetry cohorts (Milestone 2 benchmark).</div>",
        unsafe_allow_html=True,
    )
    
    st.info(
        "**Transparent Telemetry Disclosure:** The raw TED dataset lacks user login sessions and rating history. "
        "As established in Milestone 2 (Section 16 & 17), this collaborative filtering model was trained on 36,557 simulated user interaction records "
        "across 1,000 synthetic persona cohorts to validate the matrix factorization architecture."
    )
    
    col_u1, col_u2 = st.columns([1, 2])
    with col_u1:
        user_id_input = st.number_input("Benchmark User ID (0 - 999):", min_value=0, max_value=999, value=0, step=1)
        top_k_cf = st.slider("Recommendations:", min_value=3, max_value=15, value=5)
        
    if st.button("⚡ Query Matrix Factorization Engine", type="primary"):
        try:
            meta, cf_recs = service.get_cf_benchmark_user_recommendations(user_id=int(user_id_input), top_k=top_k_cf)
            with col_u2:
                st.markdown(f"**Cohort Persona:** `{meta['persona']}`")
                st.markdown(f"**Historical Talks Watched:** `{meta['watched_count']}`")
                if meta["history_sample"]:
                    st.caption("Sample History: " + ", ".join([h["title"] for h in meta["history_sample"][:3]]))
                    
            st.markdown(f"### 💡 Predicted Recommendations for User #{user_id_input}")
            for rank, rec in enumerate(cf_recs, 1):
                render_talk_card(rec, is_seed=False, rank=rank)
        except Exception as e:
            st.error(f"Error executing SVD collaborative filtering: {e}")


# ==============================================================================
# MODE 6: Architecture Benchmarks & Metrics
# ==============================================================================
elif navigation == "📈 Architecture Benchmarks & Metrics":
    st.markdown("<div class='main-title'>📈 End-to-End Model Benchmarks</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='sub-title'>Rigorous quantitative evaluation of all 6 candidate architectures evaluated against 250 ground-truth talks.</div>",
        unsafe_allow_html=True,
    )
    
    # Official Milestone 2 Table
    benchmark_df = pd.DataFrame({
        "Architecture": [
            "Multi-Signal Hybrid Engine (Leader)",
            "TF-IDF Content Baseline",
            "Sentence-BERT (all-MiniLM-L6-v2)",
            "Topic Tags Only",
            "Latent Semantic Analysis (TruncatedSVD)",
            "Word2Vec (TF-IDF Weighted)",
        ],
        "Precision@5": [0.1848, 0.1664, 0.1416, 0.1288, 0.1240, 0.0152],
        "Recall@5": [0.1551, 0.1389, 0.1192, 0.1082, 0.1040, 0.0127],
        "MRR": [0.4341, 0.3871, 0.3764, 0.2779, 0.2863, 0.0332],
        "NDCG@5": [0.2111, 0.1903, 0.1702, 0.1387, 0.1393, 0.0148],
        "Latency": ["< 4 ms", "< 1 ms", "< 2 ms", "< 1 ms", "< 1 ms", "< 1 ms"],
    })
    
    st.dataframe(
        benchmark_df.style.highlight_max(subset=["Precision@5", "Recall@5", "MRR", "NDCG@5"], color="#4CAF50"),
        use_container_width=True,
    )
    
    col_img1, col_img2 = st.columns(2)
    vis_dir = PROJECT_ROOT / "assets" / "visualizations"
    if (vis_dir / "benchmark_comparison.png").exists():
        with col_img1:
            st.image(str(vis_dir / "benchmark_comparison.png"), caption="Offline Recommendation Benchmark across 250 Evaluation Cohort Talks")
    if (vis_dir / "hybrid_weights.png").exists():
        with col_img2:
            st.image(str(vis_dir / "hybrid_weights.png"), caption="Multi-Signal Hybrid Engine Component Breakdown")


# ==============================================================================
# MODE 7: Model Card & Telemetry Info
# ==============================================================================
elif navigation == "ℹ️ Model Card & Telemetry Info":
    st.markdown("<div class='main-title'>ℹ️ Recommendation Model Card</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='sub-title'>Technical specifications, intended usage, and architectural parameters.</div>",
        unsafe_allow_html=True,
    )
    
    st.markdown(
        """
        ### 1. Model Overview
        - **System Name:** TED Talks Multi-Modal Recommendation Engine
        - **Version:** 2.0 (Production Streamlit Deployment)
        - **Primary Architecture:** Multi-Signal Linear Score Hybrid Engine
        - **Embedding Backends:**
            - **Sentence-BERT:** `sentence-transformers/all-MiniLM-L6-v2` (384 dimensions, normalized cosine space)
            - **TF-IDF Content:** 10,000 sublinear unigram/bigram features across speech transcripts, descriptions, and titles
            - **TF-IDF Tags:** 464 topic tags from TED's taxonomy
            - **Popularity Prior:** Log-scaled viewership normalized on $[0, 1]$
            - **Collaborative Filtering:** TruncatedSVD Matrix Factorization (30 latent factors)
            
        ### 2. Intended Use
        - Curated talk discovery for learners, researchers, educators, and general audiences.
        - Multi-modal exploratory navigation via direct seed talks, natural-language semantic prompts, topic filters, and personal interest profiles.
        
        ### 3. Evaluation Protocol
        - **Cohort:** 250 diverse talks with verified ground truth from TED's editorial `related_talks` field.
        - **Ranking Metrics:** Precision@5, Recall@5, Mean Reciprocal Rank (MRR), and Normalized Discounted Cumulative Gain (NDCG@5).
        - **Performance:** Multi-Signal Hybrid demonstrates a **+11.1% Precision gain** and **+12.1% MRR gain** over the best single-feature baseline.
        
        ### 4. Limitations & Ethical Considerations
        - **Cold-Start:** New talks without transcripts or tags rely on editorial title and description matching.
        - **Language Corpus:** Optimized primarily for English transcripts (98.8% of corpus).
        - **Popularity Bias:** Popularity prior is deliberately bounded to $\delta=0.10$ to prevent viral talks from suffocating niche presentations.
        """
    )

