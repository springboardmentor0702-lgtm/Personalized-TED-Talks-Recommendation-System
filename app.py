
import streamlit as st
import pandas as pd
import joblib
import ast
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity

# --------------------------------------------------
# 1. PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="TED Talks Recommendation System",
    page_icon="",
    layout="wide"
)

# --------------------------------------------------
# 2. PROJECT PATHS
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = BASE_DIR / "data" / "ted_model_dataset.csv"
MATRIX_PATH = BASE_DIR / "models" / "ted_tfidf_matrix.joblib"
VECTORIZER_PATH = BASE_DIR / "models" / "ted_tfidf_vectorizer.joblib"


# --------------------------------------------------
# 3. LOAD DATA AND MODEL
# --------------------------------------------------

@st.cache_data
def load_dataset():
    return pd.read_csv(DATA_PATH)


@st.cache_resource
def load_model():
    matrix = joblib.load(MATRIX_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    return matrix, vectorizer


df = load_dataset()
tfidf_matrix, tfidf_vectorizer = load_model()


# --------------------------------------------------
# 4. DATA PREPARATION
# --------------------------------------------------

df = df.reset_index(drop=True)

if len(df) != tfidf_matrix.shape[0]:
    st.error("Dataset and TF-IDF matrix do not match.")
    st.stop()


def get_topics(value):

    if pd.isna(value):
        return []

    if isinstance(value, list):
        return value

    try:
        parsed = ast.literal_eval(str(value))

        if isinstance(parsed, list):
            return parsed

        if isinstance(parsed, str):
            return [parsed]

    except (ValueError, SyntaxError):
        pass

    return [topic.strip() for topic in str(value).split(",") if topic.strip()]


df["topics_list"] = df["topics"].apply(get_topics)

df["views"] = pd.to_numeric(
    df["views"], errors="coerce"
).fillna(0)

df["duration"] = pd.to_numeric(
    df["duration"], errors="coerce"
).fillna(0)

def format_duration(seconds):
    seconds = int(seconds)
    minutes, remaining_seconds = divmod(seconds, 60)

    return f"{minutes} min {remaining_seconds} sec"


# --------------------------------------------------
# 5. RECOMMENDATION FUNCTION
# --------------------------------------------------

def recommend_talks(selected_index, top_n=5):

    similarity_scores = cosine_similarity(
        tfidf_matrix[selected_index],
        tfidf_matrix
    ).flatten()

    # Exclude the selected talk
    similarity_scores[selected_index] = -1

    # Get ranking of similar talks
    ranked_indices = similarity_scores.argsort()[::-1]

    results = []

    for index in ranked_indices:

        if similarity_scores[index] <= 0:
            continue

        results.append({
            "index": index,
            "similarity": float(similarity_scores[index])
        })

        if len(results) == top_n:
            break

    return results


# --------------------------------------------------
# 6. APPLICATION HEADER
# --------------------------------------------------

st.title("Personalized TED Talks Recommendation System")

st.markdown("""
Discover TED Talks that match your interests!

This application uses **Machine Learning, TF-IDF Vectorization
and Cosine Similarity** to recommend talks based on their
titles, descriptions, topics and transcripts.
""")

st.divider()


# --------------------------------------------------
# 7. SIDEBAR OPTIONS
# --------------------------------------------------

st.sidebar.header("Recommendation Settings")

top_n = st.sidebar.radio(
    "Number of recommendations",
    options=[5, 10],
    index=0
)

sort_option = st.sidebar.selectbox(
    "Sort recommendations by",
    [
        "Similarity",
        "Most Viewed",
        "Shortest Duration",
        "Longest Duration"
    ]
)

# --------------------------------------------------
# 8. TALK SEARCH
# --------------------------------------------------

st.subheader("Find a TED Talk")

talk_titles = df["title"].fillna("").astype(str).tolist()

selected_title = st.selectbox(
    "Search or select a TED Talk",
    options=[""] + talk_titles,
    index=0,
    placeholder="Type a TED Talk title..."
)

if not selected_title:

    st.info("Select a TED Talk to discover similar talks.")

    st.stop()


# --------------------------------------------------
# 9. SELECTED TALK DETAILS
# --------------------------------------------------

selected_index = df.index[
    df["title"].fillna("").astype(str) == selected_title
][0]

selected_talk = df.iloc[selected_index]

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Total Views", f"{int(selected_talk['views']):,}")

with col2:
    st.metric(
        "Duration",
        format_duration(selected_talk["duration"])
    )

with col3:
    st.metric(
        "Number of Topics",
        len(selected_talk["topics_list"])
    )

st.subheader("Selected TED Talk")

st.markdown(f"### {selected_talk['title']}")

st.write(
    "**Speaker:**",
    selected_talk.get("speaker_1", "Not available")
)

st.write(
    "**Topics:**",
    ", ".join(map(str, selected_talk["topics_list"]))
)

st.write(
    "**Description:**",
    selected_talk.get("description", "Not available")
)

talk_url = selected_talk.get("url", "")

if pd.notna(talk_url) and str(talk_url).startswith("https://"):
    st.markdown(f"[Watch this TED Talk]({talk_url})")

st.divider()


# --------------------------------------------------
# 10. GENERATE RECOMMENDATIONS
# --------------------------------------------------

st.subheader("Recommended TED Talks")

recommendations = recommend_talks(
    selected_index,
    top_n=top_n
)

if not recommendations:

    st.warning("No similar TED Talks were found.")

    st.stop()


# Convert recommendations to DataFrame
recommendation_rows = []

for item in recommendations:

    talk = df.iloc[item["index"]].copy()

    talk["similarity"] = item["similarity"]

    recommendation_rows.append(talk)

recommendation_df = pd.DataFrame(recommendation_rows)


# --------------------------------------------------
# 11. TOPIC FILTER
# --------------------------------------------------

available_topics = sorted({
    str(topic)
    for topics in recommendation_df["topics_list"]
    for topic in topics
})

selected_topics = st.multiselect(
    "Filter recommendations by topic",
    options=available_topics
)

if selected_topics:

    recommendation_df = recommendation_df[
        recommendation_df["topics_list"].apply(
            lambda topics: any(
                topic in topics for topic in selected_topics
            )
        )
    ]


# --------------------------------------------------
# 12. SORT RECOMMENDATIONS
# --------------------------------------------------

if sort_option == "Most Viewed":

    recommendation_df = recommendation_df.sort_values(
        by="views", ascending=False
    )

elif sort_option == "Shortest Duration":

    recommendation_df = recommendation_df.sort_values(
        by="duration", ascending=True
    )

elif sort_option == "Longest Duration":

    recommendation_df = recommendation_df.sort_values(
        by="duration", ascending=False
    )


# --------------------------------------------------
# 13. DISPLAY RESULTS
# --------------------------------------------------

if recommendation_df.empty:

    st.warning("No recommendations match the selected topic.")

else:

    st.success(
        f"Found {len(recommendation_df)} recommended TED Talks."
    )

    for rank, (_, talk) in enumerate(
        recommendation_df.iterrows(), start=1
    ):

        with st.container(border=True):

            st.markdown(
                f"### {rank}. {talk['title']}"
            )

            st.write(
                "**Speaker:**",
                talk.get("speaker_1", "Not available")
            )

            st.write(
                "**Topics:**",
                ", ".join(map(str, talk["topics_list"]))
            )

            st.write(
                "**Description:**",
                talk.get("description", "Not available")
            )

            st.write(
                f"**Similarity Score:** {talk['similarity'] * 100:.2f}%"
            )

            st.write(
                f"**Views:** {int(talk['views']):,}"
            )

            st.write(
                f"**Duration:** {format_duration(talk['duration'])}"
            )

            url = talk.get("url", "")

            if pd.notna(url) and str(url).startswith("https://"):
                st.markdown(f"[Watch TED Talk]({url})")

st.divider()

st.caption(
    "Developed using Python, Streamlit, Pandas, Scikit-learn and TF-IDF."
)
