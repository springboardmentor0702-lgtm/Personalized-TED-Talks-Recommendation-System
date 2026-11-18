
from pathlib import Path
import ast
import re

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from scipy.sparse import load_npz
from sklearn.metrics.pairwise import cosine_similarity


# --------------------------------------------------
# 1. PAGE CONFIGURATION
# --------------------------------------------------
st.set_page_config(
    page_title="Personalized TED Talks Recommendation System",
    page_icon="🎙️",
    layout="wide"
)

# ---------- Custom UI Styling ----------
st.markdown("""
<style>
/* Main background */
.stApp {
    background-color: #F8FAFD;
    color: #263238;
}

.block-container {
    max-width: 1200px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

/* Headings */
h1 {
    color: #D92332 !important;
    font-weight: 800 !important;
}

h2, h3 {
    color: #344054 !important;
    font-weight: 700 !important;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: #F0F4FA;
    border-right: 1px solid #E1E8F0;
}

/* Input fields and dropdowns */
div[data-baseweb="select"] > div {
    background-color: #FFFFFF;
    border: 1px solid #D9E2EF;
    border-radius: 10px;
}

/* Buttons */
.stButton > button {
    background-color: #FCE8E8;
    color: #B42332;
    border: 1px solid #F4C7CC;
    border-radius: 10px;
    padding: 0.55rem 1rem;
    font-weight: 600;
    transition: all 0.2s ease;
}

.stButton > button:hover {
    background-color: #F8D4D8;
    color: #912018;
    border-color: #EBAAB2;
}

/* Recommendation cards and bordered boxes */
div[data-testid="stVerticalBlockBorderWrapper"] {
    background-color: #FFFFFF;
    border: 1px solid #E4EAF2;
    border-radius: 14px;
    padding: 12px;
    box-shadow: 0 3px 10px rgba(38, 50, 56, 0.04);
}

/* Metrics */
div[data-testid="stMetric"] {
    background-color: #EDF6FF;
    border: 1px solid #D7E9FA;
    border-radius: 12px;
    padding: 16px;
}

/* Keep URLs blue */
a, a:link, a:visited {
    color: #1769AA !important;
    text-decoration: none !important;
    font-weight: 600;
}

a:hover {
    color: #0B4F85 !important;
    text-decoration: underline !important;
}

/* Radio buttons and checkboxes */
div[data-testid="stRadio"] label,
div[data-testid="stCheckbox"] label {
    color: #344054;
}

/* Info, success, and warning messages */
div[data-testid="stAlert"] {
    border-radius: 12px;
    border: 1px solid #E2E8F0;
}

/* Separators */
hr {
    border-color: #E4EAF2;
}
</style>
""", unsafe_allow_html=True)


BASE_DIR = Path(__file__).resolve().parent

METADATA_PATH = BASE_DIR / "talk_metadata.csv"
VECTORIZER_PATH = BASE_DIR / "tfidf_vectorizer.joblib"
MATRIX_PATH = BASE_DIR / "tfidf_matrix.npz"


# --------------------------------------------------
# 2. HELPER FUNCTIONS
# --------------------------------------------------
def find_column(dataframe, possible_names):
    """Find a column using possible column-name variations."""
    lookup = {
        str(column).strip().lower().replace(" ", "_"): column
        for column in dataframe.columns
    }

    for name in possible_names:
        key = name.strip().lower().replace(" ", "_")
        if key in lookup:
            return lookup[key]

    return None


def display_value(value, default="Not available"):
    """Convert missing values into readable text."""
    if value is None:
        return default

    try:
        if pd.isna(value):
            return default
    except (TypeError, ValueError):
        pass

    value = str(value).strip()

    if not value or value.lower() in {"nan", "none", "null"}:
        return default

    return value


def parse_topics(value):
    """Convert topics into a consistent list of strings."""
    if isinstance(value, (list, tuple, set)):
        return [str(item).strip() for item in value if str(item).strip()]

    value = display_value(value, "")

    if not value:
        return []

    try:
        parsed = ast.literal_eval(value)

        if isinstance(parsed, (list, tuple, set)):
            return [
                str(item).strip()
                for item in parsed
                if str(item).strip()
            ]
    except (ValueError, SyntaxError):
        pass

    # Support comma-separated or semicolon-separated topics.
    return [
        item.strip().strip("'\"[]")
        for item in re.split(r"[,;]", value)
        if item.strip().strip("'\"[]")
    ]


def format_views(value):
    """Format view counts for display."""
    try:
        number = float(value)

        if np.isfinite(number):
            return f"{number:,.0f}"

    except (TypeError, ValueError):
        pass

    return "Not available"


def format_duration(value):
    """Display duration in a readable format."""
    try:
        seconds = float(value)

        if np.isfinite(seconds) and seconds >= 0:
            minutes, remaining_seconds = divmod(int(seconds), 60)
            return f"{minutes} min {remaining_seconds} sec"

    except (TypeError, ValueError):
        pass

    return display_value(value)


# --------------------------------------------------
# 3. LOAD SAVED ARTIFACTS
# --------------------------------------------------
@st.cache_resource
def load_artifacts():
    required_files = [
        METADATA_PATH,
        VECTORIZER_PATH,
        MATRIX_PATH
    ]

    missing_files = [
        str(path.name)
        for path in required_files
        if not path.exists()
    ]

    if missing_files:
        raise FileNotFoundError(
            "Missing required files: " + ", ".join(missing_files)
        )

    metadata = pd.read_csv(METADATA_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    matrix = load_npz(MATRIX_PATH).tocsr()

    if len(metadata) != matrix.shape[0]:
        raise ValueError(
            f"Metadata contains {len(metadata)} rows, but the "
            f"TF-IDF matrix contains {matrix.shape[0]} rows. "
            "Use matching artifacts."
        )

    if matrix.shape[1] != len(vectorizer.get_feature_names_out()):
        raise ValueError(
            f"Matrix has {matrix.shape[1]} features, but the vectorizer "
            f"has {len(vectorizer.get_feature_names_out())} features. "
            "Use the matching vectorizer and matrix."
        )

    return metadata, vectorizer, matrix


# --------------------------------------------------
# 4. MAIN APPLICATION
# --------------------------------------------------
st.title("🎙️Personlized TED Talks Recommendation System")

st.write(
    "Discover TED Talks related to your interests using "
    "TF-IDF text features and cosine similarity."
)

try:
    df, vectorizer, tfidf_matrix = load_artifacts()

except Exception as error:
    st.error(f"Unable to load the project files: {error}")
    st.info(
        "Check that talk_metadata.csv, tfidf_vectorizer.joblib, "
        "and tfidf_matrix.npz are in the same folder as app.py."
    )
    st.stop()


# Identify metadata columns.
title_col = find_column(df, ["title", "talk_title", "name"])
speaker_col = find_column(
    df, ["speaker_1", "speaker", "speakers", "author"]
)
topics_col = find_column(df, ["topics", "topic", "tags"])
description_col = find_column(
    df, ["description", "summary", "about"]
)
url_col = find_column(df, ["url", "link", "talk_url"])
views_col = find_column(
    df, ["views", "view_count", "num_views", "view"]
)
duration_col = find_column(
    df, ["duration", "duration_seconds", "runtime"]
)

if title_col is None:
    st.error(
        "A title column was not found in talk_metadata.csv. "
        f"Available columns: {', '.join(map(str, df.columns))}"
    )
    st.stop()

# Standardize titles for display and lookup.
df = df.reset_index(drop=True)
df["_display_title"] = df[title_col].apply(
    lambda value: display_value(value, "Untitled Talk")
)

if df["_display_title"].duplicated().any():
    st.warning(
        "Some titles are duplicated. Recommendations will use the "
        "first matching occurrence when selecting a title."
    )

st.caption(
    f"Loaded {len(df):,} TED Talks | "
    f"{tfidf_matrix.shape[1]:,} TF-IDF features"
)

st.divider()


# --------------------------------------------------
# 5. SEARCH AND RECOMMENDATION SETTINGS
# --------------------------------------------------
st.subheader("Find a TED Talk")

col1, col2 = st.columns([3, 1])

with col1:
    selected_title = st.selectbox(
        "Search or select a TED Talk",
        options=df["_display_title"].tolist(),
        index=None,
        placeholder="Start typing a TED Talk title..."
    )

with col2:
    number_of_recommendations = st.radio(
        "Number of recommendations",
        options=[5, 10],
        index=0,
        horizontal=True
    )

# Topic options.
all_topics = []

if topics_col is not None:
    for topic_value in df[topics_col].dropna():
        all_topics.extend(parse_topics(topic_value))

unique_topics = sorted(
    set(topic for topic in all_topics if topic),
    key=str.lower
)

filter_col, sort_col = st.columns(2)

with filter_col:
    selected_topic = st.selectbox(
        "Filter by topic (optional)",
        options=["All topics"] + unique_topics,
        index=0
    )

with sort_col:
    sort_option = st.selectbox(
        "Sort recommendations by",
        options=[
            "Similarity (highest first)",
            "Views (highest first)",
            "Duration (shortest first)"
        ]
    )


# --------------------------------------------------
# 6. DISPLAY SELECTED TALK
# --------------------------------------------------
if selected_title is None:
    st.info(
        "Select a TED Talk above to discover similar talks."
    )
    st.stop()

matches = df.index[
    df["_display_title"] == selected_title
].tolist()

if not matches:
    st.error("The selected talk could not be found.")
    st.stop()

selected_index = matches[0]
selected_row = df.iloc[selected_index]

st.divider()
st.subheader("Selected TED Talk")

st.markdown(f"### {selected_title}")

if speaker_col is not None:
    st.write(
        "**Speaker:**",
        display_value(selected_row[speaker_col])
    )

if topics_col is not None:
    selected_topics = parse_topics(selected_row[topics_col])

    st.write(
        "**Topics:**",
        ", ".join(selected_topics) if selected_topics else "Not available"
    )

if description_col is not None:
    st.write(
        "**Description:**",
        display_value(selected_row[description_col])
    )

if views_col is not None:
    st.write(
        "**Views:**",
        format_views(selected_row[views_col])
    )

if duration_col is not None:
    st.write(
        "**Duration:**",
        format_duration(selected_row[duration_col])
    )

if url_col is not None:
    selected_url = display_value(selected_row[url_col], "")

    if selected_url.startswith(("https://", "http://")):
        st.markdown(f"[Watch this TED Talk]({selected_url})")

st.divider()


# --------------------------------------------------
# 7. CALCULATE SIMILARITY SCORES
# --------------------------------------------------
st.subheader("Recommended TED Talks")

with st.spinner("Finding similar TED Talks..."):
    selected_vector = tfidf_matrix[selected_index]

    similarity_scores = cosine_similarity(
        selected_vector,
        tfidf_matrix
    ).flatten()

# Exclude the selected talk itself.
recommendations = df.copy()
recommendations["_similarity"] = similarity_scores
recommendations = recommendations[
    recommendations.index != selected_index
].copy()

# Topic filter.
if selected_topic != "All topics" and topics_col is not None:
    recommendations = recommendations[
        recommendations[topics_col].apply(
            lambda value: selected_topic.lower()
            in [topic.lower() for topic in parse_topics(value)]
        )
    ]

# Sorting.
if sort_option == "Views (highest first)" and views_col is not None:
    recommendations["_sort_views"] = pd.to_numeric(
        recommendations[views_col],
        errors="coerce"
    )
    recommendations = recommendations.sort_values(
        ["_sort_views", "_similarity"],
        ascending=[False, False],
        na_position="last"
    )

elif (
    sort_option == "Duration (shortest first)"
    and duration_col is not None
):
    recommendations["_sort_duration"] = pd.to_numeric(
        recommendations[duration_col],
        errors="coerce"
    )
    recommendations = recommendations.sort_values(
        ["_sort_duration", "_similarity"],
        ascending=[True, False],
        na_position="last"
    )

else:
    recommendations = recommendations.sort_values(
        "_similarity",
        ascending=False
    )

recommendations = recommendations.head(
    number_of_recommendations
)

if recommendations.empty:
    st.warning(
        "No recommendations match this filter. "
        "Try selecting All topics."
    )
    st.stop()

st.write(
    f"Showing {len(recommendations)} recommendations "
    f"for **{selected_title}**."
)

# --------------------------------------------------
# 8. RECOMMENDATION CARDS
# --------------------------------------------------
for rank, (_, talk) in enumerate(
    recommendations.iterrows(),
    start=1
):
    with st.container(border=True):
        st.markdown(
            f"### {rank}. {talk['_display_title']}"
        )

        st.caption(
            f"Similarity score: {talk['_similarity']:.1%}"
        )

        if speaker_col is not None:
            st.write(
                "**Speaker:**",
                display_value(talk[speaker_col])
            )

        if topics_col is not None:
            talk_topics = parse_topics(talk[topics_col])

            st.write(
                "**Topics:**",
                ", ".join(talk_topics)
                if talk_topics
                else "Not available"
            )

        if description_col is not None:
            description = display_value(
                talk[description_col],
                "No description available."
            )

            st.write("**Description:**", description)

        metric_col1, metric_col2 = st.columns(2)

        with metric_col1:
            if views_col is not None:
                st.write(
                    "**Views:**",
                    format_views(talk[views_col])
                )

        with metric_col2:
            if duration_col is not None:
                st.write(
                    "**Duration:**",
                    format_duration(talk[duration_col])
                )

        if url_col is not None:
            talk_url = display_value(talk[url_col], "")

            if talk_url.startswith(("https://", "http://")):
                st.markdown(f"[Watch this TED Talk]({talk_url})")


# --------------------------------------------------
# 9. FOOTER
# --------------------------------------------------
st.divider()

st.caption(
    "Built with Python, Streamlit, TF-IDF and cosine similarity. "
    "Recommendations are based on textual similarity, not "
    "personal viewing history."
)
