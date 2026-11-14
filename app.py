
import streamlit as st
import pandas as pd
import joblib
import ast
import streamlit.components.v1 as components
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity


# --------------------------------------------------
# 1. PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="TED Talks Recommendation System",
    page_icon="🎤",
    layout="wide"
)


# --------------------------------------------------
# 2. TED-INSPIRED WHITE THEME
# --------------------------------------------------

st.markdown("""
<style>

/* Main application */
.stApp {
    background-color: #FFFFFF !important;
    color: #222222 !important;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1200px;
}

/* General text */
.stApp p,
.stApp li,
.stApp label {
    color: #333333;
}

/* TED logo */
.ted-logo {
    color: #E62B1E !important;
    font-size: 62px;
    font-weight: 900;
    letter-spacing: -3px;
    line-height: 1.2;
}

/* Tagline */
.ted-tagline {
    color: #555555 !important;
    font-size: 16px;
    margin-top: 5px;
    margin-bottom: 35px;
}

/* Main heading */
.ted-heading {
    color: #181818 !important;
    font-size: 40px;
    font-weight: 750;
    margin-bottom: 12px;
}

/* Description */
.ted-description {
    color: #555555 !important;
    font-size: 16px;
    line-height: 1.8;
    max-width: 750px;
    margin-bottom: 30px;
}

/* Section headings */
.stApp h1,
.stApp h2,
.stApp h3,
.stApp h4,
.stApp h5,
.stApp h6 {
    color: #181818 !important;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: #F7F7F7 !important;
    border-right: 1px solid #E5E5E5;
}

section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 {
    color: #E62B1E !important;
}

section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] span {
    color: #222222 !important;
}

/* Select boxes */
.stSelectbox div[data-baseweb="select"] > div {
    background-color: #FFFFFF !important;
    border: 1px solid #BBBBBB !important;
}

.stSelectbox div[data-baseweb="select"] * {
    color: #222222 !important;
}

/* Dropdown options */
div[data-baseweb="popover"],
div[data-baseweb="menu"] {
    background-color: #FFFFFF !important;
}

div[data-baseweb="menu"] * {
    color: #222222 !important;
}

/* Radio buttons */
.stRadio label,
.stRadio p,
.stRadio span {
    color: #222222 !important;
}

/* Metrics */
[data-testid="stMetric"] {
    background-color: #F8F8F8 !important;
    padding: 18px;
    border: 1px solid #E5E5E5;
    border-radius: 9px;
}

[data-testid="stMetricLabel"],
[data-testid="stMetricLabel"] * {
    color: #555555 !important;
}

[data-testid="stMetricValue"],
[data-testid="stMetricValue"] * {
    color: #E62B1E !important;
}

/* Recommendation cards */
div[data-testid="stVerticalBlockBorderWrapper"] {
    background-color: #FFFFFF !important;
    border: 1px solid #E2E2E2 !important;
    border-radius: 10px;
}

/* Links */
.stApp a {
    color: #D92318 !important;
    font-weight: 700;
    text-decoration: none;
}

.stApp a:hover {
    color: #A81710 !important;
    text-decoration: underline;
}

/* Buttons */
.stButton > button {
    background-color: #E62B1E !important;
    color: #FFFFFF !important;
    border: 1px solid #E62B1E !important;
    border-radius: 5px;
}

.stButton > button p {
    color: #FFFFFF !important;
}

.stButton > button:hover {
    background-color: #B91F15 !important;
    border-color: #B91F15 !important;
}

/* Input fields */
.stApp input {
    background-color: #FFFFFF !important;
    color: #222222 !important;
    border-color: #BBBBBB !important;
}

.stApp input::placeholder {
    color: #777777 !important;
}

/* Dividers */
hr {
    border-color: #E5E5E5 !important;
}

/* Alerts */
[data-testid="stAlert"] {
    background-color: #FFF8F7 !important;
    border: 1px solid #F2D2CE !important;
}

[data-testid="stAlert"] p {
    color: #333333 !important;
}

/* Video player */
iframe {
    border-radius: 10px;
}

/* Footer */
.ted-footer {
    text-align: center;
    color: #666666 !important;
    font-size: 13px;
    line-height: 1.8;
    padding: 25px 0;
    border-top: 1px solid #E5E5E5;
    margin-top: 40px;
}

</style>
""", unsafe_allow_html=True)


# --------------------------------------------------
# 3. PROJECT PATHS
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = BASE_DIR / "data" / "ted_model_dataset.csv"

MATRIX_PATH = BASE_DIR / "models" / "ted_tfidf_matrix.joblib"

VECTORIZER_PATH = BASE_DIR / "models" / "ted_tfidf_vectorizer.joblib"


# --------------------------------------------------
# 4. LOAD DATASET AND MODEL
# --------------------------------------------------

@st.cache_data
def load_data():

    return pd.read_csv(DATA_PATH)


@st.cache_resource
def load_model():

    tfidf_matrix = joblib.load(MATRIX_PATH)

    tfidf_vectorizer = joblib.load(VECTORIZER_PATH)

    return tfidf_matrix, tfidf_vectorizer


# Check required files

if not DATA_PATH.exists():
    st.error(f"Dataset not found: {DATA_PATH}")
    st.stop()

if not MATRIX_PATH.exists():
    st.error(f"TF-IDF matrix not found: {MATRIX_PATH}")
    st.stop()

if not VECTORIZER_PATH.exists():
    st.error(f"TF-IDF vectorizer not found: {VECTORIZER_PATH}")
    st.stop()


try:

    df = load_data()

    tfidf_matrix, tfidf_vectorizer = load_model()

except Exception as e:

    st.error(f"Error loading project files: {e}")
    st.stop()


# Check dataset and model compatibility

if len(df) != tfidf_matrix.shape[0]:

    st.error(
        "Dataset and TF-IDF matrix do not have matching rows. "
        "Please check your model files."
    )

    st.stop()


# --------------------------------------------------
# 5. DATA PREPARATION
# --------------------------------------------------

required_columns = [
    "title",
    "speaker_1",
    "topics",
    "views",
    "duration",
    "description",
    "url"
]

missing_columns = [
    col for col in required_columns if col not in df.columns
]

if missing_columns:

    st.error(f"Missing dataset columns: {missing_columns}")
    st.stop()


# Convert views and duration to numeric

df["views"] = pd.to_numeric(
    df["views"],
    errors="coerce"
).fillna(0)

df["duration"] = pd.to_numeric(
    df["duration"],
    errors="coerce"
).fillna(0)


# --------------------------------------------------
# 6. TOPIC EXTRACTION
# --------------------------------------------------

def get_topics(value):

    if isinstance(value, list):

        return [
            str(topic).strip()
            for topic in value
            if str(topic).strip()
        ]

    if pd.isna(value):

        return []

    try:

        parsed = ast.literal_eval(str(value))

        if isinstance(parsed, list):

            return [
                str(topic).strip()
                for topic in parsed
                if str(topic).strip()
            ]

    except (ValueError, SyntaxError):
        pass

    return [
        topic.strip()
        for topic in str(value).split(",")
        if topic.strip()
    ]


df["topics_list"] = df["topics"].apply(get_topics)


# --------------------------------------------------
# 7. DURATION FORMATTING
# --------------------------------------------------

def format_duration(seconds):

    seconds = int(seconds)

    minutes = seconds // 60

    remaining_seconds = seconds % 60

    return f"{minutes} min {remaining_seconds} sec"


# --------------------------------------------------
# 8. RECOMMENDATION ENGINE
# --------------------------------------------------

def recommend_talks(selected_index, top_n=5):

    try:

        similarity_scores = cosine_similarity(
            tfidf_matrix[selected_index],
            tfidf_matrix
        ).flatten()

        recommendations = df.copy()

        recommendations["similarity"] = similarity_scores

        # Exclude selected TED Talk

        recommendations = recommendations[
            recommendations.index != selected_index
        ]

        # Sort by similarity

        recommendations = recommendations.sort_values(
            by="similarity",
            ascending=False
        )

        return recommendations.head(top_n).copy()

    except Exception as e:

        st.error(f"Recommendation error: {e}")

        return pd.DataFrame()


# --------------------------------------------------
# 9. TED VIDEO EMBEDDING FUNCTION
# --------------------------------------------------

def display_ted_video(talk_url):

    if pd.isna(talk_url) or not str(talk_url).strip():

        st.info("Video link is not available for this TED Talk.")
        return

    talk_url = str(talk_url).strip()

    try:

        if "/talks/" not in talk_url:

            st.markdown(
                f"[Watch this TED Talk on TED.com]({talk_url})"
            )
            return

        talk_slug = (
            talk_url.split("/talks/")[-1]
            .split("?")[0]
            .strip("/")
        )

        embed_url = f"https://embed.ted.com/talks/{talk_slug}"

        st.markdown("### Watch TED Talk")

        components.iframe(
            embed_url,
            height=450,
            scrolling=False
        )

        st.caption(
            "TED video player. If embedding is unavailable, "
            "you can watch the talk on TED.com."
        )

        st.markdown(
            f"[Open original TED Talk]({talk_url})"
        )

    except Exception:

        st.warning(
            "Unable to display the embedded video. "
            "Please use the original TED Talk link."
        )

        st.markdown(
            f"[Watch this TED Talk]({talk_url})"
        )


# --------------------------------------------------
# 10. TED HEADER
# --------------------------------------------------

st.markdown("""
<div class="ted-logo">TED</div>

<div class="ted-tagline">
    Ideas worth spreading.
</div>

<div class="ted-heading">
    Discover Ideas That Matter
</div>

<div class="ted-description">
    Explore inspiring TED Talks and discover more talks
    based on your interests. Find ideas, perspectives,
    and knowledge worth sharing.
</div>
""", unsafe_allow_html=True)

st.divider()


# --------------------------------------------------
# 11. SIDEBAR SETTINGS
# --------------------------------------------------

st.sidebar.title("Recommendation Settings")

st.sidebar.markdown(
    "Customize how TED Talks are recommended."
)

top_n = st.sidebar.radio(
    "Number of recommendations",
    options=[5, 10],
    horizontal=True
)

sort_option = st.sidebar.selectbox(
    "Sort recommendations by",
    options=[
        "Similarity",
        "Most Viewed",
        "Shortest Duration",
        "Longest Duration"
    ]
)


# --------------------------------------------------
# 12. FIND A TED TALK
# --------------------------------------------------

st.subheader("Find a TED Talk")

talk_titles = df["title"].fillna(
    "Untitled TED Talk"
).tolist()

selected_title = st.selectbox(
    "Search or select a TED Talk",
    options=talk_titles,
    index=None,
    placeholder="Type to search for a TED Talk..."
)


# --------------------------------------------------
# 13. SELECTED TED TALK DETAILS
# --------------------------------------------------

if selected_title is not None:

    selected_index = df.index[
        df["title"].fillna("Untitled TED Talk") == selected_title
    ][0]

    selected_talk = df.loc[selected_index]

    st.divider()

    st.subheader("Selected TED Talk")

    # Display metrics

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Total Views",
            f"{int(selected_talk['views']):,}"
        )

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

    st.markdown(
        f"### {selected_talk['title']}"
    )

    st.markdown(
        f"**Speaker:** {selected_talk['speaker_1']}"
    )

    topics = selected_talk["topics_list"]

    st.markdown(
        "**Topics:** " + ", ".join(topics)
    )

    st.markdown("**Description:**")

    st.write(selected_talk["description"])

    # Display embedded TED video

    display_ted_video(selected_talk["url"])

    st.divider()


    # --------------------------------------------------
    # 14. RECOMMENDED TED TALKS
    # --------------------------------------------------

    st.subheader("Recommended TED Talks")

    recommendations = recommend_talks(
        selected_index,
        top_n
    )


    # --------------------------------------------------
    # 15. TOPIC FILTER
    # --------------------------------------------------

    all_topics = sorted({

        topic

        for topic_list in df["topics_list"]

        for topic in topic_list

    })

    selected_topic = st.selectbox(
        "Filter recommendations by topic",
        options=["All Topics"] + all_topics
    )


    if selected_topic != "All Topics":

        recommendations = recommendations[
            recommendations["topics_list"].apply(
                lambda topics: selected_topic in topics
            )
        ]


    # --------------------------------------------------
    # 16. SORT RECOMMENDATIONS
    # --------------------------------------------------

    if sort_option == "Similarity":

        recommendations = recommendations.sort_values(
            by="similarity",
            ascending=False
        )

    elif sort_option == "Most Viewed":

        recommendations = recommendations.sort_values(
            by="views",
            ascending=False
        )

    elif sort_option == "Shortest Duration":

        recommendations = recommendations.sort_values(
            by="duration",
            ascending=True
        )

    elif sort_option == "Longest Duration":

        recommendations = recommendations.sort_values(
            by="duration",
            ascending=False
        )


    # --------------------------------------------------
    # 17. DISPLAY RECOMMENDATIONS
    # --------------------------------------------------

    st.write(
        f"Found {len(recommendations)} recommended TED Talks."
    )

    if recommendations.empty:

        st.info(
            "No recommendations found for this topic. "
            "Try selecting another topic."
        )

    else:

        for rank, (_, talk) in enumerate(
            recommendations.iterrows(),
            start=1
        ):

            with st.container(border=True):

                st.markdown(
                    f"### {rank}. {talk['title']}"
                )

                st.markdown(
                    f"**Speaker:** {talk['speaker_1']}"
                )

                st.markdown(
                    "**Topics:** "
                    + ", ".join(talk["topics_list"])
                )

                st.markdown("**Description:**")

                st.write(talk["description"])

                col1, col2, col3 = st.columns(3)

                with col1:

                    st.metric(
                        "Similarity Score",
                        f"{talk['similarity'] * 100:.2f}%"
                    )

                with col2:

                    st.metric(
                        "Views",
                        f"{int(talk['views']):,}"
                    )

                with col3:

                    st.metric(
                        "Duration",
                        format_duration(talk["duration"])
                    )

                if pd.notna(talk["url"]):

                    st.markdown(
                        f"[Watch TED Talk on TED.com]({talk['url']})"
                    )


else:

    st.info(
        "Select a TED Talk above to discover similar TED Talks."
    )


# --------------------------------------------------
# 18. FOOTER
# --------------------------------------------------

st.markdown("""
<div class="ted-footer">
    Developed using Python, Streamlit, Pandas,
    Scikit-learn and TF-IDF.
    <br><br>
    Personalized TED Talks Recommendation System
</div>
""", unsafe_allow_html=True)
