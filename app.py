
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import ast
import html
from datetime import datetime

from pathlib import Path
from scipy.sparse import load_npz
from sklearn.metrics.pairwise import cosine_similarity


# ==========================================
# 1. PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="TED Talks Recommender",
    page_icon="🎤",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# ==========================================
# 2. CUSTOM DESIGN
# ==========================================

st.markdown("""
<style>

.stApp {
    background: #F6F8FA;
    color: #2B2D42;
}

.block-container {
    max-width: 1250px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

#MainMenu, footer {
    visibility: hidden;
}

header[data-testid="stHeader"] {
    background: transparent;
}

/* Hero banner */

.hero-title {
    color: #FFFFFF;
    font-size: 36px;
    font-weight: 800;
    text-align: center;
    line-height: 1.3;
}

.hero-subtitle {
    color: #E3EAF1;
    font-size: 16px;
    text-align: center;
    line-height: 1.7;
    margin-top: 12px;
}

/* Section headings */

.section-heading {
    color: #1D3557;
    font-size: 25px;
    font-weight: 800;
    margin-top: 22px;
    margin-bottom: 8px;
}

/* Statistics */

.stat-card {
    background: #EDF2F7;
    border: 1px solid #DCE4EC;
    border-left: 4px solid #457B9D;
    border-radius: 14px;
    padding: 19px 8px;
    text-align: center;
    margin: 12px 0 25px 0;
}

.stat-number {
    color: #1D3557;
    font-size: 26px;
    font-weight: 800;
}

.stat-label {
    color: #5C677D;
    font-size: 13px;
    margin-top: 4px;
}

/* Selected talk */

.selected-card {
    background: #E8EEF4;
    border: 1px solid #D7E0E9;
    border-radius: 14px;
    padding: 20px;
    margin: 12px 0 25px 0;
}

.selected-label {
    color: #5C677D;
    font-size: 11px;
    font-weight: 750;
    letter-spacing: 1px;
}

.selected-title {
    color: #1D3557;
    font-size: 21px;
    font-weight: 800;
    margin-top: 7px;
}

.selected-speaker {
    color: #5C677D;
    font-size: 14px;
    margin-top: 8px;
}

/* Recommendation cards */

div[data-testid="stVerticalBlockBorderWrapper"] {
    background: #FFFFFF;
    border: 1px solid #E0E5EA;
    border-top: 3px solid #A8C5D8;
    border-radius: 15px;
    padding: 10px;
    box-shadow: 0 3px 12px rgba(29, 53, 87, 0.05);
}

.card-number {
    color: #E07A68;
    font-size: 12px;
    font-weight: 750;
    letter-spacing: 1px;
}

.card-title {
    color: #1D3557;
    font-size: 18px;
    font-weight: 750;
    line-height: 1.45;
}

.card-speaker {
    color: #5C677D;
    font-size: 14px;
}

/* Expanders */

.streamlit-expanderHeader {
    color: #1D3557;
    font-weight: 650;
}

/* Buttons */

.stButton > button {
    background: #1D3557;
    color: #FFFFFF;
    border: none;
    border-radius: 10px;
    min-height: 46px;
    font-weight: 700;
}

.stButton > button:hover {
    background: #29496F;
    color: #FFFFFF;
    border: none;
}

/* TED links */

.stLinkButton > a {
    background: #1D3557;
    color: #FFFFFF;
    border: none;
    border-radius: 9px;
    font-weight: 650;
}

/* Dropdown */

div[data-baseweb="select"] > div {
    background: #FFFFFF;
    border-radius: 9px;
    border-color: #CFD8E2;
    min-height: 46px;
}

/* Footer */

.footer-text {
    text-align: center;
    color: #7A8492;
    font-size: 13px;
    padding: 25px 0 10px 0;
}

</style>
""", unsafe_allow_html=True)


# ==========================================
# 3. PROJECT FILE PATHS
# ==========================================

BASE_DIR = Path(__file__).parent

MODEL_DIR = BASE_DIR / "module4_outputs"

VECTOR_PATH = MODEL_DIR / "tfidf_vectorizer.joblib"
MATRIX_PATH = MODEL_DIR / "tfidf_matrix.npz"
METADATA_PATH = MODEL_DIR / "talk_metadata.csv"

DATASET_PATH = BASE_DIR / "ted_talks_en.csv"


# ==========================================
# 4. LOAD SAVED DATA
# ==========================================

@st.cache_resource
def load_project_data():

    vectorizer = joblib.load(VECTOR_PATH)

    matrix = load_npz(MATRIX_PATH)

    metadata = pd.read_csv(
        METADATA_PATH,
        dtype={"talk_id": str}
    )

    dataset = pd.read_csv(
        DATASET_PATH,
        dtype={"talk_id": str}
    )

    metadata["talk_id"] = metadata["talk_id"].str.strip()
    dataset["talk_id"] = dataset["talk_id"].str.strip()

    extra_columns = [
    "talk_id",
    "speaker_1",
    "description",
    "url",
    "topics",
    "views"
]

    columns_to_add = [
        column
        for column in extra_columns
        if column == "talk_id" or column not in metadata.columns
    ]

    extra_data = dataset[columns_to_add].drop_duplicates(
        subset="talk_id"
    )

    metadata = metadata.merge(
        extra_data,
        on="talk_id",
        how="left",
        sort=False
    )

    if len(metadata) != matrix.shape[0]:
        raise ValueError(
            "The metadata and recommendation matrix "
            "do not have matching row counts."
        )

    return vectorizer, matrix, metadata


try:

    vectorizer, tfidf_matrix, talks_df = load_project_data()

except Exception as error:

    st.error("Unable to load the TED Talks collection.")

    st.info(
        "Please check your dataset and Module 4 folder."
    )

    st.exception(error)
    st.stop()


# ==========================================
# 5. HELPER FUNCTIONS
# ==========================================

def valid_value(value):

    return (
        value is not None
        and not pd.isna(value)
        and str(value).strip() != ""
    )


def safe_text(value, default="Information not available"):

    if not valid_value(value):
        return default

    return html.escape(str(value))


def format_topics(value):

    if not valid_value(value):
        return []

    try:

        parsed = ast.literal_eval(str(value))

        if isinstance(parsed, (list, tuple, set)):
            return [str(item) for item in parsed]

    except (ValueError, SyntaxError):
        pass

    return [
        item.strip()
        for item in str(value).split(",")
        if item.strip()
    ]


# ==========================================
# 6. RECOMMENDATION FUNCTION
# ==========================================

def recommend_talks(talk_index, top_n=6):

    scores = cosine_similarity(
        tfidf_matrix[talk_index],
        tfidf_matrix
    ).ravel()

    scores[talk_index] = -1

    top_indices = np.argsort(scores)[::-1][:top_n]

    recommendations = talks_df.iloc[top_indices].copy()

    recommendations["similarity_score"] = scores[top_indices]

    return recommendations


# ==========================================
# 7. APPLICATION HEADER
# ==========================================

st.html("""
<div style="
    background:#1D3557;
    border-radius:18px;
    padding:38px 20px;
    text-align:center;
    margin-bottom:25px;
">

    <div style="
        color:white;
        font-size:36px;
        font-weight:800;
        line-height:1.3;
    ">
        🎤 Discover Ideas That Inspire.
    </div>

    <div style="
        color:#E3EAF1;
        font-size:16px;
        line-height:1.7;
        margin-top:12px;
    ">
        Explore fascinating talks. Find fresh perspectives.
        Get inspired.
    </div>

</div>
""")


# ==========================================
# 8. SIDEBAR PREFERENCES
# ==========================================

with st.sidebar:

    st.markdown(
        "<h3 style='color:#1D3557;'>Your Preferences</h3>",
        unsafe_allow_html=True
    )

    top_n = st.slider(
        "Recommendations to view",
        min_value=2,
        max_value=10,
        value=6,
        step=2
    )

    st.divider()

    st.metric(
        "Available TED Talks",
        f"{len(talks_df):,}"
    )

    st.caption(
        "Choose a talk you enjoy to discover similar ideas."
    )


# ==========================================
# 9. SEARCH SECTION
# ==========================================

search_col1, search_col2, search_col3 = st.columns(
    [1, 2, 1]
)

talk_options = {}

for index, row in talks_df.iterrows():

    title = str(row["title"])

    talk_id = str(row["talk_id"])

    label = f"{title} (ID: {talk_id})"

    talk_options[label] = index


with search_col2:

    st.markdown(
        """
        <h3 style="color:#1D3557;">
            🔍 Find your next favourite talk
        </h3>
        """,
        unsafe_allow_html=True
    )

    selected_label = st.selectbox(
        "Search or select a TED Talk",
        options=list(talk_options.keys()),
        index=None,
        placeholder="Type here to search talks...",
        label_visibility="collapsed"
    )

    recommend_btn = st.button(
        "✨ Discover Similar Talks",
        type="primary",
        use_container_width=True
    )


st.divider()


# ==========================================
# 10. EXPLORE TED TALKS BY TOPICS
# ==========================================

st.divider()

st.markdown(
    """
    <h3 style="color:#1D3557;">
        🌍 Explore TED Talks by Topics
    </h3>
    <p style="color:#5C677D;">
        Choose a topic that interests you and discover
        talks related to it.
    </p>
    """,
    unsafe_allow_html=True
)

all_topics = set()

if "topics" in talks_df.columns:

    for value in talks_df["topics"].dropna():

        all_topics.update(format_topics(value))

topic_options = sorted(all_topics)

selected_topics = st.multiselect(
    "Choose your favourite topics",
    options=topic_options,
    placeholder="Search and select topics..."
)

if selected_topics:

    matching_talks = talks_df[
        talks_df["topics"].apply(
            lambda value: any(
                topic in format_topics(value)
                for topic in selected_topics
            )
        )
    ]

    st.success(
        f"Found {len(matching_talks):,} talks "
        f"matching your selected topics."
    )

    if not matching_talks.empty:

        display_count = min(6, len(matching_talks))

        st.markdown("#### Talks you may enjoy")

        topic_columns = st.columns(2)

        for index, (_, talk) in enumerate(
            matching_talks.head(display_count).iterrows()
        ):

            with topic_columns[index % 2]:

                with st.container(border=True):

                    st.markdown(f"**{talk['title']}**")

                    speaker = talk.get("speaker_1")

                    if valid_value(speaker):
                        st.caption(f"🎙️ {speaker}")

                    topics = format_topics(
                        talk.get("topics")
                    )

                    if topics:
                        st.caption("🏷️ " + ", ".join(topics[:4]))

                    url = talk.get("url")

                    if (
                        valid_value(url)
                        and str(url).startswith(("http://", "https://"))
                    ):

                        st.link_button(
                            "Watch on TED.com ↗",
                            str(url),
                            use_container_width=True
                        )

    else:

        st.info("No matching talks were found.")

else:

    st.info(
        "Select one or more topics to explore related TED Talks."
    )

# ==========================================
# 11. INITIALIZE SESSION STATE
# ==========================================

if "favorite_ids" not in st.session_state:
    st.session_state.favorite_ids = set()

if "selected_index" not in st.session_state:
    st.session_state.selected_index = None

if "recommendations" not in st.session_state:
    st.session_state.recommendations = None

if "recommendation_history" not in st.session_state:
    st.session_state.recommendation_history = []


# ==========================================
# 12. GENERATE RECOMMENDATIONS
# ==========================================

if recommend_btn:

    if selected_label is None:

        st.warning(
            "Please select a TED Talk before continuing."
        )

    else:

        selected_index = talk_options[selected_label]

        recommendations = recommend_talks(
            selected_index,
            top_n
        )

        st.session_state.selected_index = selected_index

        st.session_state.recommendations = recommendations

        # Save recommendation history
        selected_talk = talks_df.iloc[selected_index]

        history_item = {
            "talk_id": str(selected_talk["talk_id"]),
            "title": str(selected_talk["title"]),
            "speaker": str(selected_talk.get("speaker_1", "")),
            "timestamp": datetime.now().strftime(
                "%d %b %Y, %I:%M %p"
            ),
            "recommendation_count": len(recommendations)
        }

        st.session_state.recommendation_history.insert(
            0,
            history_item
        )

        # Keep only the latest 20 searches
        st.session_state.recommendation_history = (
            st.session_state.recommendation_history[:20]
        )


# ==========================================
# 13. DISPLAY SELECTED TALK
# ==========================================

if st.session_state.selected_index is not None:

    selected_talk = talks_df.iloc[
        st.session_state.selected_index
    ]

    st.markdown(
        """
        <h3 style="color:#1D3557;">
            🎯 Your Selected Talk
        </h3>
        """,
        unsafe_allow_html=True
    )

    selected_title = safe_text(selected_talk["title"])

    selected_speaker = safe_text(
        selected_talk.get("speaker_1"),
        "Speaker information unavailable"
    )

    st.html(f"""
    <div class="selected-card">

        <div class="selected-label">
            YOUR SELECTION
        </div>

        <div class="selected-title">
            {selected_title}
        </div>

        <div class="selected-speaker">
            🎙️ {selected_speaker}
        </div>

    </div>
    """)


# ==========================================
# 14. DISPLAY RECOMMENDATIONS
# ==========================================

if st.session_state.recommendations is not None:

    recommendations = st.session_state.recommendations

    st.html("""
    <div style="margin-top:25px;margin-bottom:8px;">

        <div style="
            color:#1D3557;
            font-size:26px;
            font-weight:800;
        ">
            ✨ Handpicked Ideas for You
        </div>

        <div style="
            color:#718096;
            font-size:14px;
            margin-top:5px;
        ">
            Explore talks selected based on your interests
            and the talk you chose.
        </div>

    </div>
    """)

    grid_cols = st.columns(2)

    for index, (_, talk) in enumerate(
        recommendations.iterrows()
    ):

        col = grid_cols[index % 2]

        with col:

            with st.container(border=True):

                st.markdown(
                    f"""
                    <div class="card-number">
                        RECOMMENDATION {index + 1}
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                st.markdown(
                    f"""
                    <div class="card-title">
                        {safe_text(talk.get("title"))}
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                speaker = talk.get("speaker_1")

                if valid_value(speaker):

                    st.markdown(
                        f"""
                        <div class="card-speaker">
                            🎙️ {safe_text(speaker)}
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                score = float(talk["similarity_score"])

                score = max(0.0, min(1.0, score))

                st.progress(
                    score,
                    text=f"Content Match: {score:.1%}"
                )

                with st.expander("📚 Show Topics & Details"):

                    topics = format_topics(
                        talk.get("topics")
                    )

                    if topics:

                        st.markdown("**Topics:**")

                        st.write(" · ".join(topics))

                    else:

                        st.caption("Topics not available.")

                    description = talk.get("description")

                    if valid_value(description):

                        st.write(str(description))

                    else:

                        st.caption(
                            "Description not available."
                        )

                url = talk.get("url")

                if (
                    valid_value(url)
                    and str(url).startswith(
                        ("http://", "https://")
                    )
                ):

                    st.link_button(
                        "Watch on TED.com ↗",
                        str(url),
                        use_container_width=True
                    )

                else:

                    st.caption("TED Talk link unavailable.")

                # SAVE FAVOURITE BUTTON
                talk_id = str(talk["talk_id"])

                if talk_id in st.session_state.favorite_ids:

                    button_label = "♥ Remove from Favourites"

                else:

                    button_label = "♡ Add to Favourites"

                if st.button(
                    button_label,
                    key=f"favorite_{talk_id}",
                    use_container_width=True
                ):

                    if talk_id in st.session_state.favorite_ids:

                        st.session_state.favorite_ids.remove(
                            talk_id
                        )

                    else:

                        st.session_state.favorite_ids.add(
                            talk_id
                        )

                    st.rerun()


# ==========================================
# 15. WELCOME SECTION
# ==========================================

else:

    st.html("""
    <div style="
        background:#FFFFFF;
        border:1px solid #E0E5EA;
        border-radius:16px;
        padding:34px;
        text-align:center;
        margin-top:15px;
    ">

        <div style="font-size:43px;">
            💡
        </div>

        <div style="
            color:#1D3557;
            font-size:22px;
            font-weight:800;
            margin:12px 0;
        ">
            A world of ideas, all in one place.
        </div>

        <div style="
            color:#5C677D;
            font-size:14px;
            line-height:1.7;
        ">
            Choose a TED Talk you enjoy and discover
            more talks that match your interests.
        </div>

    </div>
    """)


# ==========================================
# 16. DISPLAY FAVOURITE TALKS
# ==========================================

st.divider()

st.markdown(
    "<h3 style='color:#1D3557;'>♥ Your Favourite Talks</h3>",
    unsafe_allow_html=True
)

if st.session_state.favorite_ids:

    favourites = talks_df[
        talks_df["talk_id"].astype(str).isin(
            st.session_state.favorite_ids
        )
    ]

    if not favourites.empty:

        for _, favourite_talk in favourites.iterrows():

            with st.container(border=True):

                st.markdown(
                    f"**{safe_text(favourite_talk['title'])}**"
                )

                speaker = favourite_talk.get("speaker_1")

                if valid_value(speaker):

                    st.caption(f"🎙️ {safe_text(speaker)}")

                topics = format_topics(
                    favourite_talk.get("topics")
                )

                if topics:

                    st.caption(
                        "🏷️ " + ", ".join(topics[:4])
                    )

                url = favourite_talk.get("url")

                if (
                    valid_value(url)
                    and str(url).startswith(
                        ("http://", "https://")
                    )
                ):

                    st.link_button(
                        "Watch on TED.com ↗",
                        str(url),
                        use_container_width=True
                    )

                favourite_id = str(
                    favourite_talk["talk_id"]
                )

                if st.button(
                    "♥ Remove from Favourites",
                    key=f"remove_favorite_{favourite_id}",
                    use_container_width=True
                ):

                    st.session_state.favorite_ids.remove(
                        favourite_id
                    )

                    st.rerun()

    else:

        st.info("Your saved talks could not be found.")

else:

    st.info(
        "You haven't saved any favourite talks yet. "
        "Explore recommendations and add your favourites!"
    )


# ==========================================
# 17. RECOMMENDATION HISTORY
# ==========================================

st.divider()

st.markdown(
    "<h3 style='color:#1D3557;'>🕘 Your Recommendation History</h3>",
    unsafe_allow_html=True
)

st.caption(
    "Your 20 most recent recommendation searches appear here."
)

if st.session_state.recommendation_history:

    for history_number, history_item in enumerate(
        st.session_state.recommendation_history
    ):

        with st.container(border=True):

            st.markdown(
                f"**{safe_text(history_item['title'])}**"
            )

            if valid_value(history_item.get("speaker")):

                st.caption(
                    f"🎙️ {safe_text(history_item['speaker'])}"
                )

            st.caption(
                f"🕒 {history_item['timestamp']} · "
                f"{history_item['recommendation_count']} recommendations"
            )

            if st.button(
                "View these recommendations",
                key=(
                    f"history_view_{history_number}_"
                    f"{history_item['talk_id']}"
                ),
                use_container_width=True
            ):

                matching_rows = talks_df.index[
                    talks_df["talk_id"].astype(str)
                    == str(history_item["talk_id"])
                ]

                if len(matching_rows) > 0:

                    previous_index = int(matching_rows[0])

                    st.session_state.selected_index = previous_index

                    st.session_state.recommendations = recommend_talks(
                        previous_index,
                        top_n
                    )

                    st.rerun()

                else:

                    st.warning(
                        "This talk is no longer available "
                        "in the collection."
                    )

else:

    st.info(
        "Your recommendation history is empty. "
        "Choose a TED Talk and discover similar talks to get started!"
    )


# ==========================================
# 18: POPULAR TED TALKS
# ==========================================

st.divider()

st.markdown(
    "<h3 style='color:#1D3557;'>🔥 Popular TED Talks</h3>",
    unsafe_allow_html=True
)

st.caption(
    "Discover some of the most-viewed talks in our collection."
)

if "views" in talks_df.columns:

    popular_talks = talks_df.copy()

    popular_talks["views"] = pd.to_numeric(
        popular_talks["views"],
        errors="coerce"
    )

    popular_talks = popular_talks.dropna(
        subset=["views"]
    )

    popular_talks = popular_talks.sort_values(
        by="views",
        ascending=False
    ).head(10)

    if not popular_talks.empty:

        popular_columns = st.columns(2)

        for index, (_, talk) in enumerate(
            popular_talks.iterrows()
        ):

            with popular_columns[index % 2]:

                with st.container(border=True):

                    st.markdown(
                        f"""
                        <div style="
                            color:#E07A68;
                            font-size:12px;
                            font-weight:750;
                        ">
                            POPULAR TALK #{index + 1}
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                    st.markdown(
                        f"**{safe_text(talk['title'])}**"
                    )

                    speaker = talk.get("speaker_1")

                    if valid_value(speaker):

                        st.caption(
                            f"🎙️ {safe_text(speaker)}"
                        )

                    views = int(talk["views"])

                    st.metric(
                        label="👁️ Total Views",
                        value=f"{views:,}"
                    )

                    url = talk.get("url")

                    if (
                        valid_value(url)
                        and str(url).startswith(
                            ("http://", "https://")
                        )
                    ):

                        st.link_button(
                            "Watch on TED.com ↗",
                            str(url),
                            use_container_width=True
                        )

    else:

        st.info(
            "View information is not available for these talks."
        )

else:

    st.warning(
        "The dataset does not contain a views column. "
        "Please check the original TED Talks dataset."
    )


# ==========================================
# 19. FOOTER
# ==========================================

st.html("""
<div style="
    margin-top:35px;
    border-top:1px solid #E1E6EB;
    padding-top:22px;
    padding-bottom:10px;
    text-align:center;
    color:#7A8492;
    font-size:13px;
">
    Discover ideas. Explore perspectives. Stay inspired. ✨
</div>
""")

