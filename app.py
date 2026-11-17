import ast
import html
import re
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from scipy.sparse import load_npz
from sklearn.metrics.pairwise import cosine_similarity
from urllib.parse import urlparse

#Page Configuration

st.set_page_config(
    page_title="TED Talks Recommendation System",
    page_icon="🎤",
    layout="wide"
)

# 2. CSS DESIGN

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background:
        radial-gradient(circle at 85% 5%, rgba(190,0,25,.13), transparent 25%),
        linear-gradient(135deg,#08090d 0%,#101116 55%,#090a0e 100%);
    color: #f5f5f7;
}

[data-testid="stHeader"] {
    background: transparent;
}

.block-container {
    padding-top: 1.5rem;
    padding-bottom: 2rem;
    max-width: 1450px;
}

/* CLICKABLE CURSOR */

button,
div[data-baseweb="select"],
div[data-baseweb="select"] *,
input[type="range"],
[role="slider"],
a {
    cursor: pointer !important;
}


/* SIDEBAR */

[data-testid="stSidebar"] {
    background: linear-gradient(180deg,#171a23 0%,#101218 100%);
    border-right: 1px solid #292c35;
}

[data-testid="stSidebar"] > div {
    padding-top: 1.5rem;
}

.brand {
    font-size: 46px;
    font-weight: 800;
    color: #f52b3b;
    letter-spacing: -3px;
}

.brand-sub {
    color: #f4f4f5;
    font-size: 25px;
    font-weight: 750;
    margin-top: 4px;
}

.muted {
    color: #b8bbc5;
}

.red-line {
    width: 45px;
    height: 3px;
    background: #f22c3d;
    margin: 17px 0 25px;
    border-radius: 5px;
}


/* HERO SECTION */

.hero {
    min-height: 220px;
    padding: 35px 40px;
    border-radius: 20px;
    border: 1px solid #30232b;
    background:
        radial-gradient(circle at 90% 20%, rgba(220,20,45,.18), transparent 35%),
        linear-gradient(120deg,#111217,#1a1017);
    box-shadow: 0 12px 35px rgba(0,0,0,.3);
    margin-bottom: 35px;
    display: flex;
    flex-direction: column;
    justify-content: center;
}

.hero-label {
    color: #b8bbc5;
    font-size: 11px;
    letter-spacing: 2px;
    font-weight: 600;
}

.hero-title {
    font-size: clamp(30px,3vw,43px);
    line-height: 1.2;
    color: white;
    font-weight: 800;
    margin: 14px 0;
}

.hero-title span {
    color: #f52b3b;
}

.hero-description {
    color: #c5c7d0;
    font-size: 16px;
    margin: 0;
}


/* SEARCH SECTION */

.search-heading {
    text-align: center;
    font-size: 23px;
    font-weight: 750;
    margin: 10px 0 18px;
}

div[data-testid="stTextInput"] input {
    background: #1b1d26 !important;
    color: white !important;
    border: 1px solid #48404a !important;
    border-radius: 10px !important;
    height: 46px !important;
}

div[data-testid="stTextInput"] input:focus {
    border-color: #f02b3b !important;
    box-shadow: 0 0 0 1px #f02b3b !important;
}

.search-button div[data-testid="stButton"] button {
    height: 46px !important;
    min-height: 46px !important;
    margin-top: 0 !important;
    border-radius: 10px !important;
    white-space: nowrap;
}


/* BUTTONS */

.stButton > button {
    background: linear-gradient(90deg,#f02b3b,#c20e25) !important;
    color: white !important;
    border: 0 !important;
    border-radius: 10px !important;
    font-weight: 700 !important;
    min-height: 41px;
    cursor: pointer !important;
}

.stButton > button:hover {
    filter: brightness(1.12);
}


/* DROPDOWNS */

div[data-baseweb="select"] > div {
    background: #1b1d26 !important;
    border-color: #3a3d49 !important;
    border-radius: 10px !important;
}

div[data-baseweb="select"] svg {
    fill: #ffffff !important;
    color: #ffffff !important;
    opacity: 1 !important;
    visibility: visible !important;
}

div[data-baseweb="select"] svg path {
    fill: #ffffff !important;
    stroke: #ffffff !important;
}

div[data-testid="stSelectbox"] label {
    color: #e7e7eb;
}


/* RECOMMENDATION CARDS */

.talk-card {
    background: linear-gradient(145deg,#1b1d25,#12141b);
    border: 1px solid #30323c;
    border-radius: 16px;
    padding: 17px;

    height: 290px;
    min-height: 290px;
    box-sizing: border-box;

    display: flex;
    flex-direction: column;

    overflow: hidden;

    box-shadow: 0 8px 22px rgba(0,0,0,.2);
}

.talk-card:hover {
    border-color: #7d2932;
    transform: translateY(-2px);
    transition: .18s ease;
}


/* CARD TITLE */

.talk-title {
    font-size: 16px;
    line-height: 1.4;
    font-weight: 750;
    color: white;
    margin-bottom: 8px;

    min-height: 44px;
    max-height: 44px;

    display: -webkit-box;
    -webkit-line-clamp: 2;
    -webkit-box-orient: vertical;

    overflow: hidden;
}


/* SPEAKER */

.speaker {
    font-size: 13px;
    color: #c7c9d1;
    margin-bottom: 10px;
}


/* TOPIC TAGS */

.tag {
    display: inline-block;
    padding: 5px 9px;
    margin: 2px 4px 2px 0;
    background: #292c36;
    border: 1px solid #373a46;
    color: #e2e3e8;
    border-radius: 20px;
    font-size: 11px;
}


/* DESCRIPTION */

.description {
    color: #b9bbc4;
    font-size: 12px;
    line-height: 1.5;

    margin-top: 10px;

    min-height: 54px;
    max-height: 54px;

    overflow: hidden;

    display: -webkit-box;
    -webkit-line-clamp: 3;
    -webkit-box-orient: vertical;
}


/* CARD FOOTER */

.card-footer {
    margin-top: auto;

    display: flex;
    flex-direction: column;
    align-items: flex-start;

    gap: 8px;
}


/* SIMILARITY SCORE */

.score {
    display: inline-block;
    padding: 6px 10px;
    border-radius: 9px;
    border: 1px solid #a52a37;
    color: #ff6874;
    font-size: 12px;
    font-weight: 700;
    background: rgba(220,20,45,.08);
}


/* WATCH BUTTON */

.watch-link {
    display: inline-block;
    background: linear-gradient(90deg,#f02b3b,#bd1024);
    color: white !important;
    text-decoration: none !important;
    padding: 9px 13px;
    border-radius: 9px;
    font-size: 12px;
    font-weight: 700;
    margin-top: 0;
}


/* SELECTED TALK */

.selected-card {
    background: linear-gradient(110deg,#1b1d25,#12141b);
    border: 1px solid #30323c;
    border-left: 4px solid #f12b3c;
    border-radius: 16px;
    padding: 20px;
    margin: 10px 0 25px;
    box-shadow: 0 10px 28px rgba(0,0,0,.2);
}


/* SECTION TITLE */

.section-title {
    font-size: 22px;
    font-weight: 750;
    margin: 25px 0 15px;
}

</style>
""", unsafe_allow_html=True)

# 3. HELPER FUNCTIONS

def safe_text(value, fallback="Not available"):

    if value is None:
        return fallback

    try:
        if pd.isna(value):
            return fallback
    except (TypeError, ValueError):
        pass

    value = str(value).strip()

    if not value or value.lower() in ["nan", "none"]:
        return fallback

    return value


def parse_topics(value):

    if isinstance(value, (list, tuple, set)):
        items = list(value)

    else:

        raw = safe_text(value, "")

        if not raw:
            return []

        try:
            parsed = ast.literal_eval(raw)

            if isinstance(parsed, (list, tuple, set)):
                items = list(parsed)
            else:
                items = [raw]

        except (ValueError, SyntaxError):
            items = re.split(r"[,;|]", raw)

    return [
        str(item).strip()
        for item in items
        if safe_text(item, "")
    ]

def get_value(row, columns, fallback="Not available"):

    for col in columns:

        if col in row.index:

            value = safe_text(row[col], "")

            if value:
                return value

    return fallback


def valid_url(value):

    raw = safe_text(value, "")

    if not raw:
        return ""

    if not re.match(r"^https?://", raw, re.I):
        raw = "https://" + raw.lstrip("/")

    parsed = urlparse(raw)

    if parsed.scheme in ["http", "https"] and parsed.netloc:
        return raw

    return ""


# 4. LOAD DATASET


@st.cache_data
def load_dataset():

    return pd.read_csv(
        "ted_talks_final.csv"
    ).reset_index(drop=True)


# 5. LOAD MODEL FILES

@st.cache_resource
def load_model():

    data = load_dataset()

    transcript_matrix = load_npz(
        "transcript_tfidf_matrix.npz"
    ).tocsr()

    tags_matrix = load_npz(
        "tags_tfidf_matrix.npz"
    ).tocsr()

    transcript_vectorizer = joblib.load(
        "transcript_tfidf_vectorizer.pkl"
    )

    tags_vectorizer = joblib.load(
        "tags_tfidf_vectorizer.pkl"
    )

    if (
        len(data) != transcript_matrix.shape[0]
        or len(data) != tags_matrix.shape[0]
    ):

        raise ValueError(
            "CSV and TF-IDF matrices have different row counts. "
            "Check that you are using the same dataset and row order."
        )

    return (
        data,
        transcript_matrix,
        tags_matrix,
        transcript_vectorizer,
        tags_vectorizer
    )


try:

    (
        df,
        transcript_matrix,
        tags_matrix,
        transcript_vectorizer,
        tags_vectorizer
    ) = load_model()

except Exception as error:

    st.error(f"Unable to load project files: {error}")
    st.stop()


if "title" not in df.columns:

    st.error("The dataset does not contain a title column.")
    st.stop()



# 6. SIDEBAR

with st.sidebar:

    st.markdown(
        '<div class="brand">TED</div>'
        '<div class="brand-sub">Talks</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<p class="muted">Discover Ideas<br>Worth Spreading</p>'
        '<div class="red-line"></div>',
        unsafe_allow_html=True
    )

    st.markdown("### ⚙ Filters")

    topic_column = (
        "topics" if "topics" in df.columns
        else "topic_list"
    )

    all_topics = []

    if topic_column in df.columns:

        all_topics = sorted({
            topic
            for value in df[topic_column].tolist()
            for topic in parse_topics(value)
        })

    selected_topic = st.selectbox(
        "Topic",
        ["All Topics"] + all_topics,
        key="topic_filter"
    )

    sort_option = st.selectbox(
        "Sort recommendations",
        [
            "Highest Similarity",
            "Most Viewed",
            "Shortest Duration"
        ],
        key="sort_filter"
    )

    result_count = st.select_slider(
        "Number of recommendations",
        options=[5, 6, 8, 10],
        value=6,
        key="count_filter"
    )

    st.markdown("---")

    st.markdown("### ⓘ About")

    st.markdown(
        '<p class="muted">'
        'A personalized TED Talks discovery experience '
        'powered by machine learning.'
        '</p>',
        unsafe_allow_html=True
    )

# 7. HERO SECTION

st.markdown("""
<div class="hero">
<div class="hero-label">IDEAS WORTH DISCOVERING</div>

<div class="hero-title">
Personalized <span>TED</span> Talks<br>
Recommendation System
</div>

<p class="hero-description">
Find your next inspiring talk with the power of machine learning.
</p>
</div>
""", unsafe_allow_html=True)

# 8. CENTERED SEARCH SECTION


st.markdown(
    '<div class="search-heading">Find a TED Talk</div>',
    unsafe_allow_html=True
)

left_space, search_area, right_space = st.columns([3, 4, 3])

with search_area:

    search_col, button_col = st.columns(
        [4, 1.35],
        gap="small",
        vertical_alignment="center"
    )

    with search_col:

        query = st.text_input(
            "Search TED Talks",
            placeholder="Search by TED Talk title...",
            label_visibility="collapsed",
            key="search_input"
        )

    with button_col:

        st.markdown(
            '<div class="search-button">',
            unsafe_allow_html=True
        )

        search_clicked = st.button(
            "Search →",
            use_container_width=True,
            key="search_button"
        )

        st.markdown(
            '</div>',
            unsafe_allow_html=True
        )

# 9. SEARCH FUNCTIONALITY

if search_clicked:

    st.session_state["searched_query"] = query.strip()

    st.session_state.pop("selected_index", None)

    st.session_state.pop("recommendation_scores", None)


searched_query = st.session_state.get(
    "searched_query",
    ""
)


if searched_query:

    matching_indices = [
        i
        for i, title in enumerate(
            df["title"].fillna("").astype(str)
        )
        if searched_query.lower() in title.lower()
    ]

    left_result, result_area, right_result = st.columns([3, 4, 3])

    with result_area:

        if matching_indices:

            st.success(
                f"Found {len(matching_indices)} matching TED Talks."
            )

            selected_index = st.selectbox(
                "Select a TED Talk",
                matching_indices,
                format_func=lambda i: str(df.iloc[i]["title"]),
                key="matched_talk_select"
            )

            if st.button(
                "Get Recommendations",
                key="recommend_button",
                use_container_width=True
            ):

                transcript_scores = cosine_similarity(
                    transcript_matrix[selected_index],
                    transcript_matrix
                ).ravel()

                topic_scores = cosine_similarity(
                    tags_matrix[selected_index],
                    tags_matrix
                ).ravel()

                # 70% transcript similarity
                # 30% topic similarity

                final_scores = (
                    0.70 * transcript_scores
                    + 0.30 * topic_scores
                )

                final_scores[selected_index] = -1

                st.session_state["selected_index"] = int(
                    selected_index
                )

                st.session_state["recommendation_scores"] = (
                    final_scores
                )

        else:

            st.warning(
                "No TED Talk found. Please try another title."
            )


# 10. RECOMMENDATION CARD FUNCTION


def render_talk_card(row, score=None, selected=False):

    title = html.escape(
        get_value(row, ["title"], "Untitled TED Talk")
    )

    speaker = html.escape(
        get_value(
            row,
            ["speaker_1", "speaker", "all_speakers"],
            "Speaker not listed"
        )
    )

    description = html.escape(
        get_value(row, ["description"], "No description available.")
    )

    if len(description) > 175:
        description = description[:172].rsplit(" ", 1)[0] + "..."

    topics = parse_topics(
        row.get("topics", row.get("topic_list", ""))
    )

    tags = "".join(
        f'<span class="tag">{html.escape(topic)}</span>'
        for topic in topics[:4]
    )

    if not tags:
        tags = '<span class="tag">TED Talk</span>'

    url = valid_url(row.get("url", ""))

    score_html = ""

    if score is not None:
        score_html = (
            f'<span class="score">Similarity: {score:.2f}</span>'
        )

    if url:
        link = (
            f'<a class="watch-link" href="{html.escape(url, quote=True)}" '
            f'target="_blank" rel="noopener">Watch on TED ↗</a>'
        )
    else:
        link = '<span class="muted">TED link unavailable</span>'

    if selected:
        card_class = "selected-card"
        heading = "YOUR SELECTED TALK"
    else:
        card_class = "talk-card"
        heading = ""

    card_html = f"""
    <div class="{card_class}">
        <div class="small-label">{heading}</div>
        <div class="talk-title">{title}</div>
        <div class="speaker">Speaker: {speaker}</div>
        <div>{tags}</div>
        <div class="description">{description}</div>
        <div class="card-footer">
            {score_html}
            {link}
        </div>
    </div>
    """

    st.html(card_html)


# 11. DISPLAY RECOMMENDATIONS


if (
    "selected_index" in st.session_state
    and "recommendation_scores" in st.session_state
):

    selected_index = st.session_state["selected_index"]

    scores = st.session_state["recommendation_scores"]

    st.markdown("### Your Selected Talk")

    render_talk_card(
        df.iloc[selected_index],
        selected=True
    )

    ranked_indices = [
        int(i)
        for i in np.argsort(scores)[::-1]
        if i != selected_index and scores[i] >= 0
    ]

    # TOPIC FILTER

    if selected_topic != "All Topics":

        ranked_indices = [
            i
            for i in ranked_indices
            if selected_topic.lower() in [
                topic.lower()
                for topic in parse_topics(
                    df.iloc[i].get(topic_column, "")
                )
            ]
        ]

    # SORTING

    if (
        sort_option == "Most Viewed"
        and "views" in df.columns
    ):

        ranked_indices.sort(
            key=lambda i: pd.to_numeric(
                pd.Series([df.iloc[i]["views"]]),
                errors="coerce"
            ).fillna(0).iloc[0],
            reverse=True
        )

    elif (
        sort_option == "Shortest Duration"
        and "duration" in df.columns
    ):

        ranked_indices.sort(
            key=lambda i: pd.to_numeric(
                pd.Series([df.iloc[i]["duration"]]),
                errors="coerce"
            ).fillna(float("inf")).iloc[0]
        )

    # DISPLAY RECOMMENDATION CARDS

    st.markdown(
        '<div class="section-title">⭐ Recommended For You</div>',
        unsafe_allow_html=True
    )

    if not ranked_indices:

        st.info(
            "No recommendations found for this filter. "
            "Try selecting All Topics."
        )

    else:

        display_indices = ranked_indices[:result_count]

        columns = st.columns(3, gap="medium")

        for position, talk_index in enumerate(display_indices):

            with columns[position % 3]:

                render_talk_card(
                    df.iloc[talk_index],
                    score=float(scores[talk_index])
                )

                st.write("")