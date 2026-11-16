import streamlit as st

from recommendation import (
    load_data,
    build_model,
    recommend_talks,
    get_popular_talks
)


# -----------------------------------------
# PAGE SETTINGS
# -----------------------------------------

st.set_page_config(
    page_title="TEDFinder",
    page_icon="T",
    layout="wide"
)


# -----------------------------------------
# DESIGN
# -----------------------------------------

st.markdown("""
<style>

.block-container {
    max-width: 1100px;
    padding-top: 2rem;
}

h1 {
    font-size: 43px;
    letter-spacing: -1.5px;
    font-weight: 700;
}

h2, h3 {
    letter-spacing: -0.5px;
}

div[data-testid="stMetric"] {
    background-color: #F0EDE7;
    padding: 16px;
    border-radius: 0px;
}

div[data-testid="stVerticalBlockBorderWrapper"] {
    border-radius: 5px;
}

.stButton button {
    border-radius: 4px;
}

</style>
""", unsafe_allow_html=True)


# -----------------------------------------
# LOAD DATA AND MODEL
# -----------------------------------------

@st.cache_data
def get_data():
    return load_data()


@st.cache_resource
def get_model():
    data = load_data()
    return build_model(data)


try:
    df = get_data()
    similarity_matrix = get_model()

except Exception as error:
    st.error("Unable to load TEDFinder.")
    st.exception(error)
    st.stop()


# -----------------------------------------
# HELPER FUNCTIONS
# -----------------------------------------

def format_views(views):

    views = int(views)

    if views >= 1_000_000:
        return f"{views / 1_000_000:.1f}M"

    if views >= 1_000:
        return f"{views / 1_000:.1f}K"

    return str(views)


def format_duration(seconds):

    seconds = int(seconds)

    minutes = seconds // 60
    remaining = seconds % 60

    return f"{minutes}:{remaining:02d} min"


def talk_card(talk, score=None):

    with st.container(border=True):

        st.subheader(str(talk["title"]))

        st.write(f"**{talk['speaker_1']}**")

        if "topics" in talk:
            st.caption(str(talk["topics"]))

        if "description" in talk:

            description = str(talk["description"])

            if len(description) > 220:
                description = description[:217] + "..."

            st.write(description)

        col1, col2 = st.columns(2)

        col1.caption(
            f"Views: {format_views(talk['views'])}"
        )

        col2.caption(
            f"Duration: {format_duration(talk['duration'])}"
        )

        if score is not None:

            st.progress(
                max(0.0, min(float(score), 1.0)),
                text=f"Similarity: {score:.3f}"
            )

        url = str(talk["url"])

        if url.startswith("http"):
            st.markdown(f"[Watch on TED ↗]({url})")


# -----------------------------------------
# SIDEBAR
# -----------------------------------------

with st.sidebar:

    st.title("TEDFinder")

    st.caption("Discover ideas worth sharing.")

    st.divider()

    page = st.radio(
        "Explore",
        [
            "Home",
            "Popular Talks",
            "Find Similar Talks"
        ]
    )

    st.divider()

    st.caption("THE COLLECTION")

    st.write(f"**{len(df):,}** TED Talks")

    st.caption(
        "A content-based recommendation platform."
    )


# =========================================
# HOME PAGE
# =========================================

if page == "Home":

    st.caption("TED TALK DISCOVERY PLATFORM")

    st.title("Ideas worth discovering.")

    st.write(
        "Explore conversations that challenge perspectives, "
        "introduce new ideas and inspire curiosity. "
        "Find your next interesting TED Talk."
    )

    st.divider()

    # Overview

    st.subheader("The collection")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Total Talks",
        f"{len(df):,}"
    )

    col2.metric(
        "Speakers",
        f"{df['speaker_1'].nunique():,}"
    )

    col3.metric(
        "Topics",
        f"{df['topics'].nunique():,}"
    )

    st.divider()

    # Featured talk

    st.subheader("Editor's pick")

    st.caption(
        "The most viewed talk in the collection."
    )

    featured = get_popular_talks(df, 1).iloc[0]

    talk_card(featured)

    st.divider()

    # Popular selection

    st.subheader("Popular talks")

    popular = get_popular_talks(df, 10)

    columns = st.columns(2)

    for i, (_, talk) in enumerate(popular.iterrows()):

        with columns[i % 2]:
            talk_card(talk)



# =========================================
# POPULAR TALKS PAGE
# =========================================

elif page == "Popular Talks":

    st.caption("EXPLORE THE COLLECTION")

    st.title("Popular TED Talks")

    st.write(
        "Explore the most-watched TED Talks "
        "and discover ideas from inspiring speakers."
    )

    st.divider()

    topic_list = sorted(
        set(
            topic.strip()
            for value in df["topics"].dropna()
            for topic in str(value).split(",")
            if topic.strip()
        )
    )

    selected_topic = st.selectbox(
        "Filter by topic",
        ["All Topics"] + topic_list
    )

    filtered = df.copy()

    if selected_topic != "All Topics":

        filtered = filtered[
            filtered["topics"].str.contains(
                selected_topic,
                case=False,
                na=False,
                regex=False
            )
        ]

    filtered = filtered.sort_values(
        "views",
        ascending=False
    )

    st.divider()

    st.write(f"**{len(filtered):,} talks found**")

    if filtered.empty:

        st.info("No talks available for this topic.")

    else:

        for _, talk in filtered.head(50).iterrows():

            talk_card(talk)

# =========================================
# RECOMMENDATION PAGE
# =========================================

elif page == "Find Similar Talks":

    st.caption("PERSONALISED DISCOVERY")

    st.title("Find talks that connect.")

    st.write(
        "Select a TED Talk you enjoy. "
        "Our content-based recommendation system "
        "will find five talks with similar themes and content."
    )

    st.divider()

    # Searchable title selection

    selected_title = st.selectbox(
        "Choose a TED Talk",
        options=df["title"].tolist(),
        index=None,
        placeholder="Search and select a talk..."
    )

    if selected_title:

        selected_talk = df[
            df["title"] == selected_title
        ].iloc[0]

        st.divider()

        st.subheader("Selected talk")

        talk_card(selected_talk)

        st.divider()

        # Recommendation button

        if st.button(
            "Find Similar Talks",
            type="primary",
            use_container_width=True
        ):

            results = recommend_talks(
                df,
                similarity_matrix,
                selected_title,
                top_n=5
            )

            if results is None or results.empty:

                st.warning("No recommendations found.")

            else:

                st.session_state["recommendations"] = results
                st.session_state["selected_title"] = selected_title

    # Display recommendations

    if (
        "recommendations" in st.session_state
        and st.session_state.get("selected_title") == selected_title
    ):

        st.divider()

        st.subheader("Recommended for you")

        st.caption(
            "Ranked by cosine similarity between talk content."
        )

        results = st.session_state["recommendations"]

        for _, result in results.iterrows():

            with st.container(border=True):

                st.subheader(result["title"])

                st.write(
                    f"**{result['speaker_1']}**"
                )

                st.caption(
                    f"Published: {int(result['year']) if st.session_state.get('selected_title') and str(result['year']).replace('.0', '').isdigit() else 'N/A'}"
                )

                score = float(result["similarity_score"])

                st.progress(
                    max(0.0, min(score, 1.0)),
                    text=f"Similarity score: {score:.3f}"
                )

                if str(result["url"]).startswith("http"):

                    st.markdown(
                        f"[Watch this talk ↗]({result['url']})"
                    )


# -----------------------------------------
# FOOTER
# -----------------------------------------

st.divider()

st.caption(
    "TEDFinder | Content-Based TED Talk Recommendation System"
)