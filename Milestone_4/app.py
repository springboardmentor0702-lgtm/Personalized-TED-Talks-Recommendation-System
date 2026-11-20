from turtle import color
import pandas as pd
import numpy as np
import ast
import joblib
import streamlit as st
import base64
from sklearn.metrics.pairwise import cosine_similarity

# -----------------------------------
# 1. Load TED Talks Dataset
# -----------------------------------

DATA_PATH = "Data/TED_Talks_EDA_Prepared.csv"

df = pd.read_csv(DATA_PATH)

# -----------------------------------
# Topic Processing
# -----------------------------------

def extract_topics(topic_value):

    if isinstance(topic_value, list):
        return topic_value

    if pd.isna(topic_value):
        return []

    try:
        topics = ast.literal_eval(topic_value)

        if isinstance(topics, list):
            return topics

        return []

    except (ValueError, SyntaxError):
        return []


# Convert topic strings into Python lists
df['Topics_List'] = df['Topics'].apply(extract_topics)


# Extract all unique topics
all_topics = sorted(
    set(
        topic
        for topics in df['Topics_List']
        for topic in topics
        if isinstance(topic, str) and topic.strip()
    )
)

print("Total unique topics:", len(all_topics))
print("Sample topics:", all_topics[:20])

# -----------------------------------
# 2. Load Trained TF-IDF Vectorizer
# -----------------------------------

MODEL_PATH = "Models/tfidf_vectorizer.pkl"

tfidf_vectorizer = joblib.load(MODEL_PATH)

# -----------------------------------
# 3. Streamlit Page Configuration
# -----------------------------------

st.set_page_config(
    page_title="TED Talk Recommendation System",
    page_icon="🎤",
    layout="wide"
)

# -----------------------------------
# PREMIUM UI DESIGN - CUSTOM CSS
# -----------------------------------


st.markdown("""
<style>

/* Main Application Background */
.stApp {
    background-color: #101114;
    color: #F5F5F5;
}

/* Main Content Area */
.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1400px;
}

/* Main Headings */
h1, h2, h3 {
    color: #FFFFFF !important;
    font-weight: 700 !important;
}

/* Paragraph Text */
p, label, span {
    color: #D1D1D1;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: #191A20;
    border-right: 1px solid #303139;
}

/* Buttons */
.stButton > button {
    background-color: #E62B1E;
    color: white;
    border: none;
    border-radius: 8px;
    padding: 0.6rem 1.2rem;
    font-weight: 600;
    transition: 0.3s;
}

.stButton > button:hover {
    background-color: #FF4438;
    color: white;
    transform: translateY(-2px);
}

/* Selectboxes */
.stSelectbox > div > div {
    background-color: #202127;
    border-radius: 8px;
}

/* Metrics */
[data-testid="stMetric"] {
    background-color: #1D1E25;
    padding: 20px;
    border-radius: 12px;
    border: 1px solid #33343D;
}

/* Recommendation Cards */
[data-testid="stVerticalBlockBorderWrapper"] {
    background-color: #1B1C22;
    border: 1px solid #34353E;
    border-radius: 14px;
    padding: 12px;
}

/* Links */
a {
    color: #FF5147 !important;
}

/* Dividers */
hr {
    border-color: #34353E;
}

/* Hide Streamlit Branding */
#MainMenu {
    visibility: hidden;
}

footer {
    visibility: hidden;
}

</style>
""", unsafe_allow_html=True)


# -----------------------------------
# 4. Display Application Title
# -----------------------------------
# -----------------------------------
# PREMIUM HERO SECTION
# -----------------------------------

# -----------------------------------
# LOAD TED BACKGROUND IMAGE
# -----------------------------------

with open("Assets/Ted_Talks_BG.png", "rb") as image_file:
    bg_image = base64.b64encode(
        image_file.read()
    ).decode()

hero_html ="""

<div style="
    background-image:
        linear-gradient(
            90deg,
            rgba(8, 8, 12, 0.94),
            rgba(8, 8, 12, 0.70),
            rgba(8, 8, 12, 0.25)
        ),
        url('data:image/png;base64,BG_IMAGE');

    background-size: cover;
    background-position: center;
    background-repeat: no-repeat;

    min-height: 320px;
    padding: 45px 40px;
    border-radius: 18px;
    border: 1px solid #40343A;
    margin-bottom: 30px;
">
    

    <div style="
        color: #FF5147;
        font-size: 14px;
        font-weight: 700;
        letter-spacing: 3px;
        margin-bottom: 15px;
    ">
        DISCOVER • EXPLORE • GET INSPIRED
    </div>

    <h1 style="
        font-size: 46px;
        font-weight: 800;
        color: white;
        margin-bottom: 12px;
    ">
        AI-Powered <br>
        <span style="color: #FF5147;">TED Talk Recommendations</span>
    </h1>

    <p style="
        font-size: 18px;
        color: #C8C8D0;
        max-width: 750px;
        line-height: 1.7;
    ">
        Explore thousands of TED Talks and discover relevant talks
        based on content similarity using Machine Learning,
        Natural Language Processing and TF-IDF.
    </p>

    <div style="margin-top: 25px;">
        <span style="
            background: #E62B1E;
            color: white;
            padding: 9px 16px;
            border-radius: 6px;
            font-size: 13px;
            font-weight: 600;
            margin-right: 10px;
        ">
            AI POWERED
        </span>

        <span style="
            background: #292A32;
            color: #E0E0E0;
            padding: 9px 16px;
            border-radius: 6px;
            font-size: 13px;
            border: 1px solid #44454F;
        ">
            CONTENT-BASED RECOMMENDATIONS
        </span>
    </div>

</div>
"""
hero_html = hero_html.replace("BG_IMAGE", bg_image)
st.html(hero_html)

# -----------------------------------
# 5. Display Dataset Information
# -----------------------------------

# -----------------------------------
# DISCOVERY STATISTICS
# -----------------------------------

total_talks = len(df)

total_speakers = df['Speaker_1'].nunique()

total_topics = len(all_topics)

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        label="🎬 Total TED Talks",
        value=f"{total_talks:,}"
    )

with col2:
    st.metric(
        label="🎤 Unique Speakers",
        value=f"{total_speakers:,}"
    )

with col3:
    st.metric(
        label="🏷️ Available Topics",
        value=f"{total_topics:,}"
    )

# -----------------------------------
# 6. TED Talk Selection
# -----------------------------------

st.divider()

st.header("🔍 Find Similar TED Talks")

st.write(
    "Select a TED Talk from the dropdown "
    "to discover similar talks."
)

selected_title = st.selectbox(
    "Choose a TED Talk:",
    options=df['Title'].tolist()
)

top_n = st.radio(
    "Number of recommendations:",
    [5, 10],
    horizontal=True
)

# -----------------------------------
# Topic-Based Filtering
# -----------------------------------

selected_topic = st.selectbox(
    "Filter recommendations by topic:",
    options=["All Topics"] + all_topics
)

# -----------------------------------
# Recommendation Sorting
# -----------------------------------

sort_option = st.selectbox(
    "Sort recommendations by:",
    [
        "Highest Similarity Score",
        "Most Viewed",
        "Least Viewed",
        "Shortest Duration",
        "Longest Duration"
    ]
)

# -----------------------------------
# Selected TED Talk Preview
# -----------------------------------

st.divider()

st.subheader("📖 Selected TED Talk Details")

selected_talk = df[
    df['Title'] == selected_title
].iloc[0]

with st.container(border=True):

    st.markdown(f"### 🎬 {selected_talk['Title']}")

    st.write(f"**Speaker:** {selected_talk['Speaker_1']}")

    st.write(
        "**Topics:** " +
        ", ".join(selected_talk['Topics_List'])
    )

    st.write("**Description:**")

    st.write(selected_talk['Description'])

    if pd.notna(selected_talk['URL']):

        st.link_button(
            "▶ Watch Selected TED Talk",
            selected_talk['URL']
        )

# -----------------------------------
# 7. Generate TF-IDF Matrix
# -----------------------------------

tfidf_matrix = tfidf_vectorizer.transform(
    df['Combined_Text'].fillna('')
)

# -----------------------------------
# 8. Recommendation Function
# -----------------------------------

def recommend_talks(selected_title, top_n=5):

    # Find the selected talk
    talk_index = df.index[
        df['Title'] == selected_title
    ][0]

    # Get its TF-IDF vector
    query_vector = tfidf_matrix[talk_index]

    # Calculate cosine similarity
    similarity_scores = cosine_similarity(
        query_vector,
        tfidf_matrix
    ).flatten()

    # Exclude the selected talk itself
    similarity_scores[talk_index] = -1

    # Get top similar talks
    similar_indices = np.argsort(
        similarity_scores
    )[::-1][:top_n]

    # Prepare recommendations
    recommendations = df.iloc[similar_indices][
    ['Title', 'Speaker_1', 'Topics',
     'Description', 'URL', 'Views', 'Duration']
    ].copy()

    recommendations['Similarity Score'] = (
        similarity_scores[similar_indices]
    )

    return recommendations

# -----------------------------------
# 9. Display Recommendations
# -----------------------------------

if st.button("🔍 Find Similar Talks"):

    results = recommend_talks(
        selected_title,
        top_n
    )

    # -----------------------------------
    # Apply Topic Filter
    # -----------------------------------

    if selected_topic != "All Topics":

        results = results[
            results['Topics'].apply(
                lambda topic: selected_topic in extract_topics(topic)
            )
        ]

        # -----------------------------------
        # Sort Recommendations
        # -----------------------------------

        if sort_option == "Most Viewed":

            results = results.sort_values(
                by="Views",
                ascending=False
            )

        elif sort_option == "Least Viewed":

            results = results.sort_values(
                by="Views",
                ascending=True
            )

        elif sort_option == "Shortest Duration":

            results = results.sort_values(
                by="Duration",
                ascending=True
            )

        elif sort_option == "Longest Duration":

            results = results.sort_values(
                by="Duration",
                ascending=False
            )

        else:

            results = results.sort_values(
                by="Similarity Score",
                ascending=False
            )

    # -----------------------------------
    # Check Empty Results
    # -----------------------------------

    if results.empty:

        st.info(
            f"No recommendations found for the topic: {selected_topic}"
        )

    else:

        st.subheader("🎯 Recommended TED Talks")

        # -----------------------------------
        # Display Recommendation Cards
        # -----------------------------------

        for index, row in results.iterrows():

            with st.container(border=True):

                st.markdown(f"### 🎬 {row['Title']}")

                st.write(f"**Speaker:** {row['Speaker_1']}")

                topics = row['Topics']

                if isinstance(topics, str):
                    try:
                        topics = ast.literal_eval(topics)
                    except (ValueError, SyntaxError):
                        topics = [topics]

                st.write(
                    "**Topics:** " + ", ".join(topics)
                    if isinstance(topics, list)
                    else f"**Topics:** {topics}"
                )

                st.write("**Description:**")

                st.write(row['Description'])

                st.metric(
                    "Text Similarity Score",
                    f"{row['Similarity Score']:.4f}"
                )

                if pd.notna(row['URL']) and str(row['URL']).startswith("http"):

                    st.link_button(
                        "▶ Watch TED Talk",
                        row['URL']
                    )