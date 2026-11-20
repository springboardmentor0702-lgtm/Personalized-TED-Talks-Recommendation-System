# 🎤 AI-Powered TED Talk Recommendation System

## 1. Project Overview

The AI-Powered TED Talk Recommendation System is a machine learning-based web application designed to help users discover relevant TED Talks based on their content.

The application uses Natural Language Processing (NLP), TF-IDF (Term Frequency–Inverse Document Frequency) vectorization, and Cosine Similarity to identify and recommend talks that are similar to a user's selected TED Talk.

The system provides an interactive Streamlit interface where users can explore TED Talks, view talk details, filter recommendations by topic, and sort results using different criteria.

## 2. Project Objectives

- Develop an intelligent content-based recommendation system for TED Talks.
- Apply NLP techniques to process and represent textual information.
- Identify similar TED Talks using TF-IDF and Cosine Similarity.
- Provide an interactive and user-friendly recommendation interface.
- Help users discover talks based on their interests and selected content.

## 3. Key Features

### 🎯 TED Talk Selection
- Select a TED Talk from an interactive dropdown list.
- View information about the selected talk.

### 🤖 AI-Powered Recommendations
- Generate similar TED Talks using a content-based recommendation approach.
- Calculate similarity scores using Cosine Similarity.
- Display the top 5 or 10 recommendations.

### 🏷️ Topic-Based Filtering
- Filter recommended talks according to available topics.
- Explore recommendations related to specific subjects.

### 📊 Recommendation Sorting
Users can sort recommendations using:
- Highest Similarity Score
- Most Viewed
- Least Viewed
- Shortest Duration
- Longest Duration

### 📖 Talk Details
Display information including:
- TED Talk title
- Speaker name
- Topics
- Description
- Similarity score
- TED Talk URL

### 🎨 Interactive User Interface
- Custom TED-inspired design.
- Dark-themed interface with a TED auditorium background.
- Statistics displaying total talks, unique speakers, and available topics.
- Interactive selection, filtering, and sorting controls.

## 4. Technologies Used

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| Streamlit | Interactive web application |
| Pandas | Dataset processing and manipulation |
| NumPy | Numerical operations |
| Scikit-learn | Machine learning and text vectorization |
| TF-IDF | Text feature extraction |
| Cosine Similarity | Measuring textual similarity |
| Joblib | Loading the trained TF-IDF vectorizer |
| HTML and CSS | Custom interface styling |

## 5. Recommendation Methodology

The application uses a content-based filtering approach.

### Step 1: Text Processing
The prepared TED Talks dataset contains information such as titles, descriptions, topics, and transcripts.

The combined textual information is used to represent the content of each TED Talk.

### Step 2: TF-IDF Vectorization
TF-IDF converts textual information into numerical feature vectors.

The trained TF-IDF vectorizer is loaded into the application using Joblib.

### Step 3: Cosine Similarity
Cosine Similarity measures the similarity between the selected TED Talk and other talks based on their TF-IDF representations.

### Step 4: Recommendation Generation
The selected talk itself is excluded, and the most similar talks are ranked according to their similarity scores.

The application displays the requested number of recommendations.

### Step 5: Filtering and Sorting
Users can further refine recommendations using topic filters and sorting options.

## 6. Dataset Information

The project uses a prepared TED Talks dataset containing **4,016 records and 25 columns**.

Important dataset attributes include:

- Talk title
- Speaker information
- Views
- Recorded and published dates
- Topics
- Description
- Transcript
- Combined textual content

## 7. Project Folder Structure

```text
TED_Talk_Recommendation_System/
│
├── app.py
│
├── Data/
│   └── TED_Talks_EDA_Prepared.csv
│
├── Models/
│   └── tfidf_vectorizer.pkl
│
├── Assets/
│   └── Ted_Talks_BG.png
│
├── Screenshots/
│   ├── 01_Home_Page.png
│   ├── 02_TED_Talk_Selection.png
│   ├── 03_Selected_Talk_Details.png
│   └── 04_Recommendations.png
│
├── Documentation/
│   └── Functional_Testing.md
│
├── README.md
├── Requirements.txt
└── .gitignore
```

## 8. Installation and Setup Instructions

Follow these steps to run the project locally.

### Prerequisites

- Python 3.11
- Git
- Visual Studio Code (recommended)

### Step 1: Clone the Repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
```

Navigate to the project folder:

```bash
cd TED_Talk_Recommendation_System
```

### Step 2: Create a Virtual Environment

```bash
python -m venv venv
```

Activate the environment on Windows:

```bash
venv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
python -m pip install -r Requirements.txt
```

### Step 4: Run the Application

```bash
streamlit run app.py
```

The application will open in your default web browser.

If it does not open automatically, visit:

```text
http://localhost:8501
```

## 9. Application Screenshots

The following screenshots demonstrate the application interface and its functionality.

### Home Page

![Home Page](Screenshots/01_Home_Page.png)

### TED Talk Selection

![TED Talk Selection](Screenshots/02_TED_Talk_Selection.png)

### Selected TED Talk Details

![Selected Talk Details](Screenshots/03_Selected_Talk_Details.png)

### Recommendation Results

![Recommendation Results](Screenshots/04_Recommendations.png)

## 10. Example Recommendation Workflow

1. Open the TED Talk Recommendation System.
2. Select a TED Talk from the dropdown.
3. Choose either 5 or 10 recommendations.
4. Optionally select a topic filter.
5. Choose a sorting method.
6. Click **Find Similar TED Talks**.
7. Explore the recommended talks and their similarity scores.
8. Use the provided link to watch a TED Talk.

## 11. Evaluation and Model Selection

The recommendation approaches were evaluated using Precision@5, Recall@5 (Topic Coverage), Item Recall@5, NDCG@5, and MAP@5.

Two content-based approaches were compared:

- Combined-text TF-IDF
- Topic-based TF-IDF

Combined-text TF-IDF was selected as the primary recommendation engine because it uses broader textual information rather than relying exclusively on topic labels.

Topic-based filtering is retained as an additional application feature.

## 12. Limitations

- Recommendations depend on the textual information available in the dataset.
- The system uses content-based filtering rather than personalized user interaction histories.
- Collaborative filtering is not implemented because user ratings, clicks, or viewing-history data are unavailable.
- Recommendation quality depends on the dataset and the trained vectorizer.

## 13. Future Enhancements

- Add personalized recommendations based on user preferences.
- Integrate collaborative filtering when user interaction data becomes available.
- Explore hybrid recommendation techniques.
- Improve recommendation quality using advanced text embeddings.
- Add more advanced search and discovery features.

## 14. Conclusion

The AI-Powered TED Talk Recommendation System demonstrates the practical application of Machine Learning and Natural Language Processing in content-based recommendation systems.

By combining TF-IDF vectorization and Cosine Similarity with an interactive Streamlit interface, the application helps users discover TED Talks relevant to their selected content.

The project provides practical experience in data processing, text representation, similarity-based recommendation, model integration, and web application development.