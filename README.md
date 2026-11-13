# Personalized TED Talks Recommendation System Using Machine Learning

## 1. Project Overview

The Personalized TED Talks Recommendation System is a machine learning-based web application that recommends TED Talks based on content similarity.

The system analyzes TED Talk titles, descriptions, topics and transcripts to identify talks with similar content.

It uses **TF-IDF Vectorization** to convert text into numerical features and **Cosine Similarity** to calculate similarity between talks.

The application is developed using Python and Streamlit.

## 2. Objectives

- Recommend TED Talks based on content similarity.
- Help users discover talks related to their interests.
- Implement text preprocessing and feature extraction.
- Develop an interactive web-based recommendation application.
- Evaluate recommendation quality using Precision@K and Recall@K.

## 3. Technologies Used

| Technology | Purpose |
|---|---|
| Python | Programming language |
| Pandas | Data processing |
| NumPy | Numerical operations |
| Scikit-learn | TF-IDF and cosine similarity |
| SciPy | Sparse matrix operations |
| Joblib | Saving and loading trained artifacts |
| Streamlit | Web application development |
| Matplotlib and Seaborn | Exploratory data analysis |

## 4. Dataset

**Dataset:** TED Ultimate Dataset

**Source:** Kaggle

https://www.kaggle.com/datasets/miguelcorraljr/ted-ultimate-dataset

The project uses the English TED Talks dataset containing approximately 4,005 talks.

Important features include:

- Talk ID
- Title
- Speaker
- Topics
- Description
- Transcript
- Views
- Duration
- TED URL

## 5. System Architecture

1. Data Collection
2. Data Cleaning and Preprocessing
3. Exploratory Data Analysis
4. Feature Engineering
5. TF-IDF Vectorization
6. Cosine Similarity Calculation
7. TED Talk Recommendation
8. Streamlit Web Application

## 6. Recommendation Algorithm

### TF-IDF Vectorization

TF-IDF (Term Frequency–Inverse Document Frequency) converts text information into numerical vectors.

The project uses a vocabulary of 50,000 features.

### Cosine Similarity

Cosine Similarity measures the similarity between two TED Talk vectors.

The similarity score is calculated using:

\[
\text{Similarity}(A,B)=
\frac{A\cdot B}{\|A\|\|B\|}
\]

Higher scores indicate greater content similarity.

The system excludes the selected talk and returns the most similar talks.

**Recommendation approach:** Content-Based Filtering.

Collaborative filtering is not implemented because user-item interaction data is not available in the selected dataset.

## 7. Application Features

- TED Talk search using a dropdown.
- Recommendations for Top 5 or Top 10 talks.
- Content-based recommendations using TF-IDF.
- Similarity scores displayed as percentages.
- Selected talk information.
- Speaker and topic details.
- Talk descriptions and TED URLs.
- Topic-based filtering.
- Sorting by similarity, views and duration.
- Duration displayed in minutes and seconds.
- Handling of missing values and invalid selections.

## 8. Project Folder Structure

```text
TED_Talks_Recommendation_System/
│
├── data/
│   └── ted_model_dataset.csv
│
├── models/
│   ├── ted_tfidf_matrix.joblib
│   └── ted_tfidf_vectorizer.joblib
│
├── app.py
├── check_model.py
├── requirements.txt
└── README.md
```

## 9. Installation and Execution

### Step 1: Clone or download the project

Place the project folder on your computer.

### Step 2: Open the project directory

```bash
cd TED_Talks_Recommendation_System
```

### Step 3: Create a virtual environment

```bash
python -m venv .venv
```

### Step 4: Activate the environment

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

### Step 5: Install dependencies

```bash
python -m pip install -r requirements.txt
```

### Step 6: Run the application

```bash
python -m streamlit run app.py
```

### Step 7: Open the application

Visit:

http://localhost:8501

## 10. Testing

The following tests were performed:

| Test Case | Expected Result |
|---|---|
| Dataset loading | Dataset loads successfully |
| Model loading | TF-IDF artifacts load successfully |
| Cosine similarity | Similarity score calculated |
| TED Talk search | Talk can be selected |
| Top 5 recommendations | Five recommendations displayed when available |
| Top 10 recommendations | Up to ten recommendations displayed |
| Topic filtering | Results filtered by selected topics |
| Sorting | Results reordered correctly |
| Missing information | Application handles missing values |
| Invalid selection | Application handles invalid selections |
| TED URLs | Valid links displayed |

## 11. Evaluation Metrics

The recommendation system was evaluated using:

- Precision@K
- Recall@K
- NDCG@K
- Mean Average Precision (MAP@K)

These metrics measure recommendation relevance and ranking quality.

The evaluation uses the dataset's related-talk references as relevance labels.

## 12. Future Enhancements

- Develop user registration and login.
- Introduce personalized user profiles.
- Implement collaborative filtering when interaction data becomes available.
- Explore Word2Vec, BERT and sentence embeddings.
- Add user feedback and recommendation history.
- Deploy the application online.

## 13. Conclusion

The Personalized TED Talks Recommendation System successfully demonstrates content-based recommendation using machine learning.

By combining TF-IDF Vectorization, Cosine Similarity and Streamlit, the project provides an interactive platform for discovering relevant TED Talks.

The application integrates data preprocessing, feature extraction, recommendation generation and a user-friendly web interface into one complete system.