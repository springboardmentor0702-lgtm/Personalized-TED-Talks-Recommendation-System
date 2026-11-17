Personalized TED Talks Recommendation System Using Machine Learning

1. Project Overview

The Personalized TED Talks Recommendation System is a machine learning-based web application designed to help users discover TED Talks related to their interests.
The system analyzes TED Talk transcripts and topics to identify content similarities and recommend relevant talks based on the user's selected TED Talk.It uses TF-IDF (Term Frequency–Inverse Document Frequency) for text feature extraction and Cosine Similarity to measure the similarity between talks.
The recommendation engine combines transcript similarity and topic similarity to generate relevant recommendations.An interactive web application is developed using Streamlit, allowing users to search, select and explore recommended TED Talks.

2. Objectives

- Analyze and understand the TED Talks dataset through Exploratory Data Analysis.
- Clean and preprocess textual data.
- Extract meaningful features from transcripts and topics.
- Implement TF-IDF vectorization for text representation.
- Calculate similarity between TED Talks using Cosine Similarity.
- Develop a content-based recommendation engine.
- Evaluate recommendation performance using appropriate evaluation metrics.
- Build an interactive Streamlit application for TED Talk discovery.

3. Technologies Used

Technology   | Purpose 

Python       - Main programming language 
Pandas       - Dataset loading, cleaning and manipulation 
NumPy        - Numerical operations 
Scikit-learn - TF-IDF vectorization and cosine similarity 
SciPy        - Sparse matrix operations and storage 
Joblib       - Saving and loading trained vectorizers 
Streamlit    - Web application development 
Matplotlib   - Data visualization 
Seaborn      - Exploratory data analysis and visualization 

4. Dataset Description

Dataset: TED Talks Dataset

The project uses an English TED Talks dataset containing approximately 4,005 talks.The dataset includes information about TED Talks, speakers, topics, descriptions and transcripts.

Important Features

talk_id – Unique identifier of the TED Talk.
title – Title of the talk.
speaker_1 – Primary speaker name.
all_speakers – Information about speakers.
occupations – Speaker occupations.
about_speakers – Speaker descriptions.
views – Number of views.
recorded_date – Date when the talk was recorded.
published_date – Date when the talk was published.
duration – Duration of the talk.
topics – Topics or tags associated with the talk.
description – Short description of the talk.
transcript – Full transcript of the talk.
url – Original TED Talk link.

The cleaned dataset is used for feature extraction and recommendation generation.

5. System Architecture

The project follows these stages:

1. Dataset Collection
2. Data Cleaning and Preprocessing
3. Exploratory Data Analysis (EDA)
4. Text Preprocessing
5. Feature Extraction Using TF-IDF
6. Cosine Similarity Calculation
7. Content-Based Recommendation Generation
8. Recommendation Evaluation
9. Streamlit Application Development

6. Data Preprocessing and Exploratory Data Analysis

Data Cleaning
The dataset is cleaned before performing analysis and model development.
The preprocessing steps include:

- Standardizing column names.
- Removing unnecessary spaces from text values.
- Converting empty strings into missing values.
- Removing duplicate records.
- Handling missing values in important columns.
- Cleaning textual information.

Text Preprocessing

Text preprocessing is performed on TED Talk transcripts and topics.

The steps include:

- Converting text into lowercase.
- Removing URLs.
- Removing special characters and unnecessary symbols.
- Removing English stopwords.
- Preparing cleaned text for TF-IDF vectorization.

Exploratory Data Analysis

EDA is performed to understand the dataset and identify useful patterns.

The analysis includes:

- Distribution of TED Talk views.
- Most popular topics.
- Speaker analysis.
- Talk duration analysis.
- Transcript and description word counts.
- Recorded and published date analysis.
- Related TED Talks analysis.

Visualizations are created to understand the characteristics and distribution of the dataset.

7. Recommendation Algorithm

The project implements Content-Based Filtering using transcript and topic information.

7.1 TF-IDF Vectorization
TF-IDF is used to convert textual data into numerical feature vectors.Two separate TF-IDF representations are created.
Transcript TF-IDF:
- Extracts important terms from TED Talk transcripts.
- Uses a maximum vocabulary size of 5,000 features.
- Generates the transcript feature matrix.
Topic TF-IDF:
- Represents the topics or tags associated with TED Talks.
- Uses a maximum vocabulary size of 1,000 features.
- Generates the topic feature matrix.

The saved feature matrices have the following dimensions:

| Feature Matrix | Shape |
| Transcript TF-IDF Matrix | (4005, 5000) |
| Topic TF-IDF Matrix | (4005, 484) |

The topic matrix contains 484 actual vocabulary features after vectorization.

7.2 Cosine Similarity
Cosine Similarity measures the similarity between two numerical feature vectors.It calculates the cosine of the angle between the vectors.

Formula:

Similarity(A, B) = (A · B) / (||A|| × ||B||)

A higher similarity score indicates greater similarity between the content of two TED Talks.
The system calculates:
- Transcript similarity.
- Topic similarity.

7.3 Weighted Similarity Calculation

Both similarity scores are combined using a weighted approach.
Final Similarity = (0.70 × Transcript Similarity) + (0.30 × Topic Similarity)
This gives 70% importance to transcript similarity and 30% importance to topic similarity.

7.4 Recommendation Generation

When a user selects a TED Talk:

1. The system identifies the selected talk.
2. Retrieves its transcript and topic feature vectors.
3. Calculates similarity with other TED Talks.
4. Combines both similarity scores using the weighted formula.
5. Ranks the talks based on their final similarity scores.
6. Excludes the selected talk from the results.
7. Displays the most similar TED Talks.

The saved TF-IDF vectorizers and sparse matrices are reused in the application instead of rebuilding the feature representations every time.

8. Model Evaluation

The recommendation system is evaluated using a train-test split.

- Training data: 3,204 records.
- Testing data: 801 records.

The recommendation performance is assessed using Precision@K.

Precision@K measures the proportion of relevant recommendations among the top K recommended talks.

The evaluation uses topic overlap between talks as a relevance proxy.

The reported Precision@10 score is 98.44% under this evaluation approach.

This score represents relevance according to the topic-overlap proxy and should not be interpreted as actual user satisfaction or a guarantee of recommendation accuracy.

9. Streamlit Application Features

The Streamlit application provides the following features:

1. User-Friendly Interface
A clean and interactive interface allows users to explore TED Talks easily.

2. TED Talk Search
Users can search for TED Talks by entering part or all of a talk title.

3. Talk Selection
Matching TED Talks are displayed for users to select their preferred talk.

4. Similar Talk Recommendations
The system generates recommendations based on the selected TED Talk using the content-based recommendation engine.

5. Top Recommendations
Users can choose the number of recommendations to display, such as Top 5 or Top 10.

6. Similarity Score
Each recommendation displays a similarity score to indicate how closely it matches the selected talk.

7. Recommendation Details
Each recommendation displays:
- TED Talk title.
- Speaker name.
- Topics.
- Short description.
- Similarity score.
- Original TED Talk link.

8. Topic-Based Filtering
Users can filter recommendations based on their preferred topics.

9. Sorting Options
Users can sort recommendations based on:
- Highest similarity.
- Most viewed.
- Shortest duration.

10. Direct TED Talk Access
Users can click the Watch on TED link to access the original talk.

10. Project Folder Structure

text
TED_Talks_Recommendation/
│
├── app.py
│
├── ted_talks_final.csv
│
├── transcript_tfidf_vectorizer.pkl
├── tags_tfidf_vectorizer.pkl
│
├── transcript_tfidf_matrix.npz
├── tags_tfidf_matrix.npz
│
├── requirements.txt


File Description

| File | Description |
app.py | Main Streamlit application |
ted_talks_final.csv | Cleaned TED Talks dataset |
transcript_tfidf_vectorizer.pkl | Saved transcript TF-IDF vectorizer |
tags_tfidf_vectorizer.pkl | Saved topic TF-IDF vectorizer |
transcript_tfidf_matrix.npz | Saved transcript feature matrix |
tags_tfidf_matrix.npz | Saved topic feature matrix |
req.txt | Required Python libraries |

11. Installation and Execution

Step 1: Install Dependencies

python -m pip install -r req.txt

Step 2: Run the Streamlit Application

streamlit run app.py

Step 3: Open the Application

http://localhost:8501/

12. Testing

The application is tested using different scenarios to verify its functionality.

| Test Case | Expected Result |
| Dataset loading | Dataset loads successfully |
| Vectorizer loading | Saved TF-IDF vectorizers load successfully |
| Matrix loading | Sparse matrices load successfully |
| TED Talk search | Matching titles are displayed |
| Invalid search | Appropriate message is displayed |
| Talk selection | Selected TED Talk is identified |
| Recommendation generation | Similar talks are displayed |
| Selected talk exclusion | Selected talk does not appear in recommendations |
| Topic filtering | Recommendations are filtered correctly |
| Sorting | Results are reordered correctly |
| Missing information | Missing values are handled safely |
| TED URL | Valid TED Talk links are displayed |

13. Future Enhancements

- Implement collaborative filtering using user interaction data.
- Develop a hybrid recommendation system combining content-based and collaborative filtering.
- Introduce user profiles and personalized recommendation history.
- Explore advanced text representations such as Word2Vec and BERT embeddings.
- Add user feedback to improve recommendation relevance.
- Deploy the application online for public access.

14. Conclusion

The Personalized TED Talks Recommendation System Using Machine Learning demonstrates how Natural Language Processing and machine learning techniques can be used to recommend relevant TED Talks.
By combining transcript-based TF-IDF, topic-based TF-IDF and Cosine Similarity, the system identifies content similarities and generates recommendations.The Streamlit application integrates the recommendation engine into an interactive web interface with search, filtering and sorting features.
Overall, the project provides an effective way to explore and discover TED Talks based on content similarity.
