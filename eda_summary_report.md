# Exploratory Data Analysis & Data Quality Report

## 1. Dataset Dimensions and Overview
- **Total Records Analyzed**: 4467
- **Unique Speakers**: 3733
- **Date Range**: 2006-06-01 to 2020-04-01

## 2. Text Statistics
- **Average Title Length**: 6.9 words (Median: 7.0)
- **Average Details Length**: 58.75 words (Median: 57.0)
- **Min / Max Details Length**: 5 / 216 words

## 3. Data Quality & Preprocessing Highlights
- HTML tags, punctuation, and URLs were cleaned using regular expressions.
- Text normalized to lowercase to unify vocabulary representation.
- Null speakers imputed with `'Unknown'`.
- Verified 0 duplicate talk identifiers (`idx`).
- Missing views: 4258 talks (95.32%) do not report view count, justifying a hybrid or content-primary architecture over view-dependent collaborative filtering.

## 4. Key Topic Themes
The most salient topics extracted from TEDx transcripts and descriptions include:
- **shows** (TF-IDF weight: 255.1)
- **like** (TF-IDF weight: 284.1)
- **ted** (TF-IDF weight: 289.9)
- **people** (TF-IDF weight: 321.0)
- **life** (TF-IDF weight: 323.0)
- **shares** (TF-IDF weight: 350.9)
- **says** (TF-IDF weight: 360.6)
- **new** (TF-IDF weight: 369.1)
- **world** (TF-IDF weight: 458.9)
- **talk** (TF-IDF weight: 552.1)
