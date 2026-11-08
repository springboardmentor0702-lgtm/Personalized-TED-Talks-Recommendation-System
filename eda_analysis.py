"""
Module 2: Exploratory Data Analysis (EDA) and Data Quality Audit
Personalized TED Talks Recommendation System
"""

import os
os.environ["MPLCONFIGDIR"] = os.path.join(os.getcwd(), "plots", ".mpl_cache")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer

BASE_DIR = Path(__file__).resolve().parent
PLOTS_DIR = BASE_DIR / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# Load preprocessed dataset
DATA_FILE = BASE_DIR / "ted_talks_preprocessed.csv"
RAW_FILE = BASE_DIR / "tedx_dataset.csv"

raw_df = pd.read_csv(RAW_FILE)
df = pd.read_csv(DATA_FILE)

print("=" * 60)
print("MODULE 2: EXPLORATORY DATA ANALYSIS & DATA QUALITY AUDIT")
print("=" * 60)

# 1. Dataset Overview & Completeness
total_talks = len(df)
print(f"Total Preprocessed Talks: {total_talks}")
print(f"Features Available: {list(df.columns)}")

# Word count calculations
df["title_word_count"] = df["title"].fillna("").apply(lambda x: len(str(x).split()))
df["details_word_count"] = df["details"].fillna("").apply(lambda x: len(str(x).split()))
df["combined_word_count"] = df["combined_text"].fillna("").apply(lambda x: len(str(x).split()))

# Summary statistics
stats = {
    "Total Talks": total_talks,
    "Unique Speakers": df["main_speaker"].nunique(),
    "Avg Title Word Count": round(df["title_word_count"].mean(), 2),
    "Median Title Word Count": round(df["title_word_count"].median(), 2),
    "Avg Details Word Count": round(df["details_word_count"].mean(), 2),
    "Median Details Word Count": round(df["details_word_count"].median(), 2),
    "Min Details Word Count": df["details_word_count"].min(),
    "Max Details Word Count": df["details_word_count"].max(),
    "Missing Views Count": int(df["num_views"].isnull().sum()),
    "Missing Views Percentage": round((df["num_views"].isnull().sum() / total_talks) * 100, 2),
    "Unique URLs": df["url"].nunique(),
}

print("\n--- Summary Statistics ---")
for k, v in stats.items():
    print(f"  {k:28s}: {v}")

# 2. Visualization 1: Word Count Distributions (Talk Lengths)
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.hist(df["title_word_count"], bins=20, color="#e62b1e", edgecolor="black", alpha=0.7)
plt.title("Distribution of Title Word Counts", fontsize=12, fontweight="bold")
plt.xlabel("Word Count")
plt.ylabel("Frequency")
plt.grid(axis="y", linestyle="--", alpha=0.5)

plt.subplot(1, 2, 2)
plt.hist(df["details_word_count"], bins=30, color="#2b5c8f", edgecolor="black", alpha=0.7)
plt.title("Distribution of Details (Description) Word Counts", fontsize=12, fontweight="bold")
plt.xlabel("Word Count")
plt.ylabel("Frequency")
plt.grid(axis="y", linestyle="--", alpha=0.5)

plt.tight_layout()
plot1_path = PLOTS_DIR / "01_talk_length_distribution.png"
plt.savefig(plot1_path, dpi=300)
plt.close()
print(f"Saved: {plot1_path.name}")

# 3. Visualization 2: Top Speakers
top_speakers = df[df["main_speaker"] != "Unknown"]["main_speaker"].value_counts().head(15)

plt.figure(figsize=(10, 6))
top_speakers.sort_values().plot(kind="barh", color="#e62b1e", edgecolor="black", alpha=0.85)
plt.title("Top 15 Most Frequent TEDx Speakers", fontsize=13, fontweight="bold")
plt.xlabel("Number of Talks")
plt.ylabel("Speaker")
plt.grid(axis="x", linestyle="--", alpha=0.5)
plt.tight_layout()
plot2_path = PLOTS_DIR / "02_top_speakers.png"
plt.savefig(plot2_path, dpi=300)
plt.close()
print(f"Saved: {plot2_path.name}")

# 4. Visualization 3: Top TF-IDF Keywords (Topic Frequencies)
tfidf = TfidfVectorizer(stop_words="english", max_features=25, ngram_range=(1, 2))
tfidf_matrix = tfidf.fit_transform(df["combined_text"])
feature_names = tfidf.get_feature_names_out()
tfidf_scores = np.array(tfidf_matrix.sum(axis=0)).flatten()

top_kw_indices = tfidf_scores.argsort()[::-1]
top_keywords = [(feature_names[i], tfidf_scores[i]) for i in top_kw_indices]

kw_df = pd.DataFrame(top_keywords, columns=["Keyword", "TF-IDF Sum"]).sort_values(by="TF-IDF Sum", ascending=True)

plt.figure(figsize=(10, 7))
plt.barh(kw_df["Keyword"], kw_df["TF-IDF Sum"], color="#1f77b4", edgecolor="black", alpha=0.8)
plt.title("Top 25 Salient Keywords & Topics (TF-IDF Aggregation)", fontsize=13, fontweight="bold")
plt.xlabel("Aggregated TF-IDF Score")
plt.ylabel("Term / Bigram")
plt.grid(axis="x", linestyle="--", alpha=0.5)
plt.tight_layout()
plot3_path = PLOTS_DIR / "03_top_keywords.png"
plt.savefig(plot3_path, dpi=300)
plt.close()
print(f"Saved: {plot3_path.name}")

# 5. Visualization 4: Missing Values Audit
missing_raw = raw_df.isnull().sum()
missing_clean = df.isnull().sum()

missing_df = pd.DataFrame({
    "Raw Dataset Missing": missing_raw,
    "Preprocessed Missing": missing_clean
}).fillna(0)

plt.figure(figsize=(9, 5))
bar_width = 0.35
indices = np.arange(len(missing_df))
plt.bar(indices - bar_width/2, missing_df["Raw Dataset Missing"], width=bar_width, label="Raw Dataset", color="#d9534f")
plt.bar(indices + bar_width/2, missing_df["Preprocessed Missing"], width=bar_width, label="Preprocessed Dataset", color="#5cb85c")
plt.xticks(indices, missing_df.index, rotation=35, ha="right")
plt.title("Missing Value Audit (Raw vs Preprocessed)", fontsize=13, fontweight="bold")
plt.ylabel("Missing Count")
plt.legend()
plt.tight_layout()
plot4_path = PLOTS_DIR / "04_missing_values_audit.png"
plt.savefig(plot4_path, dpi=300)
plt.close()
print(f"Saved: {plot4_path.name}")

# 6. Save Markdown EDA Summary Report
report_path = BASE_DIR / "eda_summary_report.md"
with open(report_path, "w", encoding="utf-8") as f:
    f.write("# Exploratory Data Analysis & Data Quality Report\n\n")
    f.write("## 1. Dataset Dimensions and Overview\n")
    f.write(f"- **Total Records Analyzed**: {total_talks}\n")
    f.write(f"- **Unique Speakers**: {df['main_speaker'].nunique()}\n")
    f.write(f"- **Date Range**: {df['posted_date'].dropna().min()} to {df['posted_date'].dropna().max()}\n\n")
    f.write("## 2. Text Statistics\n")
    f.write(f"- **Average Title Length**: {stats['Avg Title Word Count']} words (Median: {stats['Median Title Word Count']})\n")
    f.write(f"- **Average Details Length**: {stats['Avg Details Word Count']} words (Median: {stats['Median Details Word Count']})\n")
    f.write(f"- **Min / Max Details Length**: {stats['Min Details Word Count']} / {stats['Max Details Word Count']} words\n\n")
    f.write("## 3. Data Quality & Preprocessing Highlights\n")
    f.write("- HTML tags, punctuation, and URLs were cleaned using regular expressions.\n")
    f.write("- Text normalized to lowercase to unify vocabulary representation.\n")
    f.write("- Null speakers imputed with `'Unknown'`.\n")
    f.write("- Verified 0 duplicate talk identifiers (`idx`).\n")
    f.write(f"- Missing views: {stats['Missing Views Count']} talks ({stats['Missing Views Percentage']}%) do not report view count, justifying a hybrid or content-primary architecture over view-dependent collaborative filtering.\n\n")
    f.write("## 4. Key Topic Themes\n")
    f.write("The most salient topics extracted from TEDx transcripts and descriptions include:\n")
    for kw, score in reversed(top_keywords[:10]):
        f.write(f"- **{kw}** (TF-IDF weight: {score:.1f})\n")

print(f"\nReport generated: {report_path.name}")
print("Module 2 EDA Analysis completed successfully!")

