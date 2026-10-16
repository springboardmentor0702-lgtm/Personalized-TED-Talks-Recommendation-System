import pandas as pd 
import re
from pathlib import Path 

BASE_DIR = Path(__file__).resolve().parent
INPUT_FILE = BASE_DIR / "tedx_dataset.csv"
OUTPUT_FILE = BASE_DIR / "ted_talks_preprocessed.csv"

df = pd.read_csv(INPUT_FILE)

print("Original dataset shape:", df.shape)
print("\nMissing values:")
print(df.isnull().sum())

df = df.drop_duplicates(subset="idx")
df = df.dropna(subset=["idx", "title"]).copy()

def clean_text(text):
    if pd.isna(text):
        return ""

    text = str(text).lower()
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text

df["title"] = df["title"].apply(clean_text)
df["details"] = df["details"].apply(clean_text)

df["main_speaker"] = (
    df["main_speaker"].fillna("Unknown").astype(str).str.strip()
)

df["url"] = df["url"].fillna("")

df["posted_date"] = pd.to_datetime(
    df["posted"].astype("string").str.replace(
        r"(?i)^posted\s+", "", regex=True
    ),
    errors="coerce"
)

df["num_views"] = pd.to_numeric(
    df["num_views"], errors="coerce"
)

df["combined_text"] = (
    df["title"] + " " + df["details"]
).str.strip()

df = df[df["combined_text"].str.len() > 0].copy()
df = df.reset_index(drop=True)

df.to_csv(OUTPUT_FILE, index=False)

print("\nPreprocessed dataset shape:", df.shape)
print("\nCleaned data preview:")
print(df[["title", "main_speaker", "combined_text"]].head())

print("\nSaved to:", OUTPUT_FILE)
print("Preprocessing completed successfully!")