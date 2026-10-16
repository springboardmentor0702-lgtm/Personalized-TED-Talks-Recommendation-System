
import pandas as pd
from pathlib import Path

# Find CSV files in the project folder and its subfolders
base_dir = Path(__file__).resolve().parent
csv_files = list(base_dir.rglob("*.csv"))

if not csv_files:
    print("No CSV files found in the project folder.")
else:
    for file in csv_files:
        print("\n" + "=" * 60)
        print("File:", file.relative_to(base_dir))

        try:
            df = pd.read_csv(file)

            print("Dataset shape:", df.shape)
            print("\nColumn names:")
            print(df.columns.tolist())

            print("\nFirst 3 rows:")
            print(df.head(3).to_string(index=False))

            print("\nMissing values per column:")
            print(df.isnull().sum().to_string())

        except Exception as e:
            print("Could not read this file:", e)
