import os
import pandas as pd

TINY_STORIES_URL = (
    "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/"
    "data/train-00000-of-00004-2d5a1467fff1081b.parquet?download=true"
)
COUNT_OF_STORIES = 15_000
RAW_SAVE_PATH = "data/raw"
PROCESSED_SAVE_PATH = "data/processed"

# Ensure output directories exist
os.makedirs(RAW_SAVE_PATH, exist_ok=True)
os.makedirs(PROCESSED_SAVE_PATH, exist_ok=True)

# 1. Download and save the raw parquet file
raw_file_path = os.path.join(RAW_SAVE_PATH, "tiny_stories.parquet")
if not os.path.exists(raw_file_path):
    print("Downloading raw parquet file...")
    df = pd.read_parquet(TINY_STORIES_URL)
    df.to_parquet(raw_file_path, index=False)
    print(f"Raw data saved to {raw_file_path}")
else:
    df = pd.read_parquet(raw_file_path)

# 2. Keep only the first COUNT_OF_STORIES stories
df = df.iloc[:COUNT_OF_STORIES]

# 3. Build the corpus with <eos> tokens
corpus = " <eos> ".join(df.text)

# 4. Save the processed text file
processed_file = os.path.join(PROCESSED_SAVE_PATH, "tiny_stories.txt")
with open(processed_file, "w", encoding="utf-8") as f:
    f.write(corpus)

print(f"Data successfully downloaded, filtered, and saved to {processed_file}")