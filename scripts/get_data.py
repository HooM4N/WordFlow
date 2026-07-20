import os
import pandas as pd

TINY_STORIES_URL = (
    "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/"
    "data/train-00000-of-00004-2d5a1467fff1081b.parquet?download=true"
)
COUNT_OF_STORIES = 16_000
VAL_SPLIT_RATIO = 0.1
RANDOM_SEED = 1212

RAW_SAVE_PATH = "data/raw"
PROCESSED_SAVE_PATH = "data/processed"

os.makedirs(RAW_SAVE_PATH, exist_ok=True)
os.makedirs(PROCESSED_SAVE_PATH, exist_ok=True)

def main():
    raw_file_path = os.path.join(RAW_SAVE_PATH, "tiny_stories.parquet")
    
    if not os.path.exists(raw_file_path):
        print("Downloading raw parquet file...")
        df = pd.read_parquet(TINY_STORIES_URL)
        df.to_parquet(raw_file_path, index=False)
        print(f"Raw data saved to {raw_file_path}")
    else:
        print("Loading raw parquet file from disk...")
        df = pd.read_parquet(raw_file_path)

    df = df.iloc[:COUNT_OF_STORIES]

    print("Splitting dataset into train and validation sets...")
    
    df = df.sample(frac=1.0, random_state=RANDOM_SEED).reset_index(drop=True)
    
    split_idx = int(len(df) * (1 - VAL_SPLIT_RATIO))
    
    train_df = df.iloc[:split_idx]
    val_df = df.iloc[split_idx:]

    train_corpus = " <eos> ".join(train_df.text)
    val_corpus = " <eos> ".join(val_df.text)

    train_path = os.path.join(PROCESSED_SAVE_PATH, "train.txt")
    val_path = os.path.join(PROCESSED_SAVE_PATH, "val.txt")
    
    with open(train_path, "w", encoding="utf-8") as f:
        f.write(train_corpus)
        
    with open(val_path, "w", encoding="utf-8") as f:
        f.write(val_corpus)

    print(f"Success! Train data saved to: {train_path}")
    print(f"Success! Val data saved to:   {val_path}")

if __name__ == "__main__":
    main()