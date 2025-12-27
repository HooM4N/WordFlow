import os
import random
from collections import Counter
from tqdm import tqdm
from .preprocess import text_cleaner, replace_entities

def get_data(
    data_path: str, 
    min_token_thresh: int = 15, 
    chunk_separator: str = "\n\n",
    do_replace_entities: bool = False
) -> list[str]:
    """
    =============================================================
    == Read & Prepare Text Data (GiTHUB.com/HooM4N/CausalLSTM) ==
    =============================================================
    """
    assert os.path.exists(data_path), "data file does not exists!"
    assert data_path.endswith(".txt"), "expects txt file as data_path"
    
    with open(data_path, "r", encoding="utf-8") as f:
        raw_chunks = f.read().split(chunk_separator)
        
    processed = []
    for doc in tqdm(raw_chunks, desc="Processing Data"):
        if do_replace_entities:
            doc = replace_entities(doc)
        doc = text_cleaner(doc)
        if len(doc.split()) >= min_token_thresh:
            processed.append(f" <bos> {doc} <eos> ")
    print(f"*** file: \"{os.path.basename(data_path)}\" loaded ***")
    return processed
        
def train_val_split(
    data: list[str], 
    val_ratio: float = 0.15, 
    shuffle: bool = False,
    seed: int = 42
) -> tuple[list[str], list[str]]:
    """
    =====================================================================
    == Train-Val Data Splitting Utility (GiTHUB.com/HooM4N/CausalLSTM) ==
    =====================================================================
    """
    if shuffle:
        random.seed(seed)
        random.shuffle(data)
        
    up = round(len(data) * (1-val_ratio))
    return data[:up], data[up:]


def summarize_data(data: list[str]) -> None:
    """
    ===========================================================
    == Summerize Tokens Count (GiTHUB.com/HooM4N/CausalLSTM) ==
    ===========================================================
    """
    token_counter = Counter()

    for chunk in data:
        tokens = chunk.split()
        token_counter.update(tokens)

    print(
        f"*** | {sum(token_counter.values()):,} tokens | "
        f"{len(token_counter):,} unique tokens | "
        f"{len(data)} chunks | ***"
    )

def flatten(xss: list[list[str]]) -> list[str]:
    """ flattens nested lists """
    return [x for xs in xss for x in xs]