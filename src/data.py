import os
import random
from collections import Counter
from tqdm import tqdm
from .preprocess import text_cleaner, replace_entities

def get_data(
    data_path: str, 
    min_token_thresh: int = 15, 
    chunk_separator: str = "\n\n",
    preprocess: bool = True,
    add_special_tokens: bool = True,
    entity_replacement: bool = False,
    entities_to_replace: list[str] = ["PERSON", "GPE", "NORP", "LANGUAGE", "LOC"]
) -> list[str]:
    """
    =============================================================
    == Read & Prepare Text Data (GiTHUB.com/HooM4N/CausalLSTM) ==
    =============================================================
    Args:
        - data_path (str): 
            Path to the input `.txt` file.
        
        - min_token_thresh (int): 
            Minimum token threshold; chunks with fewer tokens are discarded.
        
        - chunk_separator (str): 
            String used to split the text file into chunks.
        
        - preprocess (bool): 
            Whether to apply the `text_cleaner` function to each chunk.
        
        - entity_replacement (bool):
            Whether to replace named entities using spaCy 
            (default model: `en_core_web_sm`).
        
        - entities_to_replace (list): 
            List of entity types to replace. 
            See the `replace_entities` docstring for the full list of supported entities.
    """
    if data_path.endswith(".txt"):
        assert os.path.exists(data_path), "data file does not exists!"
        
        with open(data_path, "r", encoding="utf-8") as f:
            raw_chunks = f.read().split(chunk_separator)
            
        processed = []
        for doc in tqdm(raw_chunks, desc="Processing Data"):
            if entity_replacement:
                doc = replace_entities(doc, entities_to_replace)
            if preprocess:
                doc = text_cleaner(doc)
                if len(doc.split()) >= min_token_thresh:
                    processed.append(f" <bos> {doc} <eos> " if add_special_tokens else doc)
            else:
                processed.append(f" <bos> {doc} <eos> " if add_special_tokens else doc)
                
        print(f"*** file: \"{os.path.basename(data_path)}\" loaded ***")
        summarize_data(processed)
        return processed
    else:
        assert os.path.exists(data_path), "data dir does not exists!"
        files, cat_data = os.listdir(data_path), []
        for p in files:
            if p.endswith(".txt"):
                cat_data.extend(get_data(os.path.join(data_path, p)))
        print(f"*** {len(files)} files loaded and merged ***")
        summarize_data(cat_data)
        return cat_data
        
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
        f"{len(data):,} chunks | ***"
    )

def flatten(xss: list[list[str]]) -> list[str]:
    """ flattens nested lists """
    return [x for xs in xss for x in xs]