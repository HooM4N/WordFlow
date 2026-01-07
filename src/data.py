import os, random
from collections import Counter
from tqdm import tqdm
from typing import Any
from .preprocess import text_preprocess, replace_entities

def get_data(
    data_path: str, 
    min_token_thresh: int = 15, 
    chunk_separator: str = "\n\n",
    shuffle: bool = False,
    random_seed: int = 1212,
    preprocess: bool = True,
    preprocess_kwargs: dict[str, Any] = None,
    add_special_tokens: bool = True,
    print_data_info: bool = False,
    entity_replacement: bool = False,
    entities_to_replace: list[str] = ["PERSON", "GPE", "NORP", "LANGUAGE", "LOC"]
) -> list[str]:
    """
    =====================================================
    == Text Data Pipeline (GiTHUB.com/HooM4N/WordFlow) ==
    =====================================================
    Loads text data from a single .txt file or directory containing multiple .txt files,
    processes each chunk, and returns a filtered list of cleaned texts.
    
    Args:
        - data_path (str): 
            Path to the input .txt file. If a directory path is provided, all '.txt' files
            within that directory will be read and concatenated together. The resulting list
            of text chunks (from all files) will be returned.
        
        - min_token_thresh (int): 
            Minimum token threshold; chunks with fewer tokens are discarded.
        
        - chunk_separator (str): 
            String used to split the text file into chunks.
        
        - preprocess (bool): 
            Whether to apply the `text_cleaner` function to each chunk.

        - preprocess_kwargs (dict): 
            Arguments to pass preprocessor. See src.preprocess.text_preprocess for arg names.
            
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
                if preprocess_kwargs is not None:
                    doc = text_preprocess(doc, **preprocess_kwargs)
                else:
                    doc = text_preprocess(doc)
                    
                if len(doc.split()) >= min_token_thresh:
                    processed.append(f" <bos> {doc} <eos> " if add_special_tokens else doc)
            else:
                processed.append(f" <bos> {doc} <eos> " if add_special_tokens else doc)
                
        print(f"*** file: \"{os.path.basename(data_path)}\" loaded ***")
        
        if shuffle:
            random.seed(random_seed)
            random.shuffle(processed)

        if print_data_info:
            print(" | ".join(f'{k}: {v:,}' for k,v in summarize_data(processed).items()))
        return processed
        
    else:
        assert os.path.exists(data_path), "data dir does not exists!"
        files, cat_data = os.listdir(data_path), []
        for p in files:
            if p.endswith(".txt"):
                cat_data.extend(get_data(os.path.join(data_path, p)))
        print(f"*** {len(files)} files loaded and merged ***")
        if print_data_info:
            print(" | ".join(f'{k}: {v:,}' for k,v in summarize_data(cat_data).items()))
        return cat_data
        
def train_val_split(
    data: list[str], 
    val_ratio: float = 0.15, 
    shuffle: bool = False,
    seed: int = 42
) -> tuple[list[str], list[str]]:
    """
    ===================================================================
    == Train-Val Data Splitting Utility (GiTHUB.com/HooM4N/WordFlow) ==
    ===================================================================
    """
    if shuffle:
        random.seed(seed)
        random.shuffle(data)
        
    up = round(len(data) * (1-val_ratio))
    return data[:up], data[up:]


def summarize_data(data: list[str]) -> None:
    """
    =========================================================
    == Summerize Tokens Count (GiTHUB.com/HooM4N/WordFlow) ==
    =========================================================
    """
    token_counter = Counter()

    for chunk in data:
        tokens = chunk.split()
        token_counter.update(tokens)

    return {
        "total_tokens": sum(token_counter.values()),
        "unique_tokens": len(token_counter),
        "count_of_chunks": len(data),
    }

def flatten(xss: list[list[str]]) -> list[str]:
    """ flattens nested lists """
    return [x for xs in xss for x in xs]