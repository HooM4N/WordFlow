import os
from typing import Tuple
from datasets import load_dataset
from .preprocess import is_valid_row, wikitext_preprocessor, normalize_text_gutenberg

def get_data(data_name: str, data_dir: str) -> Tuple[str, str]:
    """
    =================================================================
    == Load dataset from given name (GiTHUB.com/HooM4N/CausalLSTM) ==
    =================================================================
    Available datasets names:
        - "wikitext": preprocessed version of WikiText-2
        - "sherlock_holmes": concatanation of four Sir Arthur Conan Doyle works:
            * adventures_of_sherlock_holmes
            * hound_of_baskervill
            * memories_of_sherlock_holmes, 
            * return_of_sherlock_holmes
    """
    assert data_name in ["wikitext", "sherlock_holmes"]
    if data_name == "wikitext":
        return get_wikitext(data_dir)
    elif data_name == "sherlock_holmes":
        return get_sherlock_holmes(data_dir)
        
def get_wikitext(dataset_name_or_dir:str) -> Tuple[str, str, str]:
    """
    =====================================================
    == Returns filtered & preprocessed WikiText splits ==
    =====================================================
    """
    dataset = load_dataset(dataset_name_or_dir)
    dataset = dataset.filter(is_valid_row)
    dataset = dataset.map(wikitext_preprocessor)
    
    train_corpus = " ".join(f"{t} <eos> " for t in dataset["train"]["clean_text"])
    val_corpus = " ".join(f"{t} <eos> " for t in dataset["validation"]["clean_text"])
    return train_corpus, val_corpus

def get_sherlock_holmes(data_dir:str) -> Tuple[str, str]:
    """
    ===============================================
    == Load & preprocess Sherlock Holmes dataset ==
    ===============================================
    """
    with open(os.path.join(data_dir, "train.txt"), "r") as f:
        train = f.read()

    with open(os.path.join(data_dir, "val.txt"), "r") as f:
        val = f.read()

    return normalize_text_gutenberg(train), normalize_text_gutenberg(val)