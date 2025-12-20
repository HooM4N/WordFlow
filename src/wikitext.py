import re, unicodedata
from typing import Tuple
from datasets import load_dataset

def get_wikitext(dataset_name_or_path:str) -> Tuple[str, str, str]:
    """
    =====================================================
    == Returns filtered & preprocessed WikiText splits ==
    =====================================================
    """
    dataset = load_dataset(dataset_name_or_path)
    dataset = dataset.filter(is_valid_row)
    dataset = dataset.map(wikitext_preprocessor)
    
    train_corpus = " ".join(f"{t} <eos>" for t in dataset["train"]["clean_text"])
    val_corpus = " ".join(f"{t} <eos>" for t in dataset["validation"]["clean_text"])
    return train_corpus, val_corpus

    
def is_valid_row(row, len_thresh=16):
    """
    ==========================================
    == Masks valid rows in WikiText dataset ==
    ==========================================
    """
    text = row["text"].strip()
    if text == "": # empty rows
        return False
    if re.match(r"^=+.*=+$", text): # article titles or subtitles
        return False
    if len(text.split()) < len_thresh: # rows with splits shorter than thresh
        return False
    return True


def wikitext_preprocessor(row):
    """
    ===========================================================
    == Aggressive preprocessing utility for WikiText dataset ==
    ===========================================================
    Functions:
    - Lowercase + ASCII normalize
    - Fix WikiText punctuation artifacts
    - Bucket numbers, handle ordinals
    - Clean whitespace, strip noise
    """
    t = row["text"]
    t = unicodedata.normalize("NFKD", t.lower()).encode("ascii", "ignore").decode("ascii") # lowercase + unicode → ascii
    artifact_map = {"@-@": "-", "@.@": ".", "@,@": ","}
    
    for pat, repl in artifact_map.items(): # fix wikitext artifacts
        t = t.replace(pat, repl)
    
    t = re.sub(r"[–—−]", "-", t) # normalize dashes
    t = re.sub(r"\([^)]*\)", " ", t) # remove parentheses and content
    roman_map = {
    "i": "1", "ii": "2", "iii": "3", "iv": "4", "v": "5",
    "vi": "6", "vii": "7", "viii": "8", "ix": "9", "x": "10"
    }
    t = re.sub(r"\b(i|ii|iii|iv|v)\b", lambda m: roman_map[m.group()], t) # roman numerals i–v → digits
    t = re.sub(r"\.{3,}", ".", t) # normalize ellipses (3+ dots) → "."
    t = re.sub(r"[^a-z0-9\.\,\;\:\!\?\'\-\s]", " ", t) # keep essential punctuation only
    t = re.sub(r"\b\d+\s*,\s*\d+\b", "<num>", t) # collapse comma-formatted numbers → "<num>"
    t = re.sub(r"\b(\d+)(st|nd|rd|th)\b", r"\1 th", t) # handle ordinals like 19th, 21st → "19 th"
    
    def num_bucket(m):
        n = int(m.group())
        if 0 <= n <= 9:
            return str(n)
        if 1500 <= n <= 2099:
            return "<year>"
        return "<num>"

    t = re.sub(r"\d+", num_bucket, t) # number bucketing
    t = re.sub(r"\s+", " ", t).strip() # whitespace normalization
    return {"clean_text": t}