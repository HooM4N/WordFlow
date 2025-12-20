import re, os
from typing import Tuple

def get_sherlock_holmes(data_path:str) -> Tuple[str, str]:
    """
    ===============================================
    == Load & preprocess Sherlock Holmes dataset ==
    ===============================================
    """
    with open(os.path.join(data_path, "train.txt"), "r") as f:
        train = f.read()

    with open(os.path.join(data_path, "val.txt"), "r") as f:
        val = f.read()

    return normalize_text_gutenberg(train), normalize_text_gutenberg(val)

    
def normalize_text_gutenberg(text: str) -> str:
    """
    ============================================================
    == Text normalization utility for Project Gutenberg books ==
    ============================================================
    - Lowercases and removes BOM markers 
    - Normalizes quotes, dashes, ellipses, and underscores 
    - Converts newlines to spaces or <eos> markers 
    - Filters unwanted characters
    """
    t = text.lower().replace('\ufeff', '')
    t = t.replace('``', '"').replace("''", '"').replace('…', '...')
    t = t.translate(str.maketrans({
        '“': '"', '”': '"',
        '‘': "'", '’': "'",
        '`': "'", '—': '-',
        '–': '-', '_': ' '
    }))
    t = re.sub(r'[^a-z0-9\s.,;:!?\'"()\-&<>]', '', t)
    return re.sub(r'\s+', ' ', t).strip()
