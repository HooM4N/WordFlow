import time
import numpy as np
from typing import List

def get_glove_embeddings(
    glove_txt_path: str, 
    glove_dim: int, 
    vocab: List[str]
) -> np.ndarray:
    """
    ===============================================
    == Load GloVe Embeddings (GiTHUB.com/HooM4N) ==
    ===============================================
    - Loads GloVe embeddings for a given vocab list
    - Preserves order of vocab in embedding_matrix
    - Retruns OOV words as zero rows
    """
    word2idx = {w:i for i,w in enumerate(vocab)}
    embedding_matrix = np.zeros((len(vocab), glove_dim), dtype=np.float32)
    start, found = time.time(), 0
    with open(glove_txt_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            word = parts[0]
            if len(parts[1:]) != glove_dim:
                continue
            if word in word2idx:
                embedding_matrix[word2idx[word]] = np.array(parts[1:], dtype=np.float32)
                found += 1
    
    print(f"*** {found:,} words added to embedding matrix ***")
    print(f"*** {len(vocab) - found:,} out-of-vocab words ***")
    print(f"*** time taken: {time.time() - start:.2f}s ***")
    return embedding_matrix