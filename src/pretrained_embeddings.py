import time
import numpy as np

def get_glove_embeddings(
    glove_txt_path: str,
    glove_dim: int,
    vocab: list[str],
) -> np.ndarray:
    """
    ================================================================
    == Load Pre-trained GloVe Word Embeddings (GiTHUB.com/HooM4N) ==
    ================================================================
    - Returns OOV words as zero-row
    """
    word2idx = {w: i for i, w in enumerate(vocab)}
    emb = np.zeros((len(vocab), glove_dim), dtype=np.float32)
    start, found = time.time(), 0

    with open(glove_txt_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != glove_dim + 1: # ignore corrupted words
                continue 
            word = parts[0]
            if word in word2idx:
                idx = word2idx[word]
                emb[idx] = np.asarray(parts[1:], dtype=np.float32)
                found += 1

    print(f"*** {found:,} words added to embedding matrix ***")
    print(f"*** {len(vocab) - found:,} out-of-vocab words ***")
    print(f"*** time taken: {time.time() - start:.2f}s ***")
    return emb