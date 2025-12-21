import time
import numpy as np
from typing import List, Dict

def get_glove_embeddings(
    glove_txt_path: str,
    glove_dim: int,
    vocab: List[str],
    pad_token: str = "<pad>",
    unk_token: str = "<unk>",
    anchor_map: Dict[str, str] = {
        "<eos>": ".",
        "<year>": "year",
        "<num>": "number",
    },
    noise_scale: float = 0.01,
    seed: int = 42,
) -> np.ndarray:
    """
    =================================================================
    == Loads Pre-trained GloVe Word Embeddings (GiTHUB.com/HooM4N) ==
    =================================================================
    - Keep PAD token zero
    - Anchor special tokens semantically
    - Initialize OOV tokens with UNK + noise
    - Preserve mean/std of GloVe vectors
    - Deterministic random initialization
    """

    rng = np.random.default_rng(seed)
    word2idx = {w: i for i, w in enumerate(vocab)}

    emb = np.zeros((len(vocab), glove_dim), dtype=np.float32)
    found_mask = np.zeros(len(vocab), dtype=bool)

    start, found = time.time(), 0

    with open(glove_txt_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip().split()
            if len(parts) != glove_dim + 1:
                continue # corrupted words
            word = parts[0]
            if word in word2idx:
                idx = word2idx[word]
                emb[idx] = np.asarray(parts[1:], dtype=np.float32)
                found_mask[idx] = True
                found += 1

    print(f"*** {found:,} words added to embedding matrix ***")
    print(f"*** {len(vocab) - found:,} out-of-vocab words ***")
    print(f"*** time taken: {time.time() - start:.2f}s ***")

    # compute stats of found vectors
    real_embs = emb[found_mask]
    mean = real_embs.mean(axis=0)
    std = real_embs.std(axis=0)

    # check if <unk> is in glove, use it as base
    if unk_token in word2idx and found_mask[word2idx[unk_token]]:
        unk_vec = emb[word2idx[unk_token]]
    else:
        unk_vec = mean

    # semantic anchoring for special tokens
    if anchor_map is not None:
        for special, anchor in anchor_map.items():
            if special not in word2idx:
                continue
            sidx = word2idx[special]
            if found_mask[sidx]:
                continue  # already has GloVe
            if anchor in word2idx and found_mask[word2idx[anchor]]:
                base = emb[word2idx[anchor]]
                noise = rng.normal(0, std * noise_scale, glove_dim)
                emb[sidx] = base + noise
                found_mask[sidx] = True

    # remaining oovs except padding token
    for i, word in enumerate(vocab):
        if word == pad_token:
            continue 
        if not found_mask[i]:
            noise = rng.normal(0, std * noise_scale, glove_dim)
            emb[i] = unk_vec + noise

    return emb