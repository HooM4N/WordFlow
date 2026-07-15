import torch
from torch.utils.data import Dataset

class TruncatedBPTTDataset(Dataset):
    """
    Dataset for Truncated Backpropagation Through Time (BPTT).
    
    Transforms a 1D sequence of tokenized corpus IDs into `batch_size` 
    continuous, parallel streams. Each call to `__getitem__` returns a 
    slice of length `seq_len` moving sequentially forward through the text.
    
    Args:
        corpus_ids (list[int]): Tokenized corpus as a list of integer IDs.
        batch_size (int, optional): Number of parallel streams. Defaults to 256.
        seq_len (int, optional): Length of the sequence per stream. Defaults to 128.
        
    WordFlow: Word-Level Language Modeling with RNNs GiTHub.com/HooM4N/WordFlow
    """

    def __init__(
        self,
        corpus_ids: list[int],
        batch_size: int = 256,
        seq_len: int = 128,
    ):
        full_seq = torch.tensor(corpus_ids, dtype=torch.long)
        
        # Trim sequence length to be evenly divisible by batch_size
        stream_len = full_seq.size(0) // batch_size
        full_seq = full_seq[:stream_len * batch_size]
        
        # Reshape into (batch_size, stream_len)
        self.full_seq = full_seq.view(batch_size, stream_len)

        self.seq_len = seq_len
        self.batch_size = batch_size
        self.stream_len = self.full_seq.size(1) - 1 # <-- Changed to self.full_seq!

    def __len__(self) -> int:
        return self.stream_len // self.seq_len

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        start = idx * self.seq_len
        end = start + self.seq_len
        
        # Returns input (X) and shifted target (Y)
        return (
            self.full_seq[:, start:end],           # X
            self.full_seq[:, start + 1 : end + 1], # Y
        )


def bptt_collate(
    batch: list[tuple[torch.Tensor, torch.Tensor]]
) -> tuple[torch.Tensor, torch.Tensor]:
    X, Y = batch[0]
    return X, Y