import torch
from torch.utils.data import Dataset

class TruncatedBPTTDataset(Dataset):
    """
    Dataset for Truncated Backpropagation Through Time (BPTT).
    Reshapes a 1D token sequence into continuous parallel batches.

    Args:
        corpus_ids (list[int]): Tokenized corpus as integer IDs.
        batch_size (int): Number of parallel sequences.
        seq_len (int): Length of each sequence slice.

    *WordFlow: Word-Level Language Modeling with RNNs GiTHub.com/HooM4N/WordFlow*
    """
    def __init__(
        self,
        corpus_ids: list[int],
        batch_size: int = 256,
        seq_len: int = 128,
    ):
        full_seq = torch.tensor(corpus_ids, dtype=torch.long)
        
        stream_len = full_seq.size(0) // batch_size
        full_seq = full_seq[:stream_len * batch_size]
        
        self.full_seq = full_seq.view(batch_size, stream_len)

        self.seq_len = seq_len
        self.batch_size = batch_size
        self.stream_len = self.full_seq.size(1) - 1

    def __len__(self) -> int:
        return self.stream_len // self.seq_len

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        start = idx * self.seq_len
        end = start + self.seq_len
        
        return (
            self.full_seq[:, start:end],           # (N, L)
            self.full_seq[:, start + 1 : end + 1], # (N, L)
        )

def bptt_collate(
    batch: list[tuple[torch.Tensor, torch.Tensor]]
) -> tuple[torch.Tensor, torch.Tensor]:
    X, Y = batch[0]
    return X, Y
