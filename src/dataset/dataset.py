import torch
from torch.utils.data import Dataset

class TruncatedBPTTDataset(Dataset):
    """
    =========================================================
    == Truncated BPTT Dataset (GiTHUB.com/HooM4N/WordFlow) ==
    =========================================================
    """
    def __init__(
        self, 
        corpus_ids: list[int], 
        batch_size: int = 256, 
        seq_len: int = 128
    ):
        full_seq = torch.tensor(flatten(corpus_ids), dtype=torch.long)
        # trim to multiple of batch_size
        stream_len = full_seq.size(0) // batch_size
        full_seq = full_seq[:stream_len * batch_size]
        full_seq = full_seq.view(batch_size, stream_len)

        self.seq_len = seq_len
        self.batch_size = batch_size
        self.full_seq = full_seq
        self.stream_len = full_seq.size(1) - 1

    def __len__(self):
        return self.stream_len // self.seq_len # number of batches

    def __getitem__(
        self, idx: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        start = idx * self.seq_len
        end = start + self.seq_len
        return (
            self.full_seq[:, start:end], # (N,L)
            self.full_seq[:, start+1:end+1] # (N,L)
        )

    def print_info(self):
        print(f"*** Batch Size: {self.batch_size} | Sequence Len: {self.seq_len} ***")
        print(f"*** Stream Lenght: {self.stream_len} | Number of Batches: {self.__len__()} ***")