import torch
from torch.utils.data import Dataset

class TruncatedBPTTDataset(Dataset):

    def __init__(
        self,
        corpus_ids: list[int],
        batch_size: int = 256,
        seq_len: int = 128,
    ):

        full_seq = torch.tensor(corpus_ids, dtype=torch.long)
        # trim to multiple of batch_size
        stream_len = full_seq.size(0) // batch_size
        full_seq = full_seq[:stream_len * batch_size]
        full_seq = full_seq.view(batch_size, stream_len)

        self.seq_len = seq_len
        self.batch_size = batch_size
        self.full_seq = full_seq
        self.stream_len = full_seq.size(1) - 1

    def __len__(self) -> int:
        return self.stream_len // self.seq_len

    def __getitem__(
        self, idx: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        start = idx * self.seq_len
        end = start + self.seq_len
        return (
            self.full_seq[:, start:end],       # input
            self.full_seq[:, start + 1 : end + 1],  # target
        )


def bptt_collate(
    batch: list[tuple[torch.Tensor, torch.Tensor]]
) -> tuple[torch.Tensor, torch.Tensor, None]:
    X, Y = batch[0]
    return X, Y, None