import torch
from torch.utils.data import Dataset

class TruncatedBPTTDataset(Dataset):
    """
    ===========================================================
    == Truncated BPTT Dataset (GiTHUB.com/HooM4N/CausalLSTM) ==
    ===========================================================
    - Takes full corpus token IDs as a list and converts to a torch Tensor.
    - Splits full sequence into batch_size streams, with hidden states carried across streams.
    - __len__ returns the number of batches.
    - __getitem__ returns X: (batch_size, seq_len) and Y: one timestep shifted to the right.
    """
    def __init__(self, corpus_tokens_ids: list, batch_size: int =256, seq_len: int = 128):
        full_seq = torch.tensor(corpus_tokens_ids, dtype=torch.long)
        # trim to multiple of batch_size
        stream_len = full_seq.size(0) // batch_size
        full_seq = full_seq[:stream_len * batch_size]
        full_seq = full_seq.view(batch_size, stream_len)

        self.seq_len = seq_len
        self.batch_size = batch_size
        self.full_seq = full_seq
        self.stream_len = full_seq.size(1) - 1

    def __len__(self):
        """Returns number of batches"""
        return self.stream_len // self.seq_len

    def __getitem__(self, idx):
        start = idx * self.seq_len
        end = start + self.seq_len
        X = self.full_seq[:, start:end]
        Y = self.full_seq[:, start+1:end+1]
        return X, Y

    def get_info(self):
        print(f"*** batch size: {self.batch_size} | sequence len: {self.seq_len} ***")
        print(f"*** stream lenght: {self.stream_len} | number of batches: {self.__len__()} ***")