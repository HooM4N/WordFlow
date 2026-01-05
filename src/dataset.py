import torch
from torch.utils.data import Dataset

#=========================#
#     Dataset Classes     #
#=========================#

### Truncated BPTT Dataset ###

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

### Overlapping Sequences Dataset ###

class SlidingWindowDataset(Dataset):
    """
    ==========================================================
    == Sliding Window Dataset (GiTHUB.com/HoomM4N/WordFlow) ==
    ==========================================================
    """
    def __init__(
        self, 
        corpus_ids: list[int], 
        seq_len: int = 32, 
        min_seq_len: int = 20,
    ):

        self.seq_len = seq_len
        self.min_seq_len = min(min_seq_len, seq_len)
        self._precompute_samples(corpus_ids)

    def _precompute_samples(self, corpus_ids: list):
        inputs, targets = [], []

        for doc in corpus_ids:
            if len(doc) < self.min_seq_len:
                continue

            # Sliding window
            for i in range(len(doc) - self.seq_len):
                x = doc[i : i+self.seq_len]
                y = doc[i+1 : i+1+self.seq_len]

                inputs.append(x)
                targets.append(y)
                
        self.inputs = torch.tensor(inputs, dtype=torch.long)
        self.targets = torch.tensor(targets, dtype=torch.long)
            
    def __len__(self):
        return len(self.inputs)

    def __getitem__(
        self, idx: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            self.inputs[idx], # (L,)
            self.targets[idx] # (L,)
        )
    
    def print_info(self):
        print(f"*** Number of Samples: {self.__len__()} | Sequence Lenght: {self.seq_len} ***")

### Variable Length Sequences Dataset ###

class VariableLengthDataset(Dataset):
    """
    =====================================================================
    == Variable Length Sequences Dataset (GiTHUB.com/HoomM4N/WordFlow) ==
    =====================================================================
    """
    def __init__(
        self, 
        corpus_ids: list[int],
        truncation: bool = True,
        seq_len: int = 32,
        
    ):
        self.corpus_ids = corpus_ids
        self.truncation = truncation
        self.seq_len = seq_len
        
    def __len__(self):
        return len(self.corpus_ids)

    def __getitem__(
        self, idx:int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        doc = self.corpus_ids[idx]
        doc = doc[:self.seq_len+1] if self.truncation else doc
        
        return (
            torch.tensor(doc[:-1], dtype=torch.long), # (L,)
            torch.tensor(doc[1:], dtype=torch.long), # (L,)
        )

    def print_info(self):
        print(f"*** Number of Samples: {self.__len__()} ***")
        
#===========================#
#     Collate Functions     #
#===========================#

def varlen_collate(
    batch: list[tuple[torch.Tensor, torch.Tensor]], 
    pad_token_id: int = 0
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    ==================================================================================
    == Collate Function for Variable Length Sequences (GiTHUB.com/HoomM4N/WordFlow) ==
    ==================================================================================
    """
    lengths = [len(x) for x, _ in batch]

    padded_X = torch.full((len(batch), max(lengths)), pad_token_id, dtype=torch.long)
    padded_Y = torch.full((len(batch), max(lengths)), pad_token_id, dtype=torch.long)

    for i, (x,y) in enumerate(batch):
        padded_X[i, :len(x)] = x
        padded_Y[i, :len(y)] = y

    padding_mask = torch.not_equal(padded_X, pad_token_id).long() # 1 for valid tokens, 0 for paddings
    
    return padded_X, padded_Y, padding_mask

def sliding_collate(
    batch: list[tuple[torch.Tensor, torch.Tensor]]
) -> tuple[torch.Tensor, torch.Tensor, None]: 
    Xs, Ys = zip(*batch)
    return torch.stack(Xs, 0), torch.stack(Ys, 0), None

def bptt_collate(
    batch: list[tuple[torch.Tensor, torch.Tensor]]
) -> tuple[torch.Tensor, torch.Tensor, None]: 
    X, Y = batch[0]
    return X, Y, None
    
#===============#
#     Utils     #
#===============#

def flatten(xss: list[list[str]]) -> list[str]:
    """
    Flattens nested lists
    """
    return [x for xs in xss for x in xs]