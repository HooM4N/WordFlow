import torch

def bptt_collate(
    batch: list[tuple[torch.Tensor, torch.Tensor]]
) -> tuple[torch.Tensor, torch.Tensor, None]: 
    X, Y = batch[0]
    return X, Y, None