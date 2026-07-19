import torch
import torch.nn as nn
from typing import Callable
from torch.utils.data import DataLoader

def evaluate(
    model: nn.Module, 
    eval_loader: DataLoader, 
    loss_fn: Callable, 
    device: torch.device,
    enable_mixed_precision: bool = True
) -> float:
    """
    Evaluates the model on the validation dataset using Truncated BPTT.
    
    Args:
        model (nn.Module): The WordFlow model.
        eval_loader (DataLoader): DataLoader providing (X, Y) sequence batches.
        loss_fn (Callable): The criterion (e.g., CrossEntropyLoss).
        device (torch.device): Device to perform computations on.
        enable_mixed_precision (bool): Whether to use torch.amp.
        
    Returns:
        float: The average validation loss across the dataset.
        
    WordFlow: Word-Level Language Modeling with RNNs GiTHub.com/HooM4N/WordFlow
    """
    model.eval()
    total_loss = 0.0
    batch_size = eval_loader.dataset.batch_size
    hidden = model.init_hidden(batch_size)
    
    with torch.no_grad():
        for X, Y in eval_loader:
            X, Y = X.to(device), Y.to(device)
            hidden = hidden.detach()
            
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=enable_mixed_precision):
                logits, hidden = model(X, hidden)
                loss = loss_fn(logits, Y)
                
            total_loss += loss.item()
            
    return total_loss / len(eval_loader)