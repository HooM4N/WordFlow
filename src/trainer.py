import os
import json
import math
import logging
from datetime import datetime
from typing import Callable

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .evaluate import evaluate
from .tokenizer import Tokenizer
from .config import WordFlowConfig

logger = logging.getLogger(__name__)

def trainer(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    loss_fn: Callable,
    train_loader: DataLoader,
    val_loader: DataLoader | None,
    config: WordFlowConfig,
    device: torch.device,
    run_dir: str,
    tokenizer: Tokenizer
) -> nn.Module:
    """
    Main training loop for Truncated BPTT Word-Level Modeling.
    
    Features:
        - Mixed Precision (AMP)
        - Gradient Clipping
        - Cosine Annealing LR Scheduling
        - Run artifact tracking (JSON logs, Model configs)
        - Early stopping and graceful KeyboardInterrupt recovery
        
    WordFlow: Word-Level Language Modeling with RNNs GiTHub.com/HooM4N/WordFlow
    """
    scaler = torch.amp.GradScaler(enabled=config.train.enable_mixed_precision)
    train_logs = {"train_loss": [], "val_loss": [], "val_perplexity": [], "lr": [], "epoch_time": []}
    
    best_loss = float('inf')
    es_counter = 0
    best_epoch = 0
    best_ckpnt_path = os.path.join(run_dir, "checkpoint_best.pt")
    last_ckpnt_path = os.path.join(run_dir, "checkpoint_last.pt")

    logger.info(f"Starting training run in {run_dir}")
    
    try:
        for epoch in range(config.train.n_epochs):
            epoch_start = datetime.now()
            model.train()
            total_loss = 0.0
            
            hidden = model.init_hidden(train_loader.dataset.batch_size)

            for X, Y in train_loader:
                X, Y = X.to(device), Y.to(device)
                optimizer.zero_grad(set_to_none=True)
                
                hidden = hidden.detach()
                
                with torch.autocast(
                    device_type=device.type, 
                    dtype=torch.float16, 
                    enabled=config.train.enable_mixed_precision
                    ):
                    logits, hidden = model(X, hidden)
                    loss = loss_fn(logits, Y)
                    
                total_loss += loss.item()
                scaler.scale(loss).backward()
                
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), config.train.grad_clip_norm)
                
                scaler.step(optimizer)
                scaler.update()

            # End of epoch calculations
            avg_train_loss = total_loss / len(train_loader)
            train_logs["train_loss"].append(avg_train_loss)
            train_logs["lr"].append(optimizer.param_groups[0]['lr'])
            
            log_msg = f"Epoch {epoch + 1}/{config.train.n_epochs} | Train Loss: {avg_train_loss:.4f}"

            if val_loader:
                val_loss = evaluate(
                    model, val_loader, loss_fn, device, config.train.enable_mixed_precision
                )
                train_logs["val_loss"].append(val_loss)
                train_logs["val_perplexity"].append(math.exp(val_loss))
                log_msg += f" | Val Loss: {val_loss:.4f} | Val Perplexity: {math.exp(val_loss):.4f}"
            
            train_logs["epoch_time"].append((datetime.now() - epoch_start).total_seconds())
            logger.info(log_msg)

            # Checkpointing
            torch.save(model.state_dict(), last_ckpnt_path)

            scheduler.step()

            # Early Stopping and Best Model tracking
            if val_loader:
                if val_loss < best_loss:
                    best_loss = val_loss
                    best_epoch = epoch + 1
                    torch.save(model.state_dict(), best_ckpnt_path)
                    es_counter = 0
                else:
                    es_counter += 1
        
                if es_counter >= config.train.early_stopping_patience:
                    logger.info(f"Early Stopping triggered at epoch {epoch + 1}.")
                    break

    except KeyboardInterrupt:
        logger.warning("Training interrupted by user (KeyboardInterrupt).")

    finally:
        # Wrap up training gracefully
        if os.path.exists(best_ckpnt_path):
            model.load_state_dict(torch.load(best_ckpnt_path, map_location=device, weights_only=True))
            logger.info(f"Restored best model from epoch {best_epoch}.")

        # Save final artifacts
        with open(os.path.join(run_dir, "training_logs.json"), "w") as f:
            json.dump(train_logs, f, indent=2)
            
        tokenizer.save(os.path.join(run_dir, "tokenizer.json"))
        logger.info(f"Run completed. All artifacts saved in: {run_dir}")
    
    return model