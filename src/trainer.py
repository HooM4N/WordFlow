import json, os, math
import torch
import torch.nn as nn
from collections import deque
from datetime import datetime
from tqdm import tqdm
from typing import Callable, Dict

def trainer(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    config: Dict,
    paths: Dict,
    device: torch.device, 
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    detach_hidden: Callable, 
    train_ds: torch.utils.data.Dataset,
    val_ds: torch.utils.data.Dataset,
    loss_fn: Callable, 
    checkpoint_path: str = None
) -> torch.nn.Module:
    """
    ===========================================================
    == Trainer for CausalLSTM (GiTHUB.com/HooM4N/CausalLSTM) ==
    ===========================================================
    Features:
        - Mixed Precision Training
        - Experiment Tracking: save per-run configs, logs & best checkpoint
        - Early Stopping
        - Gradient Clipping
        - Restore Best Model
        - Model Checkpoint
        - Resume Training from Checkpoint
    """
    if checkpoint_path:
        model.load_state_dict(
            torch.load(checkpoint_path, map_location=device, weights_only=True)
            )
        print(f"*** Continue training from checkpoint ***")
        
    train_logs = {"train_loss":[] , "val_loss":[] , "val_metric":[], "lr":[], "epoch_time": []}
    model.train()
    scaler = torch.amp.GradScaler(enabled = config["enable_mixed_precision"])
    best_loss, es_counter, best_state = float('inf'), 0, None
    last_checkpoints = deque(maxlen=3)
    run_name = datetime.now().strftime("run_%m-%d_%H-%M")
    print(f"*** Starting run {run_name} ***")
    
    for epoch in range(config["n_epochs"]):
        epoch_start = datetime.now()
        model.train()
        total_loss = 0.0
        hidden = model.init_hidden(config["batch_size"])

        for idx in tqdm(range(len(train_ds)), desc=f"Epoch {epoch+1}/{config["n_epochs"]}"):
            X, Y = train_ds[idx]
            X, Y = X.to(device), Y.to(device)
            optimizer.zero_grad(set_to_none=True)
            hidden = detach_hidden(hidden)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled = config["enable_mixed_precision"]):
                logits, hidden = model(X, hidden)
                loss = loss_fn(logits, Y)
            total_loss += loss.item()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), config["grad_clip_norm"])
            scaler.step(optimizer)
            scaler.update()

        # logger
        train_logs["train_loss"].append(total_loss / len(train_ds))
        val_loss = evaluate(val_ds)
        train_logs["val_loss"].append(val_loss)
        train_logs["val_metric"].append(math.exp(val_loss))
        train_logs["lr"].append(optimizer.param_groups[0]['lr'])
        train_logs["epoch_time"].append((datetime.now() - epoch_start).total_seconds())

        print( 
            f"\r Epoch {epoch + 1}/{config['n_epochs']}, " 
            f"train loss: {train_logs['train_loss'][-1]:.4f}, " 
            f"val loss: {train_logs['val_loss'][-1]:.4f}, " 
            f"val perplexity: {train_logs['val_metric'][-1]:.4f}, " 
            f"lr: {train_logs['lr'][-1]}, " 
            f"epoch time: {train_logs['epoch_time'][-1]:.2f}s"
        )

        # lr scheduler
        scheduler.step(val_loss)

        # checkpoints
        ckpnt_path =  os.path.join(paths["models"], f"CausalLSTM_ckpnt_{epoch+1}.pt") 
        torch.save(model.state_dict(), ckpnt_path)
        last_checkpoints.append(ckpnt_path)

        # track best model
        if val_loss < best_loss - config["early_stopping_epsilon"]:
            best_loss = val_loss
            best_state = model.state_dict()
            es_counter = 0
        else:
            es_counter += 1

        # early stopping
        if es_counter >= config["early_stopping_patience"]:
            print(f"*** Early Stopping triggered at epoch: {epoch+1} ***")
            break
        
    # restore best model
    if best_state:
        model.load_state_dict(best_state)

    # save artifacts
    run_dir = os.path.join(paths["models"], run_name)
    os.makedirs(run_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(run_dir, f"CausalLSTM_ckpnt.pt") )
    with open(os.path.join(run_dir, f"config.json"), "w") as f:
        json.dump(config, f, indent=2)
    with open(os.path.join(run_dir, f"training_logs.json"), "w") as f:
        json.dump(train_logs, f, indent=2)
    
    print(f"*** Run {run_name} completed. Artifacts saved in: {run_dir} ***")
    return model


@torch.no_grad()
def evaluate(
    model: torch.nn.Module, 
    eval_dataset: torch.utils.data.Dataset, 
    loss_fn: Callable, 
    device: torch.device, 
    config: Dict, 
    detach_hidden: Callable,
    disable_progress_bar: bool = False
) -> float:
    """
    ======================================================================
    == Evaluator function for CausalLSTM (GiTHUB.com/HooM4N/CausalLSTM) ==
    ======================================================================
    """
    model.eval()
    total_loss = 0.0
    hidden = model.init_hidden(config["batch_size"])
    for idx in tqdm(range(len(eval_dataset)), disable = disable_progress_bar):
        X, Y = eval_dataset[idx]
        X, Y = X.to(device), Y.to(device)
        hidden = detach_hidden(hidden)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=config["enable_mixed_precision"]):
            logits, hidden = model(X, hidden)
            total_loss += loss_fn(logits, Y).item()
    return total_loss / len(eval_dataset)