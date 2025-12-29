import os
import json
import math
from tqdm import tqdm
from typing import Callable
from datetime import datetime

import torch
import torch.nn as nn

from .tokenizer import Tokenizer
from .config import write_config

#=======================#
#     Training Loop     #
#=======================#

def trainer(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    config: dict[str, int | float | str],
    device: torch.device, 
    scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau,
    detach_hidden: Callable, 
    train_loader: torch.utils.data.DataLoader,
    loss_fn: Callable,
    tokenizer: Tokenizer,
    val_loader: torch.utils.data.DataLoader = None,
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
    if config["checkpoint_path"] is not None:
        model.load_state_dict(
            torch.load(config["checkpoint_path"], map_location=device, weights_only=True)
            )
        print(f"*** Continue training from checkpoint ***")
        
    train_logs = {"train_loss":[] , "val_loss":[] , "val_metric":[], "lr":[], "epoch_time": []}
    model.train()
    config["enable_mixed_precision"] = True if device.type == "cuda" else False
    scaler = torch.amp.GradScaler(enabled = config["enable_mixed_precision"])
    best_loss, es_counter, best_epoch, best_ckpnt_path = float('inf'), 0, None, None
    run_name = datetime.now().strftime("CausalLSTM_run_%m-%d_%H-%M")
    print(f"*** Starting run {run_name} for {config["training_mode"]} Language Modeling ***")

    try: # return best artifacts on training interruption
        for epoch in range(config["n_epochs"]):
            epoch_start = datetime.now()
            model.train()
            total_loss = 0.0
            hidden = model.init_hidden(config["batch_size"])

            for X,Y in tqdm(train_loader, desc=f"Epoch {epoch+1}/{config["n_epochs"]}"):
                X, Y = X.to(device), Y.to(device)
                optimizer.zero_grad(set_to_none=True)
                hidden = detach_hidden(hidden)
                with torch.autocast(
                    device_type=device.type, dtype=torch.float16, enabled = config["enable_mixed_precision"]
                ):
                    logits, hidden = model(
                        X,
                        hidden if config["training_mode"] == "statefull" else None
                    )
                    loss = loss_fn(logits, Y)
                total_loss += loss.item()
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), config["grad_clip_norm"])
                scaler.step(optimizer)
                scaler.update()
                if config["dry_run"]:
                    break
    
            # logger
            train_logs["train_loss"].append(total_loss / len(train_loader))
            if val_loader is not None:
                val_loss = evaluate(
                    model, val_loader, loss_fn, device, config, detach_hidden
                )
                train_logs["val_loss"].append(val_loss)
                train_logs["val_metric"].append(math.exp(val_loss))
            train_logs["lr"].append(optimizer.param_groups[0]['lr'])
            train_logs["epoch_time"].append((datetime.now() - epoch_start).total_seconds())
    
            log_msg = (
                f"\r Epoch {epoch + 1}/{config['n_epochs']}, "
                f"train loss: {train_logs['train_loss'][-1]:.4f}, "
            )
            if val_loader is not None:
                log_msg += (
                    f"val loss: {train_logs['val_loss'][-1]:.4f}, "
                    f"val perplexity: {train_logs['val_metric'][-1]:.4f}, "
                )
            log_msg += (
                f"lr: {train_logs['lr'][-1]}, "
                f"epoch time: {train_logs['epoch_time'][-1]:.2f}s"
            )
            print(log_msg)      
    
            # checkpoints
            torch.save(
                model.state_dict(), os.path.join(config["models_dir"], f"CausalLSTM_ckpnt_last.pt")
            )
            with open(os.path.join(config["models_dir"], "checkpoint_info.json"), "w") as f:
                json.dump({
                    "run_name": run_name,
                    "epoch": epoch+1,
                    "train_loss": train_logs['train_loss'][-1],
                    "val_loss": train_logs['val_loss'][-1] if val_loader else None,
                }, f)
            
            if val_loader is not None:
                # lr scheduler
                scheduler.step(val_loss)

                # track best model
                if val_loss < best_loss - config["early_stopping_epsilon"]:
                    best_loss = val_loss
                    best_epoch = epoch + 1
                    best_ckpnt_path = os.path.join(config["models_dir"], f"CausalLSTM_ckpnt_best.pt")
                    torch.save(model.state_dict(), best_ckpnt_path)
                    es_counter = 0
                else:
                    es_counter += 1
        
                # early stopping
                if es_counter >= config["early_stopping_patience"]:
                    print(f"*** Early Stopping triggered at epoch: {epoch+1} ***")
                    break
                
    except KeyboardInterrupt:
        print("\n*** Training interrupted by user ***")
    
    finally:
        # restore best model
        if best_ckpnt_path is not None:
            model.load_state_dict(
                torch.load(best_ckpnt_path, map_location=device, weights_only=True)
            )
            print(f"*** Restoring best model from epoch: {best_epoch} ***")
    
        # save artifacts
        run_dir = os.path.join(config["models_dir"], run_name)
        os.makedirs(run_dir, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(run_dir, "CausalLSTM_ckpnt.pt"))
        write_config(config, os.path.join(run_dir, f"config.yaml"))
        with open(os.path.join(run_dir, f"training_logs.json"), "w") as f:
            json.dump(train_logs, f, indent=2)
        tokenizer.save(os.path.join(run_dir, "tokenizer.json"))
        print(f"*** Run {run_name} completed. artifacts saved in: {run_dir} ***")
    
    return model

#===================#
#     Evaluator     #
#===================#

@torch.no_grad()
def evaluate(
    model: torch.nn.Module, 
    eval_loader: torch.utils.data.DataLoader, 
    loss_fn: Callable, 
    device: torch.device, 
    config:  dict[str, int | float | str], 
    detach_hidden: Callable,
    disable_progress_bar: bool = True
) -> float:
    """
    ======================================================================
    == Evaluator Function for CausalLSTM (GiTHUB.com/HooM4N/CausalLSTM) ==
    ======================================================================
    """
    model.eval()
    total_loss = 0.0
    hidden = model.init_hidden(config["batch_size"])
    
    for X, Y in tqdm(eval_loader, disable = disable_progress_bar):
        X, Y = X.to(device), Y.to(device)
        hidden = detach_hidden(hidden)
        with torch.autocast(
            device_type=device.type, dtype=torch.float16, enabled=config["enable_mixed_precision"]
        ):
            logits, hidden = model(
                X,
                hidden if config["training_mode"] == "statefull" else None
            )
            total_loss += loss_fn(logits, Y).item()
    return total_loss / len(eval_loader)