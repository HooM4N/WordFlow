import os
import json
import matplotlib
import matplotlib.pyplot as plt
import torch

from .config import read_config, resolve_device
from .tokenizer import Tokenizer
from .model import CausalLSTM

def plot_training_logs(
    train_logs: dict[str, list[float]], 
    figsize: tuple=(14, 4)
) -> matplotlib.figure.Figure:
    """
    ============================================
    == Plot Training Logs (GitHUB.com/HooM4N) ==
    ============================================
    """
    fig, ax = plt.subplots(1, 3, figsize=figsize)
    has_eval = True if train_logs.get("val_loss") and len(train_logs['val_loss']) > 0 else False
    
    # Loss
    ax[0].plot(train_logs['train_loss'], label="Train Loss")
    
    if has_eval:
        ax[0].plot(train_logs['val_loss'], label="Validation Loss")
    ax[0].set_title("Loss"); ax[0].set_xlabel("Epoch")
    ax[0].set_ylabel("Loss"); ax[0].legend(); ax[0].grid(True)

    # Validation metric
    if has_eval:
        ax[1].plot(train_logs['val_metric'], label="Validation Perpelexity", color="tab:orange")
        ax[1].set_title("Validation Perpelexity"); ax[1].set_xlabel("Epoch")
        ax[1].set_ylabel("Perpelexity"); ax[1].grid(True)

    # Learning rate
    ax[2].plot(train_logs['lr'], label="lr", color="tab:green")
    ax[2].set_title("Learning Rate per Epoch"); ax[2].set_xlabel("Epoch")
    ax[2].set_ylabel("LR"); ax[2].grid(True)

    fig.tight_layout()
    return fig


def load_run(run_path:str):
    """
    =============================================================
    == Load Given Run Artifacts (GiTHUB.com/HooM4N/CausalLSTM) ==
    =============================================================
    - Returns: model, tokenizer, config, device, training_logs
    """
    device = resolve_device()
    try:
        
        config = read_config(os.path.join(run_path, "config.yaml"))

        tokenizer = Tokenizer().load_from_file(os.path.join(run_path, "tokenizer.json"))
        model = CausalLSTM(
            tokenizer.get_vocab_size(), **config["model_params"]
        ).to(device)
        model.load_state_dict(
            torch.load(os.path.join(run_path, "CausalLSTM_ckpnt.pt"), 
                       map_location=device, 
                       weights_only=True)
        )
        with open(os.path.join(run_path, "training_logs.json")) as f:
            training_logs = json.load(f)
        return model, tokenizer, config, device, training_logs
    except Exception as e:
        print(f"*** failed to load component: {e} ***")
        return None

def list_runs(models_dir: str = "models/") -> list[str]:
    return [os.path.join(models_dir, d) for d in os.listdir(models_dir) if d.startswith("CausalLSTM_run")]