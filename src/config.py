import os
import yaml
import torch 

def read_config(config_path: str) -> dict:
    """
    ==========================================
    == Read YAML Config (GitHUB.com/HooM4N) ==
    ==========================================
    """
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        print(f"** Error reading config file: {e} **") 
    return None

def write_config(config: dict, config_path: str):
    """
    ===============================================
    == Writes to YAML Config (GitHUB.com/HooM4N) ==
    ===============================================
    """
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    try:
        with open(config_path, "w") as f:
            yaml.safe_dump(config, f, default_flow_style=False)
    except Exception as e:
        print(f"** Error writing config file: {e} **")

def resolve_device(use_accelerator: bool=True) -> torch.device:
    """
    ==========================================================
    == Resolve Available PyTorch Device (GitHUB.com/HooM4N) ==
    ==========================================================
    """
    if use_accelerator and torch.accelerator.is_available():
        device = torch.accelerator.current_accelerator()
    else:
        device = torch.device("cpu")
    print(f"*** Using device: {device.type} ***")
    return device

def ensure_dirs(paths: dict):
    """
    ================================================================
    == Ensures Existence of Given Directories (GitHUB.com/HooM4N) ==
    ================================================================
    """
    for k,p in paths.items():
        if p is not None and k.endswith("_dir"):
            os.makedirs(p, exist_ok=True)


def model_summary(model: torch.nn.Module, width: int = 80):
    """
    =========================================================
    == Print Model's Paramters Summary (GiTHUB.com/HooM4N) ==
    =========================================================
    """
    print("="*width)
    print(f"Parameter Count Summary for {model.__class__.__name__}".center(width))
    print("="*width)
    print(f"{'Module':20} | {'Class':15} | {'Trainable':10} | {'Frozen':10} | {'Total':10}")
    print("-"*width)

    for name, m in model.named_modules():
        if not name: 
            continue
        trainable_params = sum(p.numel() for p in m.parameters(recurse=False) if p.requires_grad)
        frozen_params = sum(p.numel() for p in m.parameters(recurse=False) if not p.requires_grad)
        total_params = trainable_params + frozen_params

        print(f"{name:20} | {m.__class__.__name__:15} | {trainable_params:<10,} | {frozen_params:<10,} | {total_params:<10,}")

    total_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    print("-"*width)
    print(f"{'TOTAL':20} | {'':15} | {total_trainable:<10,} | {total_frozen:<10,} | {total_trainable+total_frozen:<10,}")
    print("="*width)