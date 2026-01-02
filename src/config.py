import os
import yaml
import torch 

#==============================#
#     Config File Utilites     #
#==============================#

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

#=====================================#
#     Environment Setup Utilities     #
#=====================================#

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


#=================================#
#     Model Summary Utilities     #
#=================================#


def model_summary(model: torch.nn.Module) -> str:
    """
    =================================================================
    == Pretty Print of Model Parameter Summary (GitHUB.com/HooM4N) ==
    =================================================================
    """
    rows = []
    for name, m in model.named_modules():
        if name == "":
            continue
        trainable = sum(p.numel() for p in m.parameters(recurse=False) if p.requires_grad)
        frozen = sum(p.numel() for p in m.parameters(recurse=False) if not p.requires_grad)
        rows.append((
            name,
            f"{m.__class__.__name__}({m.extra_repr()})",
            trainable + frozen,
            trainable,
        ))

    total_params = sum(p.numel() for p in model.parameters())
    total_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    mod_w = max([6] + [len(r[0]) for r in rows])
    cls_w = max([5] + [len(r[1]) for r in rows])
    num_w = max(10, len(f"{total_params:,}"))
    width = mod_w + cls_w + num_w * 2 + 10

    lines = [
        "=" * width,
        f"Model Parameter Summary for {model.__class__.__name__}".center(width),
        "=" * width,
        f"{'Module':{mod_w}} | {'Class':{cls_w}} | {'Total':>{num_w}} | {'Trainable':>{num_w}}",
        "-" * width,
    ]

    lines += [
        f"{n:{mod_w}} | {c:{cls_w}} | {t:{num_w},} | {tr:{num_w},}"
        for n, c, t, tr in rows
    ]

    lines += [
        "-" * width,
        f"{'TOTAL':{mod_w}} | {'':{cls_w}} | {total_params:{num_w},} | {total_trainable:{num_w},}",
        "=" * width,
    ]

    return "\n".join(lines)