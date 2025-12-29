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

def get_model_summary(
    model: torch.nn.Module
) -> dict[str, str | int]:
    """
    ====================================================
    === Summarize PyTorch Modules (GitHUB.com/HooM4N) ==
    ====================================================
    """
    modules = {"name":[], "class_name":[], "repr":[],"trainable_params":[],
               "non_trainable_params":[], "total_params":[]}
    
    for n,m in model.named_modules():
        if len(n) == 0:
            name = class_name = model.__class__.__name__
        else:
            name, class_name = n, m.__class__.__name__
        
        trainable_params = sum(p.numel() for p in m.parameters() if p.requires_grad)
        non_trainable_params = sum(p.numel() for p in m.parameters() if not p.requires_grad)
        
        modules["name"].append(name)
        modules["class_name"].append(class_name)
        modules["repr"].append(f"{class_name}({m.extra_repr()})")
        
        modules["trainable_params"].append(trainable_params)
        modules["non_trainable_params"].append(non_trainable_params)
        modules["total_params"].append(trainable_params+non_trainable_params)
    return modules
    
def model_summary(model: torch.nn.Module) -> str:
    """
    =============================================
    == Model Summary Table (GitHUB.com/HooM4N) ==
    =============================================
    """
    modules = get_model_summary(model)
    num_w, mod_w = 12, 20
    reprs = modules["repr"][1:]
    class_w = max(30, max(len(r) for r in reprs) + 2) if reprs else 30
    width = mod_w + class_w + num_w*3 + 12 

    header = [
        "=" * width,
        f"Model Summary for {model.__class__.__name__}".center(width),
        "=" * width,
        f"{'Module':{mod_w}} | {'Class':{class_w}} | {'Trainable':>{num_w}} | {'Frozen':>{num_w}} | {'Total':>{num_w}}",
        "-" * width,
    ]

    rows = [
        f"{n:{mod_w}} | {r:{class_w}} | {t:{num_w},} | {f:{num_w},} | {tot:{num_w},}"
        for n, r, t, f, tot in zip(
            modules["name"][1:], modules["repr"][1:],
            modules["trainable_params"][1:], modules["non_trainable_params"][1:],
            modules["total_params"][1:]
        )
    ]

    total_trainable = modules["trainable_params"][0]
    total_frozen = modules["non_trainable_params"][0]

    footer = [
        "-" * width,
        f"{'TOTAL':{mod_w}} | {'':{class_w}} | {total_trainable:{num_w},} | {total_frozen:{num_w},} | {(total_trainable + total_frozen):{num_w},}",
        "=" * width,
    ]
    return "\n".join(header + rows + footer)