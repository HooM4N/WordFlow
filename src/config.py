import torch
import os, yaml
from typing import Dict

def read_yaml(config_path: str) -> Dict:
    """
    ===========================================
    == Reads a YAML file (GitHUB.com/HooM4N) ==
    ===========================================
    """
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        print(f"** Error reading config file: {e} **") 
    return None

def write_yaml(config: Dict, config_path: str):
    """
    ==========================================================
    == Writes a dictionary to YAML file (GitHUB.com/HooM4N) ==
    ==========================================================
    """
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    try:
        with open(config_path, "w") as f:
            yaml.safe_dump(config, f, default_flow_style=False)
    except Exception as e:
        print(f"** Error writing config file: {e} **")

def resolve_device(use_accelerator: bool=True) -> torch.device:
    """
    =================================================================
    == Detects & returns available accelerator (GitHUB.com/HooM4N) ==
    =================================================================
    """
    if use_accelerator and torch.accelerator.is_available():
        device = torch.accelerator.current_accelerator()
    else:
        device = torch.device("cpu")
    print(f"*** Using device: {device.type} ***")
    return device