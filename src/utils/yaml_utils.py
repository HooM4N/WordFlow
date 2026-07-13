import os
import yaml
import logging

logger = logging.getLogger(__name__)

def read_yaml(yaml_file_path: str) -> dict:
    """
    ========================================
    == Read YAML File (GitHUB.com/HooM4N) ==
    ========================================
    """
    try:
        with open(yaml_file_path, "r", encoding="utf-8") as f:
            yaml_file = yaml.safe_load(f)
        return yaml_file
    except Exception as e:
        logger.error(f"** Error reading config file: {e} **")
    return None

def write_yaml(config: dict, yaml_path: str):
    """
    ===============================================
    == Writes to YAML Config (GitHUB.com/HooM4N) ==
    ===============================================
    """
    os.makedirs(os.path.dirname(yaml_path), exist_ok=True)
    try:
        with open(yaml_path, "w") as f:
            yaml.safe_dump(config, f, default_flow_style=False)
    except Exception as e:
        logger.error(f"** Error writing config file: {e} **")