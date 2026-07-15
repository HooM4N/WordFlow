import torch
import logging

logger = logging.getLogger(__name__)

def resolve_device() -> torch.device:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"*** using device: {device.type} ***")
    return device