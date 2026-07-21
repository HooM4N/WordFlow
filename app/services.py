import os
import json
import shutil
import torch
from fastapi import UploadFile

from src.config import WordFlowConfig
from src.model import WordFlowModel
from src.tokenizer import Tokenizer
from src.inference import generate_story, get_similar_words
from app.schemas import GenerateRequest, SimilarRequest

class AppState:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.config = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

state = AppState()

async def process_run_upload(
    checkpoint: UploadFile,
    logs: UploadFile,
    tokenizer: UploadFile,
    config: UploadFile,
    temp_dir: str
) -> dict:
    """
    Unpacks uploaded run files and initializes the model state in memory.

    *WordFlow: Word-Level Language Modeling with RNNs GiTHub.com/HooM4N/WordFlow*
    """
    os.makedirs(temp_dir, exist_ok=True)
    
    files_map = {
        "checkpoint_best.pt": checkpoint,
        "training_logs.json": logs,
        "tokenizer.json": tokenizer,
        "config.json": config
    }
    
    for filename, upload_file in files_map.items():
        path = os.path.join(temp_dir, filename)
        with open(path, "wb") as buffer:
            shutil.copyfileobj(upload_file.file, buffer)

    with open(os.path.join(temp_dir, "config.json"), "r", encoding="utf-8") as f:
        config_data = json.load(f)
    state.config = WordFlowConfig(**config_data)
    
    with open(os.path.join(temp_dir, "training_logs.json"), "r", encoding="utf-8") as f:
        logs_data = json.load(f)
        
    state.tokenizer = Tokenizer.load_from_file(os.path.join(temp_dir, "tokenizer.json"))
    if state.tokenizer is None:
        raise ValueError("Failed to load tokenizer.")
        
    state.model = WordFlowModel(
        vocab_size=state.tokenizer.get_vocab_size(),
        embedding_dim=state.config.model.embedding_dim,
        hidden_dim=state.config.model.hidden_dim,
        num_layers=state.config.model.num_layers,
        tie_weights=state.config.model.tie_weights,
        emb_dropout_p=state.config.model.emb_dropout_p,
        rnn_dropout_p=state.config.model.rnn_dropout_p,
        out_dropout_p=state.config.model.out_dropout_p,
        padding_idx=state.tokenizer.token_to_id("<pad>")
    ).to(state.device)
    
    checkpoint_path = os.path.join(temp_dir, "checkpoint_best.pt")
    state.model.load_state_dict(torch.load(checkpoint_path, map_location=state.device, weights_only=True))
    state.model.eval()
    
    total_params = sum(p.numel() for p in set(state.model.parameters()))
    actual_epochs = len(logs_data.get("train_loss", []))
    epoch_times = logs_data.get("epoch_time", [])
    mean_epoch_time = sum(epoch_times) / len(epoch_times) if epoch_times else 0.0
    
    stats = {
        "params": total_params,
        "emb_dim": state.config.model.embedding_dim,
        "hidden_dim": state.config.model.hidden_dim,
        "layers": state.config.model.num_layers,
        "seq_len": state.config.train.seq_len,
        "epochs": actual_epochs,
        "init_lr": state.config.train.lr,
        "weight_decay": state.config.train.weight_decay,
        "mean_epoch_time": f"{mean_epoch_time:.2f}s"
    }
    
    return {"stats": stats, "logs": logs_data, "device": state.device.type}

def generate_text_service(req: GenerateRequest) -> str:
    if state.model is None or state.tokenizer is None:
        raise ValueError("No run loaded. Load a model first.")
        
    return generate_story(
        model=state.model,
        tokenizer=state.tokenizer,
        device=state.device,
        max_tokens=req.max_tokens,
        temperature=req.temperature,
        seed=req.seed
    )

def get_similar_service(req: SimilarRequest) -> dict:
    if state.model is None or state.tokenizer is None:
        raise ValueError("No run loaded. Load a model first.")
        
    return get_similar_words(
        model=state.model,
        tokenizer=state.tokenizer,
        word=req.word,
        top_n=5
    )
