import os
import sys
import json
import shutil

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import torch

# Add the project root to sys.path so we can import from src/
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.config import WordFlowConfig
from src.model import WordFlowModel
from src.tokenizer import Tokenizer
from src.inference import generate_story, get_similar_words

app = FastAPI(title="WordFlow Dashboard")

# Ensure static folder exists contextually
static_dir = os.path.join(os.path.dirname(__file__), "static")
os.makedirs(static_dir, exist_ok=True)
app.mount("/static", StaticFiles(directory=static_dir), name="static")

class AppState:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.config = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

state = AppState()

@app.get("/")
def read_root():
    """Serves the dashboard UI."""
    return FileResponse(os.path.join(static_dir, "index.html"))

@app.post("/api/load-run")
async def load_run(
    checkpoint: UploadFile = File(...),
    logs: UploadFile = File(...),
    tokenizer: UploadFile = File(...),
    config: UploadFile = File(...)
):
    """Saves the uploaded folder files to a temp directory, validates, and loads the model into memory."""
    try:
        temp_dir = os.path.join(os.path.dirname(__file__), ".temp_run")
        os.makedirs(temp_dir, exist_ok=True)
        
        # Save files locally to load via PyTorch / standard python I/O
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

        # Load Config
        with open(os.path.join(temp_dir, "config.json"), "r", encoding="utf-8") as f:
            config_data = json.load(f)
        state.config = WordFlowConfig(**config_data)
        
        # Load Logs
        with open(os.path.join(temp_dir, "training_logs.json"), "r", encoding="utf-8") as f:
            logs_data = json.load(f)
            
        # Load Tokenizer
        state.tokenizer = Tokenizer.load_from_file(os.path.join(temp_dir, "tokenizer.json"))
        if state.tokenizer is None:
            raise ValueError("Failed to load tokenizer.")
            
        # Initialize and Load Model
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
        
        # Calculate Stats (Deduplicate tied weights using a set)
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
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load run data: {str(e)}")

class GenerateRequest(BaseModel):
    max_tokens: int = 100
    temperature: float = 0.8
    seed: int | None = None

@app.post("/api/generate")
def generate(req: GenerateRequest):
    """Autoregressive generation endpoint."""
    if state.model is None or state.tokenizer is None:
        raise HTTPException(status_code=400, detail="No run loaded. Load a model first.")
        
    story = generate_story(
        model=state.model,
        tokenizer=state.tokenizer,
        device=state.device,
        max_tokens=req.max_tokens,
        temperature=req.temperature,
        seed=req.seed
    )
    return {"story": story}

class SimilarRequest(BaseModel):
    word: str

@app.post("/api/similar")
def similar(req: SimilarRequest):
    """Cosine similarity endpoint."""
    if state.model is None or state.tokenizer is None:
        raise HTTPException(status_code=400, detail="No run loaded. Load a model first.")
        
    results = get_similar_words(
        model=state.model,
        tokenizer=state.tokenizer,
        word=req.word,
        top_n=5
    )
    return {"results": results}
