import os
import json
import torch
import logging

from src.config import WordFlowConfig
from src.tokenizer import Tokenizer
from src.model import WordFlowModel
from src.inference import generate_text, get_similar_words

logger = logging.getLogger(__name__)

class InferenceService:
    def __init__(self, runs_dir="runs"):
        self.runs_dir = runs_dir
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.current_run = None
        self.model = None
        self.tokenizer = None
        self.config = None

    def list_all_runs(self) -> list[dict]:
        """Scans the runs directory and returns configs and logs for the UI"""
        runs_info = []
        if not os.path.exists(self.runs_dir):
            return runs_info

        for run_name in sorted(os.listdir(self.runs_dir), reverse=True):
            run_path = os.path.join(self.runs_dir, run_name)
            if not os.path.isdir(run_path): 
                continue
            
            config_path = os.path.join(run_path, "config.json")
            logs_path = os.path.join(run_path, "training_logs.json")
            
            run_data = {
                "id": run_name,
                "config": {},
                "logs": {}
            }
            
            if os.path.exists(config_path):
                with open(config_path, "r", encoding="utf-8") as f:
                    run_data["config"] = json.load(f)
                    
            if os.path.exists(logs_path):
                with open(logs_path, "r", encoding="utf-8") as f:
                    run_data["logs"] = json.load(f)
                    
            runs_info.append(run_data)
            
        return runs_info

    def load_run(self, run_name: str) -> dict:
        """Loads a model run into GPU/CPU memory"""
        if self.current_run == run_name:
            return self._get_model_stats()
            
        run_path = os.path.join(self.runs_dir, run_name)
        config_path = os.path.join(run_path, "config.json")
        tok_path = os.path.join(run_path, "tokenizer.json")
        
        ckpt_path = os.path.join(run_path, "checkpoint_best.pt")
        if not os.path.exists(ckpt_path):
            ckpt_path = os.path.join(run_path, "checkpoint_last.pt")

        if not all(os.path.exists(p) for p in [config_path, tok_path, ckpt_path]):
            raise FileNotFoundError(f"Missing essential files in {run_path}")

        # Load configuration
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = WordFlowConfig(**json.load(f))
            
        # Load Tokenizer
        self.tokenizer = Tokenizer.load_from_file(tok_path)
        
        # Load Model
        self.model = WordFlowModel(
            vocab_size=self.tokenizer.get_vocab_size(),
            embedding_dim=self.config.model.embedding_dim,
            hidden_dim=self.config.model.hidden_dim,
            num_layers=self.config.model.num_layers,
            tie_weights=self.config.model.tie_weights,
            emb_dropout_p=self.config.model.emb_dropout_p,
            rnn_dropout_p=self.config.model.rnn_dropout_p,
            out_dropout_p=self.config.model.out_dropout_p,
            padding_idx=self.tokenizer.token_to_id("<pad>")
        ).to(self.device)
        
        self.model.load_state_dict(torch.load(ckpt_path, map_location=self.device, weights_only=True))
        self.model.eval()
        self.current_run = run_name
        
        logger.info(f"Successfully loaded run: {run_name}")
        return self._get_model_stats()

    def _get_model_stats(self) -> dict:
        """Calculates parameters to display nicely on the UI"""
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        return {
            "run_loaded": self.current_run,
            "vocab_size": self.tokenizer.get_vocab_size(),
            "embedding_dim": self.config.model.embedding_dim,
            "hidden_dim": self.config.model.hidden_dim,
            "num_layers": self.config.model.num_layers,
            "tie_weights": self.config.model.tie_weights,
            "parameters": trainable
        }

    def generate(self, prompt: str, max_tokens: int, temperature: float, top_k: int, seed: int) -> str:
        if not self.model:
            raise RuntimeError("No model is currently loaded. Please load a run first.")
            
        return generate_text(
            self.model, self.tokenizer, self.device,
            prompt=prompt, max_tokens=max_tokens, 
            temperature=temperature, top_k=top_k, seed=seed
        )

    def similar(self, word: str, top_n: int) -> dict:
        if not self.model:
            raise RuntimeError("No model is currently loaded. Please load a run first.")
            
        return get_similar_words(self.model, self.tokenizer, word, top_n)
