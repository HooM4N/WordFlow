import yaml
import json
from pydantic import BaseModel

class DataConfig(BaseModel):
    train_data_path: str
    val_data_path: str | None = None
    max_vocab_size: int = 25000
    tokenizer_lowercase: bool = True

class ModelConfig(BaseModel):
    embedding_dim: int = 300
    hidden_dim: int = 300
    num_layers: int = 1
    tie_weights: bool = True
    emb_dropout_p: float = 0.2
    rnn_dropout_p: float = 0.25
    out_dropout_p: float = 0.2

class TrainConfig(BaseModel):
    seq_len: int = 170
    batch_size: int = 128
    n_epochs: int = 60
    lr: float = 0.001
    weight_decay: float = 0.01
    eta_min: float = 0.00005
    early_stopping_patience: int = 4
    grad_clip_norm: float = 1.0
    enable_mixed_precision: bool = True

class WordFlowConfig(BaseModel):
    seed: int = 1212
    use_accelerator: bool = True
    runs_dir: str = "runs/"  # <-- Fixed this line!
    
    data: DataConfig
    model: ModelConfig
    train: TrainConfig

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "WordFlowConfig":
        with open(yaml_path, "r", encoding="utf-8") as f:
            config_dict = yaml.safe_load(f)
        return cls(**config_dict)

    def save_json(self, save_path: str):
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(self.model_dump(), f, indent=4)