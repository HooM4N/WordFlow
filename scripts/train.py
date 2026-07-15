import os
import logging
from datetime import datetime
from argparse import ArgumentParser

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.config import WordFlowConfig
from src.data import get_data
from src.dataset import TruncatedBPTTDataset, bptt_collate
from src.tokenizer import Tokenizer
from src.model import WordFlowModel, detach_hidden
from src.trainer import trainer


def setup_run_directory_and_logger(config: WordFlowConfig) -> str:
    """Creates the timestamped run directory and configures logging."""
    run_name = datetime.now().strftime("wordflow_run_%Y_%m_%d_%H_%M")
    run_dir = os.path.join(config.runs_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)

    # Configure root logger to output to both terminal and a file in the run_dir
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
        handlers=[
            logging.FileHandler(os.path.join(run_dir, "training.log")),
            logging.StreamHandler()
        ]
    )
    return run_dir

def get_args():
    parser = ArgumentParser(description="WordFlow: Word-Level Language Modeling with RNNs")
    parser.add_argument("--config_path", type=str, default="configs/config.yaml",
                        help="Path to config file (yaml)")
    return parser.parse_args()

def main():
    args = get_args()
    
    config = WordFlowConfig.from_yaml(args.config_path)
    
    run_dir = setup_run_directory_and_logger(config)
    logger = logging.getLogger(__name__)
    
    config.save_json(os.path.join(run_dir, "config.json"))
    
    device = torch.device("cuda" if config.use_accelerator and torch.cuda.is_available() else "cpu")
    torch.manual_seed(config.seed)
    logger.info(f"Using device: {device.type}")

    logger.info("Loading corpus...")
    train_corpus = get_data(config.data.train_data_path)
    
    val_corpus = None
    if config.data.do_val_split:
        split_idx = int(len(train_corpus) * (1 - config.data.val_split_ratio))
        val_corpus = train_corpus[split_idx:]
        train_corpus = train_corpus[:split_idx]

    logger.info("Building vocabulary...")
    tokenizer = Tokenizer(
        max_tokens=config.data.max_vocab_size, 
        lowercase=config.data.tokenizer_lowercase
    )

    tokenizer.build_vocab([train_corpus])
    
    train_ids = tokenizer.encode(train_corpus)
    logger.info(f"Training corpus length: {len(train_ids):,} tokens")

    train_ds = TruncatedBPTTDataset(
        train_ids, 
        batch_size=config.train.batch_size, 
        seq_len=config.train.seq_len
        )
    
    # we already getting (N, L) from dataset, so no need for batching
    train_loader = DataLoader(
        train_ds, 
        batch_size=1, 
        shuffle=False, 
        pin_memory=True, 
        collate_fn=bptt_collate
        )
    
    val_loader = None
    if val_corpus:
        val_ids = tokenizer.encode(val_corpus)
        val_ds = TruncatedBPTTDataset(
            val_ids, batch_size=config.train.batch_size, seq_len=config.train.seq_len
            )
        val_loader = DataLoader(
            val_ds, batch_size=1, shuffle=False, pin_memory=True, collate_fn=bptt_collate
            )

    model = WordFlowModel(
        vocab_size=tokenizer.get_vocab_size(),
        embedding_dim=config.model.embedding_dim,
        hidden_dim=config.model.hidden_dim,
        num_layers=config.model.num_layers,
        tie_weights=config.model.tie_weights,
        emb_dropout_p=config.model.emb_dropout_p,
        rnn_dropout_p=config.model.rnn_dropout_p,
        out_dropout_p=config.model.out_dropout_p,
        padding_idx=tokenizer.token_to_id("<pad>")
    ).to(device)

    # all timesteps within sequences are actual word tokens and not padding, but we pass this anyway!
    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer.token_to_id("<pad>"))
    
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=config.train.lr, 
        weight_decay=config.train.weight_decay
    )
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=config.train.n_epochs, 
        eta_min=config.train.eta_min
    )

    logger.info("Beginning training loop...")
    trainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        loss_fn=loss_fn,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=device,
        run_dir=run_dir,
        tokenizer=tokenizer,
        detach_hidden=detach_hidden
    )

if __name__ == "__main__":
    main()