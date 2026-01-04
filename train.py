import os, torch
from argparse import ArgumentParser
from torch.utils.data import DataLoader

from src.config import resolve_device, read_config, ensure_dirs, model_summary
from src.data import get_data, train_val_split, summarize_data
from src.tokenizer import Tokenizer
from src.dataset import (
TruncatedBPTTDataset, SlidingWindowDataset, VariableLengthDataset, varlen_collate, sliding_collate, bptt_collate
)
from src.model import WordFlowModel, detach_hidden
from src.pretrained_embeddings import get_glove_embeddings
from src.trainer import trainer

def get_args():
    parser = ArgumentParser(description="*** WordFlow: Word-Level Language Modeling with RNNs ***")
    parser.add_argument("--config_path", type=str, default="config/config.yaml",
                        help="path to config file (yaml)")
    parser.add_argument("--training_mode", choices=["statefull", "stateless"], default=None,
                        help="train with 'statefull' (Truncated BPTT) or 'stateless' (overlapping sequence windows)")
    parser.add_argument("--n_epochs", type=int, default=None,
                        help="number of training epochs")
    parser.add_argument("--batch_size", type=int, default=None,
                        help="training batch size")
    parser.add_argument("--use_accelerator", type=bool, default=None,
                        help="use available accelerator or cpu")
    parser.add_argument("--seed", type=int, default=None,
                        help="random seed")
    parser.add_argument("--dry_run", action="store_true",
                        help="run one step of training for testing")
    return parser.parse_args()

def train(config):
    #=======================#
    #     Configuration     #
    #=======================#
    # validate_config(config)
    device = resolve_device(config["use_accelerator"])
    ensure_dirs(config)
    
    #======================#
    #     Prepare Data     #
    #======================#
    train_corpus = get_data(
        config["train_data_path"],
        **config["get_data_params"],
        random_seed = config["seed"],
    )
    evaluate = False
    
    if config["val_data_path"] is not None:
        val_corpus = get_data(config["val_data_path"])
        evaluate = True
    elif config["val_split_enable"]:
        train_corpus, val_corpus = train_val_split(
            train_corpus, 
            config["val_split_ratio"], 
            config["val_split_shuffle"],
            config["seed"]
        )
        evaluate = True

    #======================#
    #     Tokenization     #
    #======================#        
    tokenizer = Tokenizer(
        max_tokens = config["max_vocab_size"], 
        lowercase = config["tokenizer_lowercase"]
    )
    tokenizer.build_vocab(train_corpus)
    train_ids = [tokenizer.encode(d) for d in train_corpus]
    
    if evaluate:
        val_ids = [tokenizer.encode(d) for d in val_corpus]
    
    #============================#
    #     Prepare DataLoader     #
    #============================#

    if config["training_mode"] == "bptt":
        train_ds = TruncatedBPTTDataset(
            train_ids, config["batch_size"], config["seq_len"]
        )
        train_ds.print_info()
        train_loader = DataLoader(
            train_ds, batch_size=1, shuffle=False, pin_memory = True, collate_fn = bptt_collate
        )
        if evaluate:
            val_ds = TruncatedBPTTDataset(
                val_ids, config["batch_size"], config["seq_len"]
            )
            val_loader = DataLoader(
                val_ds, batch_size=1, shuffle=False, pin_memory = True, collate_fn = bptt_collate
            )
                
    elif config["training_mode"] == "sliding":
        train_ds = SlidingWindowDataset(
            train_ids, config["seq_len"], min(config["min_seq_len"], config["seq_len"])
        )
        print(f"*** {len(train_ds):,} training samples created ***")
        train_loader = DataLoader(
            train_ds, batch_size=config["batch_size"], shuffle=True, pin_memory = True, collate_fn = sliding_collate
        )
        if evaluate:
            val_ds = SlidingWindowDataset(
                val_ids, config["seq_len"], config["min_seq_len"]
            )
            val_loader = DataLoader(
                val_ds, batch_size=config["batch_size"], shuffle=False, pin_memory = True, collate_fn = sliding_collate
            )

    elif config["training_mode"] == "varlen":
        train_ds = VariableLengthDataset(train_ids)
        train_loader = DataLoader(
            train_ds, batch_size=config["batch_size"], shuffle=True,pin_memory = True, collate_fn = varlen_collate
        )
        print(f"*** {len(train_ds):,} training samples created ***")
        if evaluate:
            val_ds = VariableLengthDataset(val_ids)
            val_loader = DataLoader(
                val_ds, batch_size=config["batch_size"], shuffle=False, pin_memory = True, collate_fn = varlen_collate
            )
    #===========================#
    #     Instantiate Model     #
    #===========================#
    if config["use_glove_embeddings"]:
        try:
            print(f"*** loading pretrained GloVe word embeddings ***")
            pretrained_embeddings = get_glove_embeddings(
                config["glove_embeddings_path"], config["glove_dim"], tokenizer.get_vocab()
            )
            pretrained_embeddings = torch.tensor(pretrained_embeddings, dtype=torch.float32)
        except Exception as e:
            print(f"*** failed to load pretrained embedding matrix. initializing with random values: {e} ***")
        
    torch.manual_seed(config["seed"])
    model = WordFlowModel(
        tokenizer.get_vocab_size(), 
        **config["model_params"], 
        pretrained_embedding_matrix = pretrained_embeddings if config["use_glove_embeddings"] else None
    ).to(device)
    
    print(model_summary(model))
    
    #=======================#
    #     Training Loop     #
    #=======================#
    loss_fn = torch.nn.CrossEntropyLoss(
        ignore_index = tokenizer.token_to_id("<pad>")
    )
    optimizer = torch.optim.NAdam(
        model.parameters(), **config["optimizer_params"]
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, **config["lr_scheduler_params"]
    )
    
    model = trainer(
        model, optimizer, config, device, scheduler,
        detach_hidden, train_loader, loss_fn, tokenizer,
        val_loader = val_loader if evaluate else None
    )
    
if __name__ == "__main__":
    ## GET CLI ARGS ##
    args = get_args()
    ## READ CONFIG FILE ##
    config = read_config(args.config_path)
    ## OVERRIDE CONFIG WITH CLI ARGS ##
    for key in [
        "training_mode", "n_epochs", "batch_size", "use_accelerator", "seed", "dry_run"
    ]:
        val = getattr(args, key)
        if val is not None:
            config[key] = val
    ## TRAINING PROCESS ##
    train(config)