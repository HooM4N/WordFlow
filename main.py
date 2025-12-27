import argparse
import torch
from torch.utils.data import DataLoader

from src.config import resolve_device, read_config, ensure_dirs, model_summary
from src.data import get_data, train_val_split, summarize_data, flatten
from src.tokenizer import Tokenizer
from src.dataset import TruncatedBPTTDataset, StatelessLSTMDataset
from src.model import CausalLSTM, detach_hidden
from src.pretrained_embeddings import get_glove_embeddings
from src.trainer import trainer

def get_args():
    parser = argparse.ArgumentParser(description="*** CausalLSTM: Word-Level LSTM Language Modeling ***")
    parser.add_argument("--config_path", type=str, default=None, help="path to config file (yaml)")
    parser.add_argument("--paths_path", type=str, default="config/paths.yaml", help="path to paths file (yaml)")
    parser.add_argument("--n_epochs", type=int, default=40, help="number of training epochs")
    parser.add_argument("--batch_size", type=int, default=128, help="training batch size")
    parser.add_argument("--use_accelerator", type=bool, default=True, help="use available accelerator or cpu")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    return parser.parse_args()

def train(args):
    #=======================#
    #     Configuration     #
    #=======================#
    device = resolve_device(args.use_accelerator)
    paths = read_config(args.paths_path)
    ensure_dirs(paths)
    config = read_config(
        paths["config_path"] if args.config_path is None else args.config_path
    )
    
    #======================#
    #     Prepare Data     #
    #======================#
    train_corpus = get_data(
        paths["train_data_path"],
        **config["get_data_params"]
    )
    summarize_data(train_corpus)
    evaluate = False
        
    if paths["val_data_path"] is not None:
        val_corpus = get_data(paths["val_data_path"])
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
    tokenizer = Tokenizer(max_tokens = config["max_vocab_size"])
    tokenizer.build_vocab(train_corpus)
    train_ids = [tokenizer.encode(d) for d in train_corpus]
    
    if evaluate:
        val_ids = [tokenizer.encode(d) for d in val_corpus]
    
    #============================#
    #     Prepare DataLoader     #
    #============================#

    if config["training_mode"] == "statefull":
        def identity_collate(batch): 
            return batch[0]
            
        train_ds = TruncatedBPTTDataset(
            flatten(train_ids), config["batch_size"], config["seq_len"]
        )
        train_ds.get_info()
        train_loader = DataLoader(
            train_ds, batch_size=1, shuffle=False, pin_memory = True, collate_fn = identity_collate
        )
        if evaluate:
            val_ds = TruncatedBPTTDataset(
                flatten(val_ids), config["batch_size"], config["seq_len"]
            )
            val_loader = DataLoader(
                val_ds, batch_size=1, shuffle=False, pin_memory = True, collate_fn = identity_collate
            )
                
    elif config["training_mode"] == "stateless":
        train_ds = StatelessLSTMDataset(
            train_ids, config["seq_len"], config["min_seq_len"]
        )
        train_loader = DataLoader(
            train_ds, batch_size=config["batch_size"], shuffle=True, pin_memory = True
        )
        if evaluate:
            val_ds = StatelessLSTMDataset(
                val_ids, config["seq_len"], config["min_seq_len"]
            )
            val_loader = DataLoader(
                val_ds, batch_size=config["batch_size"], shuffle=False, pin_memory = True
            )

    #===========================#
    #     Instantiate Model     #
    #===========================#
    if config["use_glove_embeddings"]:
        try:
            print(f"*** loading pretrained GloVe word embeddings ***")
            pretrained_embeddings = get_glove_embeddings(
                paths["glove_embeddings_path"], config["glove_dim"], tokenizer.get_vocab()
            )
            pretrained_embeddings = torch.tensor(pretrained_embeddings, dtype=torch.float32)
            config["model_params"]["embedding_dim"] = config["glove_dim"]
        except Exception as e:
            print(f"*** failed to load pretrained embedding matrix. initializing with random values: {e} ***")
        
    torch.manual_seed(config["seed"])
    model = CausalLSTM(
        tokenizer.get_vocab_size(), 
        **config["model_params"], 
        pretrained_embedding_matrix = pretrained_embeddings if config["use_glove_embeddings"] else None
    ).to(device)
    
    print(model)
    model_summary(model)
    
    #=======================#
    #     Training Loop     #
    #=======================#
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.NAdam(
        model.parameters(), **config["optimizer_params"]
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, **config["lr_scheduler_params"]
    )
    
    model = trainer(
        model, optimizer, config, paths, device, scheduler,
        detach_hidden, train_loader, loss_fn, tokenizer,
        val_loader = val_loader if evaluate else None
    )

if __name__ == "__main__":
    args = get_args()
    train(args)