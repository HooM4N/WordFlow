import torch
from argparse import ArgumentParser

from src.config import resolve_device, read_config, ensure_dirs, model_summary
from src.data import get_data
from src.tokenizer import Tokenizer
from src.dataset import TruncatedBPTTDataset
from src.model import CausalLSTM, detach_hidden
from src.trainer import trainer
from src.glove_embeddings import get_glove_embeddings

def get_args():
    parser = ArgumentParser(description="*** CausalLSTM: Word-Level LSTM Language Modeling ***")
    parser.add_argument("--config_path", type=str, default=None, help="path to config file (yaml)")
    parser.add_argument("--paths_path", type=str, default="config/paths.yaml", help="path to paths file (yaml)")
    parser.add_argument("--n_epochs", type=int, default=40, help="number of training epochs")
    parser.add_argument("--batch_size", type=int, default=128, help="training batch size")
    parser.add_argument("--use_accelerator", type=bool, default=True, help="use available accelerator or cpu")
    parser.add_argument("--seed", type=int, default=45, help="random seed")
    return parser.parse_args()

def train(args):
    ###################
    ## Configuration ##
    ###################
    device = resolve_device(args.use_accelerator)
    paths = read_config(args.paths_path)
    ensure_dirs(paths)
    config = read_config(paths["config_path"] if args.config_path is None else args.config_path)
    
    ###############
    ## Load Data ##
    ###############
    splits = get_data(paths["data_name"], paths["data_dir"])
    if isinstance(splits, tuple):
        train, val = splits
    elif isinstance(splits, str):
        train, val = splits, None
    else:
        raise ValueError("*** get_data must return either (train, val) splits or a single train corpus, both as strings ***")

    print(f"*** dataset {paths["data_name"]} loaded ***")
    
    ##################
    ## Tokenization ##
    ##################
    tokenizer = Tokenizer(
        max_tokens = config["max_vocab_size"],
        tokenize_method="nltk"
    )
    tokenizer.build_vocab([train])
    train_ids = tokenizer.encode(train)
    print(f"*** training corpus size: {len(train_ids):,} tokens")
    if val is not None:
        val_ids = tokenizer.encode(val)
        print(f"*** validation corpus size: {len(val_ids):,} tokens")
    
    #####################
    ## Prepare Dataset ##
    #####################
    train_ds = TruncatedBPTTDataset(train_ids, config["batch_size"], config["seq_len"])
    if val is not None:
        val_ds = TruncatedBPTTDataset(val_ids, config["batch_size"], config["seq_len"])
    train_ds.get_info()
    
    #######################
    ## Instantiate Model ##
    #######################
    if config["use_glove_embeddings"]:
        print(f"*** loading pretrained GloVe word embeddings ***")
        glove_embeddings = get_glove_embeddings(paths["glove_embeddings_path"], config["glove_dim"], tokenizer.get_vocab())
        glove_embeddings = torch.tensor(glove_embeddings, dtype=torch.float32)
        config["model_params"]["embedding_dim"] = config["glove_dim"]
        
    torch.manual_seed(args.seed)
    model = CausalLSTM(
        tokenizer.get_vocab_size(), 
        **config["model_params"], 
        pretrained_embedding_matrix = glove_embeddings if config["use_glove_embeddings"] else None
    ).to(device)
    
    print(model)
    model_summary(model)
    
    ##############
    ## Training ##
    ##############
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.NAdam(model.parameters(), **config["optimizer_params"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, **config["lr_scheduler_params"])
    
    model = trainer(
        model, optimizer, config, paths, device, scheduler,
        detach_hidden, train_ds, loss_fn, tokenizer,
        val_ds = val_ds if val else None
    )

if __name__ == "__main__":
    args = get_args()
    train(args)