import argparse

def get_args():
    parser = argparse.ArgumentParser(description="*** CausalLSTM: Word-Level LSTM Language Modeling ***")
    parser.add_argument("--config_path", type=str, default="config/config.yaml", help="path to config file (yaml)")
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
    device = resolve_device()
    paths = read_config(args.paths_path)
    ensure_dirs(paths)
    config = read_config(args.config_path)
    
    ###############
    ## Load Data ##
    ###############
    train, val = get_sherlock_holmes(paths["data_dir"])
    
    ##################
    ## Tokenization ##
    ##################
    tokenizer = Tokenizer(tokenize_method="nltk")
    tokenizer.build_vocab([train])
    train_ids = tokenizer.encode(train)
    val_ids = tokenizer.encode(val)
    
    #####################
    ## Prepare Dataset ##
    #####################
    train_ds = TruncatedBPTTDataset(train_ids, config["batch_size"], config["seq_len"])
    val_ds = TruncatedBPTTDataset(val_ids, config["batch_size"], config["seq_len"])
    train_ds.get_info()
    
    #######################
    ## Instantiate Model ##
    #######################
    torch.manual_seed(args.seed)
    model = CausalLSTM(tokenizer.get_vocab_size(), **config["model_params"]).to(device)
    print(f"*** count of trainable parameters: {model.get_param_count():,} ***")
    print(model)
    
    ##############
    ## Training ##
    ##############
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.NAdam(model.parameters(), **config["optimizer_params"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, **config["lr_scheduler_params"])
    
    model = trainer(
        model, optimizer, config, paths, device, scheduler, detach_hidden, train_ds, val_ds, loss_fn, tokenizer
    )

if __name__ == "__main__":
    args = get_args()
    train(args)