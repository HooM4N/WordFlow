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
    pass

if __name__ == "__main__":
    args = get_args()
    train(args)