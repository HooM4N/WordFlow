import torch
from typing import Dict
from .tokenizer import Tokenizer
from .model import CausalLSTM

@torch.no_grad()
def generate(
    model: CausalLSTM, 
    tokenizer: Tokenizer, 
    config: Dict, 
    device: torch.device, 
    init_word: str = "i", 
    max_new_tokens: str = 60, 
    temperature: float = 0.9, 
    seed: int = None
):
    """
    ====================================================================
    == Autoregressive Word Generation (GiTHUB.com/HoomM4N/CausalLSTM) ==
    ====================================================================
    - Starts from given init word 
    - Samples tokens using temperature scaling
    - Skips <unk>, replaces <eos> with newline  
    """
    if seed is not None:
        torch.manual_seed(seed)
    model.eval()
    hidden = model.init_hidden(1)
    tkns = ["<unk>", "<eos>", init_word]
    unk_id, eos_id, init_idx = [tokenizer.token_to_id(t) for t in tkns]
    if init_idx is not None:
        input_ = torch.tensor(init_idx, dtype=torch.long).reshape(1,-1).to(device)
    else:
        raise Exception(f"{init_word} is not in vocab")
        
    generated_words = [init_word]
    for _ in range(max_new_tokens):
        output, hidden = model(input_, hidden)
        probs = output.squeeze().div(temperature).exp().cpu()
        while True:
            token_idx = torch.multinomial(probs, 1)[0]
            if token_idx != unk_id: break
        input_.fill_(token_idx)
        generated_words.append(tokenizer.id_to_token(token_idx) if token_idx != eos_id else "\n")
    return " ".join(generated_words)