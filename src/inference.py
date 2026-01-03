import torch
from .tokenizer import Tokenizer
from .model import WordFlowModel
from .preprocess import text_preprocess

@torch.no_grad()
def generate(
    model: WordFlowModel, 
    tokenizer: Tokenizer, 
    config: dict[str, int | float | str], 
    device: torch.device,
    init_word: str = None, 
    max_new_tokens: int = 32, 
    temperature: float = 0.9,
    post_process: bool = True,
    seed: int = None
) -> str:
    """
    ==================================================================
    == Autoregressive Word Generation (GiTHUB.com/HoomM4N/WordFlow) ==
    ==================================================================
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
    if init_idx != unk_id:
        input_ = torch.tensor(init_idx, dtype=torch.long).reshape(1,-1).to(device)
    else:
        print(f"*** {init_word} is not in vocab, picking a random initial word... ***")
        input_ = torch.randint(low=5, high=len(tokenizer.get_vocab()), size=(1,1), dtype=torch.long).to(device)
        init_word = tokenizer.id_to_token(input_.item())
        
    generated_words = [init_word]
    for _ in range(max_new_tokens):
        output, hidden = model(input_, hidden)
        probs = output.squeeze().div(temperature).exp().cpu()

        while True:
            token_idx = torch.multinomial(probs, 1)[0]
            if token_idx != unk_id: 
                break
        
        if token_idx == eos_id:
            break
        input_.fill_(token_idx)
        generated_words.append(tokenizer.id_to_token(token_idx))
        
    text = " ".join(generated_words)
    return post_process_fn(text) if post_process else text

def post_process_fn(text:str) -> str:
    return text.replace("<newline>", "\n").replace("<eos>", "")
    
@torch.no_grad()
def predict_next_word(
    model: WordFlowModel, 
    tokenizer: Tokenizer, 
    config: dict[str, int | float | str], 
    device: torch.device, 
    context: str = "it is", 
    top_k: int = 5,
    preprocess_kwargs: dict = None,
) -> dict[str, float]:
    """
    ====================================================================
    == Next Word Probability Prediction (GiTHUB.com/HoomM4N/WordFlow) ==
    ====================================================================
    - Returns top k candidate words with their probabilities given a context
    """
    model.eval()
    hidden = model.init_hidden(1)
    tkns = ["<unk>", "<eos>"]
    unk_id, eos_id = [tokenizer.token_to_id(t) for t in tkns]

    context = text_preprocess(context, **preprocess_kwargs)
    context_ids = tokenizer.encode(context)
    input_ = torch.tensor(context_ids, dtype=torch.long).reshape(1,-1).to(device)
    output, _ = model(input_, hidden) 
    top_probs, top_ids = torch.topk(torch.softmax(output[0,:,-1].cpu(), dim=0), top_k)
    return {
        tokenizer.id_to_token(i.item()): round(p.item(), 2) 
        for i, p in zip(top_ids, top_probs)
    }