import torch
from .tokenizer import Tokenizer
from .model import WordFlowModel
from .data import text_preprocessor

@torch.no_grad()
def generate(
    model: WordFlowModel, 
    tokenizer: Tokenizer, 
    device: torch.device,
    init_word: str = "<bos>", 
    max_new_tokens: int = 32, 
    temperature: float = 0.9,
    post_process: bool = True,
    seed: int = None
) -> str:
    """
    Autoregressive Word Generation.
    
    Starts from a given initial word, samples tokens using temperature scaling,
    and formats the output text.
    
    WordFlow: Word-Level Language Modeling with RNNs GiTHub.com/HooM4N/WordFlow
    """
    if seed is not None:
        torch.manual_seed(seed)
        
    model.eval()
    hidden = model.init_hidden(batch_size=1)
    
    unk_id = tokenizer.token_to_id("<unk>")
    eos_id = tokenizer.token_to_id("<eos>")
    init_idx = tokenizer.token_to_id(init_word)
    
    if init_idx == unk_id:
        print(f"*** '{init_word}' is not in vocab, picking a random initial word... ***")
        init_idx = torch.randint(low=5, high=tokenizer.get_vocab_size(), size=(1,)).item()
        init_word = tokenizer.id_to_token(init_idx)
        
    input_ = torch.tensor([[init_idx]], dtype=torch.long, device=device)
    generated_words = [init_word]
    
    for _ in range(max_new_tokens):
        logits, hidden = model(input_, hidden)
        
        # model outputs (N, vocab_size, L) -> we want the last step (L)
        last_step_logits = logits[0, :, -1] 
        
        probs = (last_step_logits / temperature).softmax(dim=0).cpu()

        while True:
            token_idx = torch.multinomial(probs, 1).item()
            if token_idx != unk_id: 
                break
        
        if token_idx == eos_id:
            break
            
        input_.fill_(token_idx)
        generated_words.append(tokenizer.id_to_token(token_idx))
        
    text = " ".join(generated_words)
    
    if post_process:
        text = text.replace(" \n ", "\n").replace("<eos>", "")
        
    return text
    
@torch.no_grad()
def predict_next_word(
    model: WordFlowModel, 
    tokenizer: Tokenizer, 
    device: torch.device, 
    context: str = "it is", 
    top_k: int = 5,
) -> dict[str, float]:
    """
    Next Word Probability Prediction.
    
    Returns top k candidate words with their probabilities given a context phrase.
    
    WordFlow: Word-Level Language Modeling with RNNs GiTHub.com/HooM4N/WordFlow
    """
    model.eval()
    hidden = model.init_hidden(batch_size=1)

    context = text_preprocessor(context)
    context_ids = tokenizer.encode(context)
    
    input_ = torch.tensor([context_ids], dtype=torch.long, device=device)
    logits, _ = model(input_, hidden) 
    
    # Extract the logits for the final time step
    last_step_logits = logits[0, :, -1]
    
    top_probs, top_ids = torch.topk(last_step_logits.softmax(dim=0).cpu(), top_k)
    
    return {
        tokenizer.id_to_token(i.item()): round(p.item(), 2) 
        for i, p in zip(top_ids, top_probs)
    }