import torch
import torch.nn.functional as F

from .model import WordFlowModel
from .tokenizer import Tokenizer

@torch.no_grad()
def generate_story(
    model: WordFlowModel, 
    tokenizer: Tokenizer, 
    device: torch.device,
    max_tokens: int = 100, 
    temperature: float = 0.8,
    seed: int | None = None
) -> str:
    """
    Autoregressively generates text using the trained model.

    *WordFlow: Word-Level Language Modeling with RNNs GiTHub.com/HooM4N/WordFlow*
    """
    if seed is not None:
        torch.manual_seed(seed)
        
    model.eval()
    hidden = model.init_hidden(batch_size=1)
    
    eos_id = tokenizer.token_to_id("<eos>")
    x = torch.tensor([[eos_id]], dtype=torch.long, device=device)
    
    logits, hidden = model(x, hidden)
    next_logits = logits[0, :, -1]

    generated_ids = []

    for _ in range(max_tokens):
  
        probs = F.softmax(next_logits.div(temperature), dim=0)
        next_id = torch.multinomial(probs, num_samples=1).item()
        
        if next_id == eos_id:
            break
            
        generated_ids.append(next_id)
        
        x = torch.tensor([[next_id]], dtype=torch.long, device=device)
        logits, hidden = model(x, hidden)
        next_logits = logits[0, :, -1]
        
    text = tokenizer.decode(generated_ids)
    
    return text.replace("<breakline>", "\n").replace("<eos>", "").strip()

@torch.no_grad()
def get_similar_words(
    model: WordFlowModel, 
    tokenizer: Tokenizer, 
    word: str, 
    top_n: int = 5
) -> dict[str, float]:
    """
    Finds the most similar words to a target word using cosine similarity on embeddings.

    *WordFlow: Word-Level Language Modeling with RNNs GiTHub.com/HooM4N/WordFlow*
    """
    word = word.lower().strip()
    word_id = tokenizer.token_to_id(word)
    
    if word_id == getattr(tokenizer, "unk_id", -1) and word != getattr(tokenizer, "unk_token", "<unk>"):
        return {"error": f"Word '{word}' is not in the vocabulary."}
        
    embeddings = model.embedding.weight.detach().cpu()
    
    target_vec = embeddings[word_id].unsqueeze(0)
    
    cos_sim = F.cosine_similarity(target_vec, embeddings, dim=1)
    
    top_scores, top_indices = torch.topk(cos_sim, top_n + 1)
    
    results = {}
    for score, idx in zip(top_scores, top_indices):
        idx = idx.item()
        
        if idx == word_id:
            continue
            
        sim_word = tokenizer.id_to_token(idx)
        
        if sim_word not in tokenizer.special_tokens:
            results[sim_word] = round(score.item(), 3)
            
        if len(results) == top_n:
            break
            
    return results
