import torch
import torch.nn.functional as F

from .model import WordFlowModel
from .tokenizer import Tokenizer
from .data import text_preprocessor

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
    Autoregressively generates a story from the WordFlow model.
    
    It begins generation by feeding the <eos> token to the model, which 
    acts as a natural delimiter signaling the model to start a new story.
    
    WordFlow: Word-Level Language Modeling with RNNs GiTHub.com/HooM4N/WordFlow
    """
    if seed is not None:
        torch.manual_seed(seed)
        
    model.eval()
    hidden = model.init_hidden(batch_size=1)
    
    # start with eos token to trigger a new story
    eos_id = tokenizer.token_to_id("<eos>")
    x = torch.tensor([[eos_id]], dtype=torch.long, device=device)
    
    # Feed initial token to build context
    logits, hidden = model(x, hidden)
    next_logits = logits[0, :, -1]

    generated_ids = []

    # autoregressive generation loop
    for _ in range(max_tokens):
  
        probs = F.softmax(next_logits.div(temperature), dim=0)
        next_id = torch.multinomial(probs, num_samples=1).item()
        
        # stop generation if eos token found
        if next_id == eos_id:
            break
            
        generated_ids.append(next_id)
        
        # generate next token
        x = torch.tensor([[next_id]], dtype=torch.long, device=device)
        logits, hidden = model(x, hidden)
        next_logits = logits[0, :, -1]
        
    # decode ids
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
    Extracts the learned semantic relationships from the model's Embedding layer.
    
    Calculates the Cosine Similarity between the target word's embedding vector 
    and every other word vector in the vocabulary to find the closest matches.
              
    WordFlow: Word-Level Language Modeling with RNNs GiTHub.com/HooM4N/WordFlow
    """
    word = word.lower().strip()
    word_id = tokenizer.token_to_id(word)
    
    # Handle Out-Of-Vocabulary words
    if word_id == getattr(tokenizer, "unk_id", -1) and word != getattr(tokenizer, "unk_token", "<unk>"):
        return {"error": f"Word '{word}' is not in the vocabulary."}
        
    # Extract the raw embedding matrix to CPU memory
    embeddings = model.embedding.weight.detach().cpu()
    
    # Grab the 1D vector for our target word and add a batch dimension: (1, embedding_dim)
    target_vec = embeddings[word_id].unsqueeze(0)
    
    # Calculate Cosine Similarity against the entire vocabulary matrix
    cos_sim = F.cosine_similarity(target_vec, embeddings, dim=1)
    
    # We get top_n + 1 because the most similar word to itself is always itself
    top_scores, top_indices = torch.topk(cos_sim, top_n + 1)
    
    results = {}
    for score, idx in zip(top_scores, top_indices):
        idx = idx.item()
        
        # Skip the original target word
        if idx == word_id:
            continue
            
        sim_word = tokenizer.id_to_token(idx)
        
        # Filter out weird special tokens from the results
        if sim_word not in tokenizer.special_tokens:
            results[sim_word] = round(score.item(), 3)
            
        # Stop once we have exactly top_n real words
        if len(results) == top_n:
            break
            
    return results
