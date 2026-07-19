import torch
import torch.nn.functional as F

from .model import WordFlowModel
from .tokenizer import Tokenizer
from .data import text_preprocessor

@torch.no_grad()
def generate_text(
    model: WordFlowModel, 
    tokenizer: Tokenizer, 
    device: torch.device,
    prompt: str | None = None, 
    max_tokens: int = 100, 
    temperature: float = 0.8,
    top_k: int = 10,
    seed: int | None = None
) -> str:
    """
    Autoregressively generates text from the WordFlow model.
    
    If a prompt is provided, it passes the sequence through the RNN to build 
    contextual memory (the hidden state) before generating new tokens. If no 
    prompt is provided, it begins with a random vocabulary token.
    
    Args:
        model (WordFlowModel): The trained GRU language model.
        tokenizer (Tokenizer): The project's tokenizer instance.
        device (torch.device): Compute device (cpu or cuda).
        prompt (str, optional): Seed text to start generation. Defaults to None.
        max_tokens (int): Maximum number of words to generate.
        temperature (float): Scales logits before softmax. Higher = more random.
        top_k (int): Limits sampling to the top K most likely tokens.
        seed (int, optional): Random seed for reproducibility.
        
    Returns:
        str: The fully generated text string.
        
    WordFlow: Word-Level Language Modeling with RNNs GiTHub.com/HooM4N/WordFlow
    """
    if seed is not None:
        torch.manual_seed(seed)
        
    model.eval()
    hidden = model.init_hidden(batch_size=1)
    
    # 1. Prepare the initial sequence
    if prompt:
        # Preprocess and encode the user's prompt
        clean_prompt = text_preprocessor(prompt)
        input_ids = tokenizer.encode(clean_prompt)
        
        # Feed the entire prompt into the model to build the hidden state memory
        x = torch.tensor([input_ids], dtype=torch.long, device=device)
        logits, hidden = model(x, hidden)
        
        # We only care about predicting the word that comes AFTER the prompt
        next_logits = logits[0, :, -1]
        generated_ids = input_ids.copy()
        
    else:
        # No prompt provided: pick a random valid starting word
        vocab_size = tokenizer.get_vocab_size()
        start_id = torch.randint(low=5, high=vocab_size, size=(1,)).item()
        
        x = torch.tensor([[start_id]], dtype=torch.long, device=device)
        logits, hidden = model(x, hidden)
        
        next_logits = logits[0, :, -1]
        generated_ids = [start_id]

    # 2. Autoregressive Generation Loop
    for _ in range(max_tokens):
        # Apply temperature scaling
        next_logits = next_logits / temperature
        
        # Apply Top-K filtering to remove long-tail gibberish
        if top_k is not None:
            v, _ = torch.topk(next_logits, top_k)
            next_logits[next_logits < v[-1]] = -float('Inf')
            
        # Convert to probabilities and sample
        probs = F.softmax(next_logits, dim=0)
        next_id = torch.multinomial(probs, num_samples=1).item()
        
        generated_ids.append(next_id)
        
        # Stop early if the model generates the End-Of-Story token
        if next_id == tokenizer.token_to_id("<eos>"):
            break
            
        # Prepare the newly generated token for the next loop
        x = torch.tensor([[next_id]], dtype=torch.long, device=device)
        logits, hidden = model(x, hidden)
        next_logits = logits[0, :, -1]
        
    # 3. Decode and clean up formatting
    text = tokenizer.decode(generated_ids)
    
    # Restore actual newlines and remove padding/EOS markers
    text = text.replace(" \n ", "\n").replace("<eos>", "").strip()
    return text


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
    
    Args:
        model (WordFlowModel): The trained GRU language model.
        tokenizer (Tokenizer): The project's tokenizer instance.
        word (str): The target word to search neighbors for.
        top_n (int): The number of similar words to return.
        
    Returns:
        dict: A dictionary mapping similar words to their cosine similarity score 
              (e.g., {"king": 0.85, "royalty": 0.72}).
              
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
    
    # We get top_n + 1 because the most similar word to 'apple' is always 'apple'
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