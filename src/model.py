import torch
import torch.nn as nn

class WordFlowModel(nn.Module):
    """
    A GRU-based recurrent neural network for word-level language modeling.

    *WordFlow: Word-Level Language Modeling with RNNs GiTHub.com/HooM4N/WordFlow*
    """
    def __init__(
        self, 
        vocab_size: int, 
        embedding_dim: int = 300, 
        hidden_dim: int = 300, 
        num_layers: int = 1,
        rnn_dropout_p: float = 0.25, 
        emb_dropout_p: float = 0.2, 
        out_dropout_p: float = 0.2,
        tie_weights: bool = True,
        padding_idx: int = 0,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.tie_weights = tie_weights
        
        if tie_weights and embedding_dim != hidden_dim:
            raise ValueError("When tie_weights=True, embedding_dim must equal hidden_dim.")
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx)
        self.emb_dropout = nn.Dropout(emb_dropout_p)
        
        self.rnn = nn.GRU(
            input_size=embedding_dim, 
            hidden_size=hidden_dim, 
            num_layers=num_layers, 
            batch_first=True, 
            dropout=rnn_dropout_p if num_layers > 1 else 0.0
        )
        self.out_dropout = nn.Dropout(out_dropout_p)

        self.fc = nn.Linear(hidden_dim, vocab_size, bias=not tie_weights) 
        if tie_weights: 
            self.fc.weight = self.embedding.weight 
            
        self._init_weights()

    def _init_weights(self):
        initrange = 0.1
        if not self.tie_weights:
            nn.init.uniform_(self.fc.weight, -initrange, initrange)
            nn.init.zeros_(self.fc.bias)
            
        nn.init.uniform_(self.embedding.weight, -initrange, initrange)
        
    def init_hidden(self, batch_size: int) -> torch.Tensor:
        w = next(self.parameters())
        return w.new_zeros((self.num_layers, batch_size, self.hidden_dim))

    def forward(
        self, 
        x: torch.Tensor, 
        hidden: torch.Tensor = None, 
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.embedding(x)  # (N, L, E)
        x = self.emb_dropout(x)

        x, hidden = self.rnn(x, hidden)  # (N, L, H)
        
        x = self.out_dropout(x)
        logits = self.fc(x) # (N, L, vocab_size)
        
        logits = logits.permute(0, 2, 1) # (N, vocab_size, L)
        
        return logits, hidden
