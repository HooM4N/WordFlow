import torch
import torch.nn as nn

class WordFlowModel(nn.Module):
    """
    Recurrent Neural Network Model for Word-Level Language Modeling.
    
    Args:
        vocab_size (int): Size of the tokenizer's vocabulary.
        embedding_dim (int, optional): Dimensionality of word embeddings. Defaults to 300.
        hidden_dim (int, optional): Dimensionality of GRU hidden states. Must equal 
                                    `embedding_dim` if `tie_weights` is True. Defaults to 300.
        num_layers (int, optional): Number of stacked GRU layers. Defaults to 1.
        rnn_dropout_p (float, optional): Dropout between GRU layers. Defaults to 0.25.
        emb_dropout_p (float, optional): Dropout after Embedding layer. Defaults to 0.2.
        out_dropout_p (float, optional): Dropout before final Linear layer. Defaults to 0.2.
        tie_weights (bool, optional): Whether to share embedding and output weights. Defaults to True.
        padding_idx (int, optional): Vocabulary padding index. Defaults to 0.
        
    WordFlow: Word-Level Language Modeling with RNNs GiTHub.com/HooM4N/WordFlow
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

        # classification Head
        self.fc = nn.Linear(hidden_dim, vocab_size, bias=not tie_weights) 
        if tie_weights: 
            self.fc.weight = self.embedding.weight 
            
        self._init_weights()

    def _init_weights(self):
        """Initializes model weights uniformly."""
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
        """
        Forward pass of the WordFlow Model.
        
        Args:
            x (torch.Tensor): Input tensor of token IDs, shape (N, L).
            hidden (torch.Tensor, optional): Previous GRU hidden state. 
                                             Defaults to None.
            
        Returns:
            tuple[torch.Tensor, torch.Tensor]:
                - Output logits of shape (N, vocab_size, L)
                - Updated hidden state of shape (num_layers, N, hidden_dim)
        """
        # embedding
        x = self.embedding(x)  # (N, L, E)
        x = self.emb_dropout(x)

        # rnn
        x, hidden = self.rnn(x, hidden)  # x: (N, L, H), hidden: (num_layers, N, H)
        
        # classification head
        x = self.out_dropout(x)
        logits = self.fc(x) # (N, L, vocab_size)
        
        # cross entropy loss expects (N, vocab_size, L)
        logits = logits.permute(0, 2, 1) # (N, vocab_size, L)
        
        return logits, hidden