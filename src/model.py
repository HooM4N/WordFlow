import torch
import torch.nn as nn
import torch.nn.functional as F

class CausalLSTM(nn.Module):
    """
    =====================================================
    == CausalLSTM Model (GiTHUB.com/HooM4N/CausalLSTM) ==
    =====================================================
    - Embedding–classification head with weight tying
    - Linear projection added if embedding_dim != hidden_dim
    - Dropout applied to embeddings, LSTM outputs, and projection
    - Weight initialization follows "Regularizing and Optimizing LSTM Language Models"
    """
    def __init__(
        self, 
        vocab_size: int, 
        embedding_dim: int = 400, 
        hidden_dim: int = 768, 
        num_layers: int = 2,
        lstm_dropout_p: float = 0.25, 
        emb_dropout_p: float = 0.2, 
        out_dropout_p: float = 0.4,
        pretrained_embedding_matrix: torch.Tensor = None,
        freeze_pretrained_embeddings: bool = True,
        proj_nonlinearity: bool = False,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        self.proj_nonlinearity = proj_nonlinearity
        
        if pretrained_embedding_matrix is not None:
            assert isinstance(pretrained_embedding_matrix, torch.Tensor)
            assert pretrained_embedding_matrix.size() == (vocab_size, embedding_dim)
            self.embedding = nn.Embedding.from_pretrained(
                pretrained_embedding_matrix, 
                freeze = freeze_pretrained_embeddings
            )
        else:
            self.embedding = nn.Embedding(vocab_size, embedding_dim)
            
        self.emb_dropout = nn.Dropout(emb_dropout_p)
        self.lstm = nn.LSTM(
            embedding_dim, hidden_dim, num_layers, batch_first=True, dropout = lstm_dropout_p
        )
        self.out_dropout = nn.Dropout(out_dropout_p)
        self.fc = nn.Linear(embedding_dim, vocab_size)
        
        self.use_proj = True if hidden_dim != embedding_dim else False
        if self.use_proj:
            self.proj = nn.Linear(hidden_dim, embedding_dim)
        self.fc.weight = self.embedding.weight 
            
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        if self.use_proj:
            nn.init.uniform_(self.proj.weight, -initrange, initrange)
            nn.init.zeros_(self.proj.bias)
        nn.init.uniform_(self.embedding.weight, -initrange, initrange)

        ## TODO: should i keep head's bias trainable?
        nn.init.zeros_(self.fc.bias)

    def init_hidden(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
        w = next(self.parameters())
        h_0 = w.new_zeros((self.num_layers, batch_size, self.hidden_dim))
        c_0 = w.new_zeros((self.num_layers, batch_size, self.hidden_dim))
        return (h_0, c_0)

    def forward(
        self, 
        x: torch.Tensor, 
        hidden: tuple[torch.Tensor, torch.Tensor] = None
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        x = self.embedding(x) # (N, L, E)
        x = self.emb_dropout(x)
        x, hidden = self.lstm(x, hidden) # (N, L, H), ((num_layers, N, H), (num_layers, N, H))
        x = self.out_dropout(x)
        if self.use_proj:
            x = self.proj(x) # (N, L, E)
            if self.proj_nonlinearity:
                x = F.gelu(x)
            x = self.out_dropout(x)
        return self.fc(x).permute(0, 2, 1), hidden # (N, vocab_size, L), ((num_layers, N, H), (num_layers, N, H))

def detach_hidden(hidden: tuple[torch.Tensor, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    """Detach hidden states from the current graph."""
    if isinstance(hidden, torch.Tensor):
        return hidden.detach()
    return tuple(detach_hidden(h) for h in hidden)