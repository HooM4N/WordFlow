import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class WordFlowModel(nn.Module):
    """
    =================================================
    == WordFlow Model (GiTHUB.com/HooM4N/WordFlow) ==
    =================================================
    """
    def __init__(
        self, 
        vocab_size: int, 
        rnn_type: str = "GRU",
        embedding_dim: int = 400, 
        hidden_dim: int = 768, 
        num_layers: int = 2,
        rnn_dropout_p: float = 0.25, 
        emb_dropout_p: float = 0.2, 
        out_dropout_p: float = 0.35,
        tie_weights: bool = True,
        pretrained_embedding_matrix: torch.Tensor = None,
        freeze_pretrained_embeddings: bool = True,
        proj_nonlinearity: bool = False,
        padding_idx: int = 0,
    ):
        super().__init__()
        assert rnn_type in ["LSTM", "GRU"]
        self.rnn_type = rnn_type
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.proj_nonlinearity = proj_nonlinearity
        self.tie_weights = tie_weights
        
        if pretrained_embedding_matrix is not None:
            assert isinstance(pretrained_embedding_matrix, torch.Tensor)
            assert pretrained_embedding_matrix.size(0) == vocab_size
            self.embedding = nn.Embedding.from_pretrained(
                pretrained_embedding_matrix, 
                freeze = freeze_pretrained_embeddings, 
                padding_idx = padding_idx
            )
        else:
            self.embedding = nn.Embedding(
                vocab_size, embedding_dim, padding_idx
            )
            
        self.embedding_dim = self.embedding.embedding_dim
        self.emb_dropout = nn.Dropout(emb_dropout_p)
        
        self.rnn = getattr(nn, rnn_type)(
            self.embedding_dim, hidden_dim, num_layers, batch_first=True, dropout = rnn_dropout_p
        )
        self.out_dropout = nn.Dropout(out_dropout_p)

        if tie_weights: 
            self.fc = nn.Linear(embedding_dim, vocab_size, bias=False) 
            self.fc.weight = self.embedding.weight 
            self.use_proj = hidden_dim != embedding_dim 
            if self.use_proj: 
                self.proj = nn.Linear(hidden_dim, embedding_dim) 
        else: 
            self.fc = nn.Linear(hidden_dim, vocab_size)
            self.use_proj = False
            
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        
        if self.use_proj:
            nn.init.uniform_(self.proj.weight, -initrange, initrange)
            nn.init.zeros_(self.proj.bias)
            
        if not self.tie_weights:
            nn.init.uniform_(self.fc.weight, -initrange, initrange)
            nn.init.zeros_(self.fc.bias)
            
        nn.init.uniform_(self.embedding.weight, -initrange, initrange)
        
    def init_hidden(
        self, batch_size: int
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        w = next(self.parameters())
        if self.rnn_type == "LSTM":
            return (w.new_zeros((self.num_layers, batch_size, self.hidden_dim)),
                    w.new_zeros((self.num_layers, batch_size, self.hidden_dim)))
        else:
            return w.new_zeros((self.num_layers, batch_size, self.hidden_dim))

    def forward(
        self, 
        x: torch.Tensor, # (N, L)
        hidden: torch.Tensor | tuple[torch.Tensor, torch.Tensor] = None, # GRU: (num_layers, N, H) 
        padding_mask: torch.Tensor = None, # (N, L) , 0 -> padding tokens & 1 -> valid tokens 
    ) -> tuple[torch.Tensor, torch.Tensor | tuple[torch.Tensor, torch.Tensor]]:

        # Embedding
        x = self.embedding(x) # (N, L, E)
        x = self.emb_dropout(x)

        # RNN
        if padding_mask is not None:
            x = pack_padded_sequence(
                x, padding_mask.sum(dim=1).cpu(), batch_first=True, enforce_sorted=False
            )
        x, hidden = self.rnn(x, hidden) # x (packed if padded): (N, L, H), hidden (GRU): (num_layers, N, H)
        if padding_mask is not None:
            x, _ =  pad_packed_sequence(
                x, batch_first=True, total_length=padding_mask.size(1)
            )
        
        # Projection
        if self.use_proj:
            x = self.proj(x) # (N, L, E)
            if self.proj_nonlinearity:
                x = F.gelu(x)
            
        # Classification Head
        x = self.out_dropout(x)
        return (
            self.fc(x).permute(0, 2, 1), # (N, vocab_size, L)
            hidden # (GRU): (num_layers, N, H)
        )

def detach_hidden(
    hidden: tuple[torch.Tensor, torch.Tensor]
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """
    Detach hidden states from the current graph
    """
    if isinstance(hidden, torch.Tensor):
        return hidden.detach()
    return tuple(detach_hidden(h) for h in hidden)