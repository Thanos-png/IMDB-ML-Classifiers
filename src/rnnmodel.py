import torch
import torch.nn as nn
import numpy as np


class StackedBiRNN(nn.Module):
    def __init__(self, embedding_matrix: np.ndarray, hidden_dim=128, num_layers=2, dropout=0.5, num_classes=2) -> None:
        super(StackedBiRNN, self).__init__()
        vocab_size, embedding_dim = embedding_matrix.shape
        
        self.embedding: nn.Embedding = nn.Embedding(vocab_size, embedding_dim)
        # Initialize with pre-trained embeddings
        self.embedding.weight = nn.Parameter(torch.tensor(embedding_matrix, dtype=torch.float32))
        # Freeze embeddings:
        self.embedding.weight.requires_grad = False

        # Create a stacked bidirectional GRU.
        self.gru = nn.GRU(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0  # dropout only applies for num_layers > 1
        )
        self.dropout: nn.Dropout = nn.Dropout(dropout)
        # The fully-connected layer takes the pooled representation (hidden_dim * 2 because bidirectional)
        self.fc: nn.Linear = nn.Linear(hidden_dim * 2, num_classes)


    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """`X` is a LongTensor of shape [batch_size, seq_len] with token indices."""

        # Convert word indices to embeddings
        embeds: torch.Tensor = self.embedding(X)  # [batch_size, seq_len, embedding_dim]
        # Pass through GRU
        gru_out, hidden_state = self.gru(embeds)  # [batch_size, seq_len, hidden_dim*2]
        # Global max pooling
        pooled, pooled_indices = torch.max(gru_out, dim=1)  # [batch_size, hidden_dim*2]
        pooled = self.dropout(pooled)
        logits: torch.Tensor = self.fc(pooled)  # [batch_size, num_classes]
        return logits
