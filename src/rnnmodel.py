import torch
import torch.nn as nn


class StackedBiRNN(nn.Module):
    def __init__(self, embedding_matrix, hidden_dim=128, num_layers=2, dropout=0.5, num_classes=2):
        """
        Args:
          embedding_matrix: a NumPy array of shape (vocab_size, embedding_dim) with pre-trained embeddings.
          hidden_dim: Number of hidden units in each direction of the GRU.
          num_layers: Number of stacked bidirectional GRU layers.
          dropout: Dropout probability (applied after the RNN, before the classification layer).
          num_classes: Number of output classes.
        """
        super(StackedBiRNN, self).__init__()
        vocab_size, embedding_dim = embedding_matrix.shape
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        # Initialize with pre-trained embeddings.
        self.embedding.weight = nn.Parameter(torch.tensor(embedding_matrix, dtype=torch.float32))
        # Optionally freeze embeddings:
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
        self.dropout = nn.Dropout(dropout)
        # The fully-connected layer takes the pooled representation (hidden_dim * 2 because bidirectional)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)


    def forward(self, x):
        """x: LongTensor of shape [batch_size, seq_len] with token indices."""

        # Convert word indices to embeddings.
        embeds = self.embedding(x)  # [batch_size, seq_len, embedding_dim]
        # Pass through GRU.
        gru_out, _ = self.gru(embeds)  # [batch_size, seq_len, hidden_dim*2]
        # Global max pooling over the sequence dimension.
        pooled, _ = torch.max(gru_out, dim=1)  # [batch_size, hidden_dim*2]
        pooled = self.dropout(pooled)
        logits = self.fc(pooled)  # [batch_size, num_classes]
        return logits
