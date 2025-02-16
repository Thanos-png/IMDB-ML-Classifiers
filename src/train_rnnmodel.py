import os
import random
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
from preprocess import load_imdb_data, build_vocabulary, tokenize
from rnnmodel import StackedBiRNN
from utils import to_tensor


# For reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


def numericalize_texts(texts: list[str], vocab: dict[str, int], max_len: int = 500) -> np.ndarray:
    """Convert texts into sequences of token indices and pad/truncate to max_len."""

    sequences = []
    for text in texts:
        tokens = tokenize(text)  # make sure tokenize is imported from preprocess.py
        # Convert tokens to indices; use 0 for unknown tokens.
        seq = [vocab.get(token, 0) for token in tokens]
        seq = seq[:max_len]
        if len(seq) < max_len:
            seq += [0] * (max_len - len(seq))
        sequences.append(seq)
    return np.array(sequences)


def main():
    # Hyperparameters
    embedding_dim_values = [300, 400]  # Dimension of the pre-trained word embeddings
    hidden_dim_values = [256, 512]  # Hidden dimension of the RNN
    num_layers_values = [2, 3, 4]  # Number of stacked RNN layers
    dropout_values = [0.2, 0.3]  # Dropout probability
    lr_values = [0.0001, 0.0002, 0.0005]  # Learning rate
    num_epochs_values = [10, 15]  # Number of epochs

    batch_size = 64  # Number of examples per batch
    max_seq_len = 500  # Maximum sequence length for the RNN

    i = 0
    best_acc = 0
    best_params = {}

    # Load Training Data
    print("Loading training data...")
    root = os.path.join("..", "data", "aclImdb")
    texts, labels = load_imdb_data(split='train', root=root)
    print(f"Loaded {len(texts)} training examples.")

    # Split into Training and Development Sets (80/20 split)
    indices = list(range(len(texts)))
    random.shuffle(indices)
    split_index = int(0.8 * len(texts))
    train_indices = indices[:split_index]
    dev_indices = indices[split_index:]

    train_texts = [texts[i] for i in train_indices]
    train_labels = [labels[i] for i in train_indices]
    dev_texts = [texts[i] for i in dev_indices]
    dev_labels = [labels[i] for i in dev_indices]

    # Build Vocabulary
    vocab = build_vocabulary(train_texts, train_labels)

    # Create sequences
    X_train = numericalize_texts(train_texts, vocab, max_len=max_seq_len)
    X_dev = numericalize_texts(dev_texts, vocab, max_len=max_seq_len)
    y_train = np.array(train_labels)
    y_dev = np.array(dev_labels)

    # Remap labels: -1 -> 0, 1 stays 1.
    y_train = np.where(np.array(train_labels) == -1, 0, 1)
    y_dev = np.where(np.array(dev_labels) == -1, 0, 1)

    # Convert to PyTorch Tensors and Move Data to GPU if available
    X_train = to_tensor(X_train).long()
    y_train = to_tensor(y_train).long()
    X_dev = to_tensor(X_dev).long()
    y_dev = to_tensor(y_dev).long()

    # Create DataLoaders for batching
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    dev_dataset = TensorDataset(X_dev, y_dev)
    dev_loader = DataLoader(dev_dataset, batch_size=batch_size)

    # Try Different Hyperparameter Combinations
    for embedding_dim in embedding_dim_values:
        for hidden_dim in hidden_dim_values:
            for num_layers in num_layers_values:
                for dropout in dropout_values:
                    for lr in lr_values:
                        # Embedding matrix (vocab_size, embedding_dim)
                        embedding_matrix = np.random.rand(len(vocab), embedding_dim)

                        # Instantiate the model
                        device = "cuda" if torch.cuda.is_available() else "cpu"
                        model = StackedBiRNN(embedding_matrix, hidden_dim, num_layers, dropout, num_classes=2)
                        model.to(device)

                        # Define the optimizer and loss function
                        optimizer = optim.Adam(model.parameters(), lr=lr)
                        criterion = nn.CrossEntropyLoss()

                        # For plotting loss curves
                        train_loss_history = []
                        dev_loss_history = []

                        # Try Different num_epochs Combinations
                        for num_epochs in num_epochs_values:
                            print(f"\n--- Training with embedding_dim={embedding_dim}, hidden_dim={hidden_dim}, num_layers={num_layers}, dropout={dropout}, num_epochs={num_epochs}, lr={lr} ---")

                            # Training Loop
                            for epoch in range(1, num_epochs + 1):
                                model.train()
                                epoch_loss = 0.0
                                for batch_X, batch_y in train_loader:
                                    batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                                    optimizer.zero_grad()
                                    outputs = model(batch_X)
                                    loss = criterion(outputs, batch_y)
                                    loss.backward()
                                    optimizer.step()
                                    epoch_loss += loss.item() * batch_X.size(0)

                                avg_train_loss = epoch_loss / len(train_dataset)
                                train_loss_history.append(avg_train_loss)
                                print(f"Epoch {epoch}/{num_epochs} - Loss: {avg_train_loss:.4f}")

                                # Evaluate on Development Data
                                model.eval()
                                dev_epoch_loss = 0.0
                                correct = 0
                                total = 0
                                with torch.no_grad():
                                    for batch_X, batch_y in dev_loader:
                                        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                                        outputs = model(batch_X)
                                        loss_dev = criterion(outputs, batch_y)
                                        dev_epoch_loss += loss_dev.item() * batch_X.size(0)
                                        max_values, preds = torch.max(outputs, dim=1)
                                        correct += (preds == batch_y).sum().item()
                                        total += batch_y.size(0)

                                avg_dev_loss = dev_epoch_loss / len(dev_dataset)
                                dev_loss_history.append(avg_dev_loss)
                                dev_acc = correct / total * 100
                                print(f"Dev Accuracy: {dev_acc:.2f}%")

                                if (dev_acc > best_acc):
                                    best_acc = dev_acc
                                    best_params = {'embedding_dim': embedding_dim, 'hidden_dim': hidden_dim, 'num_layers': num_layers, 'dropout': dropout, 'lr': lr, 'num_epochs': num_epochs}

                                    # Save Model and Vocabulary
                                    results_dir = os.path.join('..', 'results')
                                    os.makedirs(results_dir, exist_ok=True)
                                    model_path = os.path.join(results_dir, 'rnn_model.pth')
                                    vocab_path = os.path.join(results_dir, 'rnn_vocab.pkl')
                                    torch.save(model.state_dict(), model_path)
                                    with open(vocab_path, 'wb') as f:
                                        pickle.dump(vocab, f)
                                    print(f"\nRNN Model and vocabulary saved to {model_path} and {vocab_path}")

                            # Create the results/plots directory if it doesn't exist
                            plots_dir = os.path.join('..', 'results', 'plots')
                            os.makedirs(plots_dir, exist_ok=True)
                            plot_path = os.path.join(plots_dir, f"train_rnnmodel_{i}.png")
                            i += 1

                            # Plot training and development loss curves
                            plt.figure(figsize=(10, 6))
                            plt.plot(range(1, len(train_loss_history) + 1), train_loss_history, label='Train Loss', marker='o')
                            plt.plot(range(1, len(dev_loss_history) + 1), dev_loss_history, label='Dev Loss', marker='s')
                            plt.xlabel('Epoch')
                            plt.ylabel('Loss')
                            plt.title(f'Loss Curves (embedding_dim={embedding_dim}, hidden_dim={hidden_dim}, num_layers={num_layers}, dropout={dropout}, lr={lr})')
                            plt.legend()
                            plt.grid(True)
                            plt.savefig(plot_path)
                            print(f"Plot saved at {plot_path}")
                            # plt.show()

    print("\n--- Best Hyperparameters ---")
    print(best_params)


if __name__ == "__main__":
    main()
