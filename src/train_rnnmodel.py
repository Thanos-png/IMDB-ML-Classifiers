import os
import random
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from preprocess import load_imdb_data, build_vocabulary, vectorize_texts
from rnnmodel import StackedBiRNN
from utils import to_tensor


# For reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


def main():
    # Hyperparameters
    num_epochs = 10
    lr = 0.001  # Learning rate
    embedding_dim = 300
    hidden_dim = 128
    num_layers = 2
    dropout = 0.5
    batch_size = 64

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

    # Vectorize Texts
    X_train = vectorize_texts(train_texts, vocab)
    X_dev = vectorize_texts(dev_texts, vocab)
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

    # Dummy embedding matrix (vocab_size, embedding_dim)
    embedding_matrix = np.random.rand(len(vocab), embedding_dim)

    # Instantiate the model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = StackedBiRNN(embedding_matrix, hidden_dim, num_layers, dropout, num_classes=2)
    model.to(device)

    # Define the optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

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

        avg_loss = epoch_loss / len(train_dataset)
        print(f"Epoch {epoch}/{num_epochs} - Loss: {avg_loss:.4f}")

        # Evaluate on Development Data
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_X, batch_y in dev_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = model(batch_X)
                _, preds = torch.max(outputs, dim=1)
                correct += (preds == batch_y).sum().item()
                total += batch_y.size(0)

        accuracy = correct / total * 100
        print(f"Dev Accuracy: {accuracy:.2f}%")

    # Save Model and Vocabulary
    results_dir = os.path.join('..', 'results')
    os.makedirs(results_dir, exist_ok=True)
    model_path = os.path.join(results_dir, 'rnn_model.pth')
    vocab_path = os.path.join(results_dir, 'vocab.pkl')
    torch.save(model.state_dict(), model_path)
    with open(vocab_path, 'wb') as f:
        pickle.dump(vocab, f)
    print(f"Model and vocabulary saved to {model_path} and {vocab_path}")


if __name__ == "__main__":
    main()
