import os
import random
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
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
    dev_indices   = indices[split_index:]

    train_texts = [texts[i] for i in train_indices]
    train_labels = [labels[i] for i in train_indices]
    dev_texts   = [texts[i] for i in dev_indices]
    dev_labels  = [labels[i] for i in dev_indices]

    # Build Vocabulary
    vocab = build_vocabulary(train_texts, train_labels)

    # Vectorize Texts
    X_train = vectorize_texts(train_texts, vocab)
    X_dev = vectorize_texts(dev_texts, vocab)
    y_train = np.array(train_labels)
    y_dev = np.array(dev_labels)

    # Convert to PyTorch Tensors and Move Data to GPU if available
    X_train = to_tensor(X_train).long()
    y_train = to_tensor(y_train)
    X_dev = to_tensor(X_dev).long()
    y_dev = to_tensor(y_dev)

    # Dummy embedding matrix (vocab_size, embedding_dim)
    embedding_matrix = np.random.rand(len(vocab), embedding_dim)

    # Instantiate the model
    model = StackedBiRNN(embedding_matrix, hidden_dim, num_layers, dropout, num_classes=2)
    model.to("cuda" if torch.cuda.is_available() else "cpu")

    # Define the optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    # Training Loop
    for epoch in range(1, num_epochs + 1):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()

        print(f"Epoch {epoch}/{num_epochs} - Loss: {loss.item():.4f}")

        # Evaluate on Development Data
        model.eval()
        with torch.no_grad():
            outputs = model(X_dev)
            _, preds = torch.max(outputs, dim=1)
            accuracy = (preds == y_dev).float().mean().item() * 100
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
