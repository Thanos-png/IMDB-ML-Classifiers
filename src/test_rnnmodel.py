import os
import random
import pickle
import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from preprocess import load_imdb_data, vectorize_texts
from rnnmodel import StackedBiRNN
from utils import to_tensor


# For reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


def main():
    # Load the Saved Model and Vocabulary
    results_dir = os.path.join('..', 'results')
    model_path = os.path.join(results_dir, 'rnn_model.pth')
    vocab_path = os.path.join(results_dir, 'rnn_vocab.pkl')
    with open(vocab_path, 'rb') as f:
        vocab = pickle.load(f)

    # Embedding matrix (assumes same vocab size as training)
    embedding_matrix = np.random.rand(len(vocab), 300)

    # Instantiate model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = StackedBiRNN(embedding_matrix, hidden_dim=128, num_layers=2, dropout=0.5, num_classes=2)
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.to(device)
    model.eval()

    # Load and Vectorize Test Data
    print("Loading test data...")
    root = os.path.join("..", "data", "aclImdb")
    texts, labels = load_imdb_data(split='test', root=root)
    print(f"Loaded {len(texts)} test examples.")
    X_test = vectorize_texts(texts, vocab)
    y_test = np.array(labels)

    # Remap labels: -1 -> 0, 1 remains 1.
    y_test = np.where(y_test == -1, 0, 1)

    # Convert to PyTorch Tensors and create a DataLoader for batching
    X_test = to_tensor(X_test).long()
    y_test = to_tensor(y_test).long()
    test_dataset = TensorDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # Evaluate Model in Batches
    total_correct = 0
    total_samples = 0
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            outputs = model(batch_X)
            _, preds = torch.max(outputs, dim=1)
            total_correct += (preds == batch_y).sum().item()
            total_samples += batch_y.size(0)

    test_acc = total_correct / total_samples * 100
    print(f"Test Accuracy: {test_acc:.2f}%")


if __name__ == "__main__":
    main()
