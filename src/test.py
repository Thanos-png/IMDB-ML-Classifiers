import os
import pickle
import numpy as np
import torch
from preprocess import load_imdb_data, vectorize_texts
from adaboost import adaboost_predict


def to_tensor(data):
    """Converts NumPy arrays to PyTorch tensors and moves to GPU if available."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if isinstance(data, torch.Tensor):  # If already tensor, just move to GPU
        return data.to(device)
    return torch.tensor(data, dtype=torch.float32, device=device)


def main():
    # Load the Saved Model and Vocabulary
    results_dir = os.path.join('..', 'results')
    model_path = os.path.join(results_dir, 'adaboost_model.pkl')
    vocab_path = os.path.join(results_dir, 'vocab.pkl')
    with open(model_path, 'rb') as f:
        stumps = pickle.load(f)
    with open(vocab_path, 'rb') as f:
        vocab = pickle.load(f)

    # Load and Vectorize Test Data
    print("Loading test data...")
    texts, labels = load_imdb_data(split='test', root='../data/aclImdb')
    print(f"Loaded {len(texts)} test examples.")
    X_test = vectorize_texts(texts, vocab)
    y_test = np.array(labels)

    # Convert to PyTorch Tensors and Move Data to GPU
    X_test = to_tensor(X_test)
    y_test = to_tensor(y_test)

    # Evaluate
    preds = adaboost_predict(X_test, stumps)
    test_acc = torch.mean((preds == y_test).float())
    # test_acc = np.mean(preds == y_test)
    print(f"Test Accuracy: {test_acc * 100:.2f}%")

if __name__ == "__main__":
    main()
