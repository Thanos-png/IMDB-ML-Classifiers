import os
import random
import pickle
import torch
import numpy as np
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
    model = StackedBiRNN(embedding_matrix, hidden_dim=128, num_layers=2, dropout=0.5, num_classes=2)
    model.load_state_dict(torch.load(model_path))
    model.to("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    # Load and Vectorize Test Data
    print("Loading test data...")
    root = os.path.join("..", "data", "aclImdb")
    texts, labels = load_imdb_data(split='test', root=root)
    print(f"Loaded {len(texts)} test examples.")
    X_test = vectorize_texts(texts, vocab)
    y_test = np.array(labels)

    # Convert to PyTorch Tensors and Move Data to GPU if available
    X_test = to_tensor(X_test).long()
    y_test = to_tensor(y_test).long()

    # Evaluate Model
    with torch.no_grad():
        outputs = model(X_test)
        _, preds = torch.max(outputs, dim=1)
        accuracy = (preds == y_test).float().mean().item() * 100
    print(f"Test Accuracy: {accuracy:.2f}%")


if __name__ == "__main__":
    main()
