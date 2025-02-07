import os
import random
import numpy as np
import pickle
import torch
from preprocess import load_imdb_data, build_vocabulary, vectorize_texts
from adaboost import adaboost_train, adaboost_predict


SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# Convert data to PyTorch tensor and move to GPU
def to_tensor(data):
    """Converts NumPy arrays to PyTorch tensors and moves to GPU."""
    if isinstance(data, torch.Tensor):  # If already tensor, just move to GPU
        return data.to("cuda")
    return torch.tensor(data, dtype=torch.float32, device="cuda")


def main():
    # Hyperparameters
    T_values = [50, 100, 150]  # Number of boosting iterations
    m_values = [2000, 3000, 5000]  # Vocabulary size
    n_most_values = [20, 50]  # Most frequent words removed
    k_rarest_values = [20, 50]  # Rarest words removed

    best_acc = 0
    best_params = {}

    # Load Training Data
    print("Loading training data...")
    texts, labels = load_imdb_data(split='train', root='../data/aclImdb')
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

    # Try Different Hyperparameter Combinations
    for T in T_values:
        for m in m_values:
            for n_most in n_most_values:
                for k_rarest in k_rarest_values:
                    print(f"\n--- Training with T={T}, m={m}, n_most={n_most}, k_rarest={k_rarest} ---")

                    # Build Vocabulary from Training Data
                    print("Building vocabulary...")
                    vocab = build_vocabulary(train_texts, train_labels, n_most=n_most, k_rarest=k_rarest, m=m)
                    # print(f"Vocabulary size: {len(vocab)}")

                    # Vectorize Texts
                    print("Vectorizing texts...")
                    X_train = vectorize_texts(train_texts, vocab)
                    X_dev = vectorize_texts(dev_texts, vocab)
                    y_train = np.array(train_labels)
                    y_dev = np.array(dev_labels)

                    # Convert and Move Data to GPU
                    X_train = to_tensor(X_train)
                    y_train = to_tensor(y_train)
                    X_dev = to_tensor(X_dev)
                    y_dev = to_tensor(y_dev)

                    # Train AdaBoost Classifier
                    print("Training AdaBoost classifier...")
                    stumps = adaboost_train(X_train, y_train, T)

                    # Evaluate on the Development Set
                    dev_preds = adaboost_predict(X_dev, stumps)
                    dev_acc = torch.mean((dev_preds == y_dev).float())
                    # dev_acc = np.mean(dev_preds == y_dev)
                    print(f"Development Accuracy: {dev_acc * 100:.2f}%")

                    if (dev_acc > best_acc):
                        best_acc = dev_acc
                        best_params = {'T': T, 'm': m, 'n_most': n_most, 'k_rarest': k_rarest}

                        # Save the Model and Vocabulary
                        results_dir = os.path.join('..', 'results')
                        os.makedirs(results_dir, exist_ok=True)
                        model_path = os.path.join(results_dir, 'adaboost_model.pkl')
                        vocab_path = os.path.join(results_dir, 'vocab.pkl')
                        with open(model_path, 'wb') as f:
                            pickle.dump(stumps, f)
                        with open(vocab_path, 'wb') as f:
                            pickle.dump(vocab, f)
                        print(f"Model and vocabulary saved to {model_path} and {vocab_path}.")

    print("\n--- Best Hyperparameters ---")
    print(best_params)

if __name__ == "__main__":
    main()
