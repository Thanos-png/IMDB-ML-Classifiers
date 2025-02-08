import os
import random
import numpy as np
import pickle
import torch
import matplotlib.pyplot as plt
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


def compute_metrics_for_class(y_true, y_pred, target=1):
    """Computes precision, recall, and F1 for the given target class."""

    # Compute true positives, false positives, and false negatives
    tp = torch.sum((y_true == target) & (y_pred == target)).item()
    fp = torch.sum((y_true != target) & (y_pred == target)).item()
    fn = torch.sum((y_true == target) & (y_pred != target)).item()

    # Compute precision, recall, and F1
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return precision, recall, f1


def plot_learning_curve(train_sizes, train_prec, train_rec, train_f1, dev_prec, dev_rec, dev_f1):
    """Plots precision, recall, and F1-score for train and dev sets as training size increases."""

    plt.figure(figsize=(10, 6))
    
    plt.plot(train_sizes, train_f1, marker='o', linestyle='-', label="Train F1-score", color='blue')
    plt.plot(train_sizes, dev_f1, marker='s', linestyle='-', label="Dev F1-score", color='green')
    plt.plot(train_sizes, train_prec, marker='^', linestyle='-', label="Train Precision", color='red')
    plt.plot(train_sizes, dev_prec, marker='v', linestyle='-', label="Dev Precision", color='orange')
    plt.plot(train_sizes, train_rec, marker='d', linestyle='-', label="Train Recall", color='purple')
    plt.plot(train_sizes, dev_rec, marker='x', linestyle='-', label="Dev Recall", color='brown')
    
    plt.xlabel("Training Set Size")
    plt.ylabel("Score")
    plt.title("Learning Curve: Train vs Dev Metrics")
    plt.legend()
    plt.grid(True)
    plt.show()


def main():
    # Hyperparameters
    T_values = [200]  # Number of boosting iterations
    m_values = [5000]  # Vocabulary size
    n_most_values = [50]  # Most frequent words removed
    k_rarest_values = [50]  # Rarest words removed

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

                    # Convert to PyTorch Tensors and Move Data to GPU
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
                        print(f"Model and vocabulary saved to {model_path} and {vocab_path}")

                        # Learning Curve Experiment
                        # We will train models on increasing fractions of the training data and evaluate
                        # precision, recall, and F1 for the positive class on both the training subset and full dev set.
                        print("\nRunning learning curve experiment (evaluating for positive class)...")
                        fractions = [0.1, 0.2, 0.4, 0.6, 0.8, 1.0]
                        train_sizes, train_prec, train_rec, train_f1 = [], [], [], []
                        dev_prec, dev_rec, dev_f1 = [], [], []
                        print("{:<10} {:<15} {:<15} {:<15} {:<15} {:<15} {:<15}".format(
                            "Size", "Train Prec", "Train Rec", "Train F1", "Dev Prec", "Dev Rec", "Dev F1"))

                        # For reproducibility, we use the first N examples of the (already shuffled) training data.
                        for frac in fractions:
                            subset_size = int(frac * len(train_texts))
                            sub_train_texts = train_texts[:subset_size]
                            sub_train_labels = to_tensor(train_labels[:subset_size])
                            # sub_train_labels = np.array(train_labels[:subset_size])
                            X_sub_train = vectorize_texts(sub_train_texts, vocab)

                            # Train AdaBoost on the subset (suppress iteration prints by setting verbose=False)
                            stumps_subset = adaboost_train(X_sub_train, sub_train_labels, T, verbose=False)

                            # Evaluate on the training subset and the full development set.
                            train_preds_subset = adaboost_predict(X_sub_train, stumps_subset)
                            dev_preds_subset = adaboost_predict(X_dev, stumps_subset)

                            prec_train, rec_train, f1_train = compute_metrics_for_class(sub_train_labels, train_preds_subset, target=1)
                            prec_dev, rec_dev, f1_dev = compute_metrics_for_class(y_dev, dev_preds_subset, target=1)
                            print("{:<10} {:<15.4f} {:<15.4f} {:<15.4f} {:<15.4f} {:<15.4f} {:<15.4f}".format(
                                subset_size, prec_train, rec_train, f1_train, prec_dev, rec_dev, f1_dev))

                            train_sizes.append(subset_size)
                            train_prec.append(prec_train)
                            train_rec.append(rec_train)
                            train_f1.append(f1_train)
                            dev_prec.append(prec_dev)
                            dev_rec.append(rec_dev)
                            dev_f1.append(f1_dev)

                        # Plot learning curve
                        plot_learning_curve(train_sizes, train_prec, train_rec, train_f1, dev_prec, dev_rec, dev_f1)

    print("\n--- Best Hyperparameters ---")
    print(best_params)

if __name__ == "__main__":
    main()
