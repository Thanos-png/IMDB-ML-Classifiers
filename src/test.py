import os
import pickle
import numpy as np
import torch
import matplotlib.pyplot as plt
from preprocess import load_imdb_data, vectorize_texts
from adaboost import adaboost_predict


def to_tensor(data):
    """Converts NumPy arrays to PyTorch tensors and moves to GPU if available."""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if isinstance(data, torch.Tensor):  # If already tensor, just move to GPU
        return data.to(device)
    return torch.tensor(data, dtype=torch.float32, device=device)


def compute_class_metrics(y_true, y_pred, cls):
    """Computes precision, recall, and F1 for a given class."""
    
    # Compute true positives, false positives, and false negatives
    tp = torch.sum((y_true == cls) & (y_pred == cls)).item()
    fp = torch.sum((y_true != cls) & (y_pred == cls)).item()
    fn = torch.sum((y_true == cls) & (y_pred != cls)).item()

    # Compute precision, recall, and F1
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return precision, recall, f1


def plot_test_results(categories, precision, recall, f1):
    """Plots precision, recall, and F1-score for test evaluation."""

    x = np.arange(len(categories))
    width = 0.25  # Bar width
    
    plt.figure(figsize=(8, 5))
    plt.bar(x - width, precision, width, label='Precision', color='blue')
    plt.bar(x, recall, width, label='Recall', color='green')
    plt.bar(x + width, f1, width, label='F1-score', color='red')
    
    plt.xlabel("Categories")
    plt.ylabel("Score")
    plt.title("Test Evaluation Metrics")
    plt.xticks(x, categories)
    plt.legend()
    plt.grid(axis='y')
    plt.show()


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

    # Compute Evaluation Metrics for Each Class
    prec_pos, rec_pos, f1_pos = compute_class_metrics(y_test, preds, 1)
    prec_neg, rec_neg, f1_neg = compute_class_metrics(y_test, preds, -1)

    # Compute True Positives, False Positives, and False Negatives for Each Class
    tp_pos = torch.sum((y_test == 1) & (preds == 1)).item()
    tp_neg = torch.sum((y_test == -1) & (preds == -1)).item()
    fp_pos = torch.sum((y_test == -1) & (preds == 1)).item()
    fp_neg = torch.sum((y_test == 1) & (preds == -1)).item()
    fn_pos = torch.sum((y_test == 1) & (preds == -1)).item()
    fn_neg = torch.sum((y_test == -1) & (preds == 1)).item()

    # Compute Total Counts
    total_tp = tp_pos + tp_neg
    total_fp = fp_pos + fp_neg
    total_fn = fn_pos + fn_neg

    # Compute Micro-Averaged Metrics
    micro_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    micro_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    micro_f1 = 2 * micro_precision * micro_recall / (micro_precision + micro_recall) if (micro_precision + micro_recall) > 0 else 0.0
    
    # Compute Macro-Averaged Metrics
    macro_precision = (prec_pos + prec_neg) / 2
    macro_recall = (rec_pos + rec_neg) / 2
    macro_f1 = (f1_pos + f1_neg) / 2

    categories = ["Positive", "Negative"]
    precision_values = [prec_pos, prec_neg]
    recall_values = [rec_pos, rec_neg]
    f1_values = [f1_pos, f1_neg]
    
    # Print the Evaluation Results
    print("\nEvaluation Metrics on Test Data:")
    print("{:<10} {:<10} {:<10} {:<10}".format("Category", "Precision", "Recall", "F1"))
    print("{:<10} {:<10.4f} {:<10.4f} {:<10.4f}".format("Positive", prec_pos, rec_pos, f1_pos))
    print("{:<10} {:<10.4f} {:<10.4f} {:<10.4f}".format("Negative", prec_neg, rec_neg, f1_neg))
    print("\nMicro-averaged: Precision: {:.4f}, Recall: {:.4f}, F1: {:.4f}".format(
        micro_precision, micro_recall, micro_f1))
    print("Macro-averaged: Precision: {:.4f}, Recall: {:.4f}, F1: {:.4f}".format(
        macro_precision, macro_recall, macro_f1))

    # Plot the results
    plot_test_results(categories, precision_values, recall_values, f1_values)

if __name__ == "__main__":
    main()
