import os
import random
import pickle
import numpy as np
import torch
import matplotlib.pyplot as plt
from preprocess import load_imdb_data, vectorize_texts
from adaboost import adaboost_predict
from utils import to_tensor, compute_metrics_for_class_sklearn, compute_metrics_for_class_torch

from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier


# For reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


def plot_test_results_A(categories, precision, recall, f1):
    """Plots precision, recall, and F1-score for test evaluation."""

    x = np.arange(len(categories))
    width = 0.25
    
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

    # Save the plot
    plots_dir = os.path.join('..', 'results', 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    plt.savefig(os.path.join(plots_dir, 'test-1.png'))
    print(f"Plot saved at {os.path.join(plots_dir, 'test-1.png')}")

    plt.show()


def plot_test_results_B(categories, prec_values, rec_values, f1_values, sklearn_precs, sklearn_recs, sklearn_f1s):
    """Plots bar charts of precision, recall, and F1 for test evaluation for both implementations."""

    x = np.arange(len(categories))
    width = 0.14

    plt.figure(figsize=(10, 6))
    # Custom Model Bars
    plt.bar(x - 2 * width, prec_values, width, label='Custom Precision', color='blue')
    plt.bar(x - width, rec_values, width, label='Custom Recall', color='green')
    plt.bar(x, f1_values, width, label='Custom F1', color='red')

    # Sklearn Model Bars
    plt.bar(x + width, sklearn_precs, width, label='Sklearn Precision', color='cyan')
    plt.bar(x + 2 * width, sklearn_recs, width, label='Sklearn Recall', color='magenta')
    plt.bar(x + 3 * width, sklearn_f1s, width, label='Sklearn F1', color='yellow')

    plt.xlabel("Categories")
    plt.ylabel("Score")
    plt.title("Test Evaluation Metrics: Custom vs Sklearn AdaBoost")
    plt.xticks(x, categories)
    plt.ylim(0, 1)
    plt.legend(loc="lower right")
    plt.grid(axis='y')

    # Save the plot
    plots_dir = os.path.join('..', 'results', 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    plt.savefig(os.path.join(plots_dir, 'test-2.png'))
    print(f"Plot saved at {os.path.join(plots_dir, 'test-2.png')}")

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
    root = os.path.join("..", "data", "aclImdb")
    texts, labels = load_imdb_data(split='test', root=root)
    print(f"Loaded {len(texts)} test examples.")
    X_test_np = vectorize_texts(texts, vocab)
    y_test_np = np.array(labels)

    # Convert to PyTorch Tensors and Move Data to GPU if available
    X_test = to_tensor(X_test_np)
    y_test = to_tensor(y_test_np)

    # Evaluate
    preds = adaboost_predict(X_test, stumps)
    test_acc = torch.mean((preds == y_test).float())
    # test_acc = np.mean(preds == y_test)
    print(f"Test Accuracy: {test_acc * 100:.2f}%")

    # Compute Evaluation Metrics for Each Class
    prec_pos, rec_pos, f1_pos = compute_class_metrics_torch(y_test, preds, 1)
    prec_neg, rec_neg, f1_neg = compute_class_metrics_torch(y_test, preds, -1)

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
    micro_prec = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    micro_rec = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    micro_f1 = 2 * micro_prec * micro_rec / (micro_prec + micro_rec) if (micro_prec + micro_rec) > 0 else 0.0
    
    # Compute Macro-Averaged Metrics
    macro_prec = (prec_pos + prec_neg) / 2
    macro_rec = (rec_pos + rec_neg) / 2
    macro_f1 = (f1_pos + f1_neg) / 2
    
    # Print the Evaluation Results (A)
    # print("\nEvaluation Metrics on Test Data:")
    # print("\nMicro-averaged: Precision: {:.4f}, Recall: {:.4f}, F1: {:.4f}".format(micro_prec, micro_rec, micro_f1))
    # print("Macro-averaged: Precision: {:.4f}, Recall: {:.4f}, F1: {:.4f}".format(macro_prec, macro_rec, macro_f1))

    # Load the trained Sklearn AdaBoost model
    sklearn_model_path = os.path.join(results_dir, 'sklearn_adaboost.pkl')
    with open(sklearn_model_path, 'rb') as f:
        sklearn_model = pickle.load(f)
    print(f"\nLoaded trained Sklearn AdaBoost model from {sklearn_model_path}")

    # Move tensors to CPU before converting to NumPy
    if isinstance(X_test_np, torch.Tensor):
        X_test_np = X_test_np.cpu().numpy()
    if isinstance(y_test_np, torch.Tensor):
        y_test_np = y_test_np.cpu().numpy()

    # Evaluate the Sklearn AdaBoost model
    sklearn_model.fit(X_test_np, y_test_np)
    sklearn_test_preds = sklearn_model.predict(X_test_np)
    sklearn_acc = np.mean(sklearn_test_preds == y_test_np)
    print(f"Sklearn AdaBoost Test Accuracy: {sklearn_acc * 100:.2f}%")
    sklearn_prec_pos, sklearn_rec_pos, sklearn_f1_pos = compute_class_metrics_sklearn(y_test_np, sklearn_test_preds, 1)
    sklearn_prec_neg, sklearn_rec_neg, sklearn_f1_neg = compute_class_metrics_sklearn(y_test_np, sklearn_test_preds, -1)

    # Print Detailed Evaluation Tables (B)
    print("\nCustom AdaBoost Test Evaluation Metrics:")
    print("{:<10} {:<10} {:<10} {:<10}".format("Category", "Precision", "Recall", "F1"))
    print("{:<10} {:<10.4f} {:<10.4f} {:<10.4f}".format("Positive", prec_pos, rec_pos, f1_pos))
    print("{:<10} {:<10.4f} {:<10.4f} {:<10.4f}".format("Negative", prec_neg, rec_neg, f1_neg))

    print("\nSklearn AdaBoost Test Evaluation Metrics:")
    print("{:<10} {:<10} {:<10} {:<10}".format("Category", "Precision", "Recall", "F1"))
    print("{:<10} {:<10.4f} {:<10.4f} {:<10.4f}".format("Positive", sklearn_prec_pos, sklearn_rec_pos, sklearn_f1_pos))
    print("{:<10} {:<10.4f} {:<10.4f} {:<10.4f}".format("Negative", sklearn_prec_neg, sklearn_rec_neg, sklearn_f1_neg))

    # Compute and Print Micro and Macro Sklearn Averages
    sklearn_macro_prec = (sklearn_prec_pos + sklearn_prec_neg) / 2
    sklearn_macro_rec = (sklearn_rec_pos + sklearn_rec_neg) / 2
    sklearn_macro_f1 = (sklearn_f1_pos + sklearn_f1_neg) / 2

    print("\nCustom AdaBoost Micro-averaged: Precision: {:.4f}, Recall: {:.4f}, F1: {:.4f}".format(micro_prec, micro_rec, micro_f1))
    print("Custom AdaBoost Macro-averaged: Precision: {:.4f}, Recall: {:.4f}, F1: {:.4f}".format(macro_prec, macro_rec, macro_f1))

    print("\nSklearn AdaBoost Micro-averaged: Precision: {:.4f}, Recall: {:.4f}, F1: {:.4f}".format(sklearn_acc, sklearn_acc, sklearn_acc))
    print("Sklearn AdaBoost Macro-averaged: Precision: {:.4f}, Recall: {:.4f}, F1: {:.4f}".format(sklearn_macro_prec, sklearn_macro_rec, sklearn_macro_f1))

    # Plot the Evaluation Results
    categories = ["Positive", "Negative"]
    prec_values = [prec_pos, prec_neg]
    rec_values = [rec_pos, rec_neg]
    f1_values = [f1_pos, f1_neg]
    sklearn_precs = [sklearn_prec_pos, sklearn_prec_neg]
    sklearn_recs = [sklearn_rec_pos, sklearn_rec_neg]
    sklearn_f1s = [sklearn_f1_pos, sklearn_f1_neg]

    # Plot the results
    # plot_test_results_A(categories, prec_values, rec_values, f1_values)
    plot_test_results_B(categories, prec_values, rec_values, f1_values, sklearn_precs, sklearn_recs, sklearn_f1s)


if __name__ == "__main__":
    main()
