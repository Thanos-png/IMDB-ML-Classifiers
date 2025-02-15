import os
import random
import pickle
import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from preprocess import load_imdb_data, vectorize_texts
from rnnmodel import StackedBiRNN
from utils import to_tensor, compute_metrics_for_class_torch
from train_rnnmodel import numericalize_texts


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
    
    # Use numericalize_texts to convert texts to sequences
    X_test = numericalize_texts(texts, vocab, max_len=500)
    y_test = np.array(labels)

    # Remap labels: -1 -> 0, 1 remains 1.
    y_test = np.where(y_test == -1, 0, 1)

    # Convert to PyTorch Tensors and create a DataLoader for batching
    X_test = to_tensor(X_test).long()
    y_test = to_tensor(y_test).long()
    test_dataset = TensorDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # Evaluate Model in Batches
    all_preds = []
    all_labels = []
    total_correct = 0
    total_samples = 0
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            outputs = model(batch_X)
            max_values, preds = torch.max(outputs, dim=1)
            all_preds.append(preds.cpu().numpy())
            all_labels.append(batch_y.cpu().numpy())
            total_correct += (preds == batch_y).sum().item()
            total_samples += batch_y.size(0)

    test_acc = total_correct / total_samples * 100
    print(f"Test Accuracy: {test_acc:.2f}%")

    #Concatenate predictions and true labels from all batches
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    # Compute per-class metrics
    precision_pos, recall_pos, f1_pos = compute_metrics_for_class_torch(to_tensor(all_labels), to_tensor(all_preds), 1)
    precision_neg, recall_neg, f1_neg = compute_metrics_for_class_torch(to_tensor(all_labels), to_tensor(all_preds), 0)

    # Compute True Positives, False Positives, and False Negatives for Each Class
    tp_pos = np.sum((all_labels == 1) & (all_preds == 1))
    tp_neg = np.sum((all_labels == 0) & (all_preds == 0))
    fp_pos = np.sum((all_labels == 0) & (all_preds == 1))
    fp_neg = np.sum((all_labels == 1) & (all_preds == 0))
    fn_pos = np.sum((all_labels == 1) & (all_preds == 0))
    fn_neg = np.sum((all_labels == 0) & (all_preds == 1))

    # Compute Total Counts
    total_tp = tp_pos + tp_neg
    total_fp = fp_pos + fp_neg
    total_fn = fn_pos + fn_neg

    # Compute Micro-Averaged Metrics
    micro_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    micro_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    micro_f1 = 2 * micro_precision * micro_recall / (micro_precision + micro_recall) if (micro_precision + micro_recall) > 0 else 0.0

    # Compute Macro-Averaged Metrics
    macro_precision = (precision_pos + precision_neg) / 2
    macro_recall = (recall_pos + recall_neg) / 2
    macro_f1 = (f1_pos + f1_neg) / 2

    # Print the Evaluation Results in Table Format
    print("\nEvaluation Metrics on Test Data:")
    print("{:<10} {:<10} {:<10} {:<10}".format("Category", "Precision", "Recall", "F1"))
    print("{:<10} {:<10.4f} {:<10.4f} {:<10.4f}".format("Positive", precision_pos, recall_pos, f1_pos))
    print("{:<10} {:<10.4f} {:<10.4f} {:<10.4f}".format("Negative", precision_neg, recall_neg, f1_neg))
    print("\nMicro-averaged: Precision: {:.4f}, Recall: {:.4f}, F1: {:.4f}".format(micro_precision, micro_recall, micro_f1))
    print("Macro-averaged: Precision: {:.4f}, Recall: {:.4f}, F1: {:.4f}".format(macro_precision, macro_recall, macro_f1))


if __name__ == "__main__":
    main()
