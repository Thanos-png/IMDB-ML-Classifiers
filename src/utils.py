import torch
from sklearn.metrics import precision_score, recall_score, f1_score


def to_tensor(data):
    """Converts NumPy arrays to PyTorch tensors and moves to GPU."""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if isinstance(data, torch.Tensor):  # If already tensor, just move to GPU
        return data.to(device)
    return torch.tensor(data, dtype=torch.float32, device=device)


def compute_metrics_for_class_torch(y_true, y_pred, target=1):
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


def compute_metrics_for_class_sklearn(y_true, y_pred, target=1):
    """Computes precision, recall, and F1 for a given target class using scikit-learn."""

    precision = precision_score(y_true, y_pred, pos_label=target, zero_division=0)
    recall = recall_score(y_true, y_pred, pos_label=target, zero_division=0)
    f1 = f1_score(y_true, y_pred, pos_label=target, zero_division=0)
    return precision, recall, f1
