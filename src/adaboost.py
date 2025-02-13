import numpy as np
import math
import torch
from typing import Optional


class DecisionStump:
    """
    A decision stump that classifies based on one binary feature.
    For binary features (0/1), we use a fixed threshold (0.5). The prediction is:
    if x[feature_index] >= threshold: predict = polarity
    else: predict = -polarity
    """
    def __init__(self, feature_index: Optional[int] = None, polarity=1, threshold=0.5, alpha=0.0) -> None:
        self.feature_index = feature_index
        self.polarity = polarity
        self.threshold = threshold
        self.alpha = alpha

    def predict(self, X: np.ndarray) -> np.ndarray:
        n_samples: int = X.shape[0]
        feature_values: np.ndarray = X[:, self.feature_index]
        predictions = torch.ones(n_samples, device="cuda" if torch.cuda.is_available() else "cpu")
        predictions[feature_values < self.threshold] = -1
        return self.polarity * predictions


def adaboost_train(X: np.ndarray, y: np.ndarray, T: int, verbose=True) -> list[DecisionStump]:
    """
    Trains an AdaBoost classifier for T iterations.
    At each iteration the algorithm selects (over all features and both polarities)
    the decision stump with the lowest weighted error.
    """
    # `X` is a binary feature matrix of shape (n_texts, len(vocab))
    n_samples, n_features = X.shape
    weights: torch.Tensor = torch.full((n_samples,), 1 / n_samples, device="cuda" if torch.cuda.is_available() else "cpu")
    stumps: list[DecisionStump] = []  # A list of trained decision stumps

    for t in range(T):
        best_stump: Optional[DecisionStump] = None
        best_error: float = float('inf')

        # Try every feature and both polarities.
        for feature in range(n_features):
            for polarity in [1, -1]:
                stump = DecisionStump(feature_index=feature, polarity=polarity, threshold=0.5)
                predictions = stump.predict(X)
                # `y` is a np.ndarray with the labels (+1 or -1)
                error: torch.Tensor = torch.sum(weights[predictions != y])

                # Flip polarity if error > 0.5 (equivalently, use the complement)
                if (error > 0.5):
                    error = 1 - error
                    stump.polarity = -polarity

                if (error < best_error):
                    best_error = error
                    best_stump = stump

        # Compute the stump’s weight (alpha)
        alpha = 0.5 * torch.log((1 - best_error) / (best_error + 1e-10))
        best_stump.alpha = alpha
        # if (best_error == 0):
        #     alpha: float = 1e10  # assign a large value if perfect classification
        # else:
        #     alpha: float = 0.5 * math.log((1 - best_error) / (best_error + 1e-10))
        # best_stump.alpha = alpha

        # Update sample weights
        predictions = best_stump.predict(X)
        weights = weights * torch.exp(-alpha * y * predictions)
        weights /= torch.sum(weights)

        stumps.append(best_stump)
        if verbose:
            print(f"Iteration {t+1}/{T}: feature {best_stump.feature_index}, polarity {best_stump.polarity}, "
                  f"error {best_error:.4f}, alpha {alpha:.4f}")

    return stumps


def adaboost_predict(X: np.ndarray, stumps: list[DecisionStump]) -> torch.Tensor:
    """Makes predictions on X by combining the weighted votes of the decision stumps."""

    # `X` is a binary feature matrix of shape (n_texts, len(vocab))
    n_samples: int = X.shape[0]
    agg_predictions: torch.Tensor = torch.zeros(X.shape[0], device="cuda" if torch.cuda.is_available() else "cpu")
    for stump in stumps:
        agg_predictions += stump.alpha * stump.predict(X)

    # Returns a torch.Tensor with predicted labels (+1 or -1)
    return torch.sign(agg_predictions)
