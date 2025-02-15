import numpy as np
import math
import torch
from typing import Optional
from scipy.stats import mode
from sklearn.tree import DecisionTreeClassifier
from utils import to_tensor


def randomforest_train(X: np.ndarray, y: np.ndarray, n_estimators: int=100, m_features:int = None,max_depth:int=None) -> list[tuple[DecisionTreeClassifier, np.ndarray]]:
    """Trains a Random Forest classifier for n_estimators trees."""

    X_tensor = to_tensor(X)
    y_tensor = to_tensor(y)
    
    n_samples, n_features = X_tensor.shape
    forest = []  # Stores (tree, selected features) pairs

    if m_features is None:
        m_features = int(math.sqrt(n_features))

    for _ in range(n_estimators):
        # Bagging
        inds = np.random.choice(n_samples, size=n_samples, replace=True)
        X_sample, y_sample = X_tensor[inds], y_tensor[inds]

        # Select random subset of m_features
        sample_features = np.random.choice(np.arange(n_features), m_features, replace=False)

        # DecisionTree train
        tree = DecisionTreeClassifier(criterion="entropy", max_depth=max_depth, max_features=None)
        X_sample_np = X_sample.cpu().numpy()
        tree.fit(X_sample_np[:, sample_features], y_sample.cpu().numpy())

        forest.append((tree, sample_features))

    return forest



def randomforest_predict(X: np.ndarray, forest: list[tuple[DecisionTreeClassifier, np.ndarray]]) -> np.ndarray:
    """It makes predictions on X by combining the votes of a trained Random Forest model."""

    X_tensor = to_tensor(X)
    X_np = X_tensor.cpu().numpy()

    preds_list = []
    for tree, sample_features in forest:
        pred = tree.predict(X_np[:, sample_features])  # Use selected features
        preds_list.append(pred)

    preds = np.array(preds_list)

    # Majority voting
    majority_preds = mode(preds, axis=0)[0].flatten()

    return to_tensor(majority_preds)
