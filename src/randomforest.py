import numpy as np
import math
import torch
from typing import Optional
from statistics import mode
from id3 import ID3

def randomforest_train(X: np.ndarray, y: np.ndarray, n_estimators: int=100, m:int = None) -> list[ID3]:
    """
    Trains an Random Forest classifier for n_estimators trees.
    Every tree is made using ID3 algorithm
    """
    # `X` is a binary feature matrix of shape (n_texts, len(vocab))
    n_samples, n_features = X.shape
    forest: list[ID3] = []  # A list of trained decision trees created with ID3 algorithm

    if m is None:
        m = int(math.sqrt(total_features))  # Use sqrt of total features if not specified
    
    for i in range(n_estimators):
        # Bagging
        inds = np.random.choice(n_samples,size=n_samples,replace=True)
        X_sample, y_sample = X[inds], y[inds]
        
        # random subset of m features
        sample_features = np.random.choice(n_features, m, replace=False)
        
        #ID3 tree training
        tree = ID3(sample_features)
        tree.fit(X_sample,y_sample)
        forest.append(tree)
              
    return forest


def randomforest_predict(X: np.ndarray, forest: list[ID3]) -> np.ndarray:
    """It makes predictions on X by combining the votes of a trained Random Forest model"""
    
    preds_list = []
    for tree in forest:
        pred = tree.predict(X)
        preds_list.append(pred)
    preds = np.array(preds_list)
    majority_preds_list = []
    for pred in preds.T:    # Transpose, pred is for each sample, not tree 
        majority_preds_list.append(mode(pred))  # mode funvtion returns the most common value
    majority_preds = np.array(preds)
    
    return majority_preds