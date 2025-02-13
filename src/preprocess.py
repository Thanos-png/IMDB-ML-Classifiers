import os
import re
import math
import numpy as np
import torch
from collections import Counter
from torchtext.datasets import IMDB
from typing import Dict, Tuple


def tokenize(text) -> list[str]:
    """Tokenizes a text string into lowercase word tokens using a simple regex."""

    return re.findall(r'\b\w+\b', text.lower())


def load_imdb_data(split='train', root='../data') -> Tuple[list[str], list[int]]:
    """Loads the IMDB dataset using TorchText."""

    data_path: str = os.path.join(root, split)
    texts: list[str] = []  # A list of the movie reviews
    labels: list[int] = []  # A list of the corresponding labels: +1 for positive and -1 for negative

    for sentiment in ['pos', 'neg']:  # Read both positive and negative reviews
        label = 1 if sentiment == 'pos' else -1
        sentiment_path: str = os.path.join(data_path, sentiment)

        for filename in os.listdir(sentiment_path):
            file_path: str = os.path.join(sentiment_path, filename)
            with open(file_path, "r", encoding="utf-8", errors="ignore") as file:
                texts.append(file.read().strip())
                labels.append(label)

    return texts, labels


def build_vocabulary(texts: list[str], labels: list[int], n_most=20, k_rarest=20, m=2000) -> Dict[str, int]:
    """
    Hyperparameters:
      n_most: Number of most frequent words to omit.
      k_rarest: Number of rarest words to omit.
      m: Number of vocabulary words to keep (based on highest information gain).
    """
    doc_freq = Counter()
    docs_tokens: list[set] = []
    for text in texts:
        tokens = tokenize(text)
        unique_tokens = set(tokens)
        docs_tokens.append(unique_tokens)
        doc_freq.update(unique_tokens)

    total_docs: int = len(texts)

    # Remove the n most frequent and k rarest words.
    most_common: set = set([word for word, freq in doc_freq.most_common(n_most)])
    rarest: set = set([word for word, freq in sorted(doc_freq.items(), key=lambda x: x[1])[:k_rarest]])
    candidates: set = set(doc_freq.keys()) - most_common - rarest

    # Compute overall entropy H(Y)
    pos_count: int = sum(1 for label in labels if label == 1)
    neg_count: int = sum(1 for label in labels if label == -1)

    def entropy(p: int, n: int) -> float:
        tot: int = p + n
        if (tot == 0):
            return 0
        p_ratio: float = p / tot
        n_ratio: float = n / tot
        ent = 0
        if (p_ratio > 0):
            ent -= p_ratio * math.log2(p_ratio)
        if (n_ratio > 0):
            ent -= n_ratio * math.log2(n_ratio)
        return ent
    H_Y: float = entropy(pos_count, neg_count)

    info_gain_scores: Dict[str, float] = {}
    for word in candidates:
        present_pos = 0
        present_neg = 0
        for tokens, label in zip(docs_tokens, labels):
            if (word in tokens):
                if (label == 1):
                    present_pos += 1
                else:
                    present_neg += 1

        present_total: int = present_pos + present_neg
        absent_pos: int = pos_count - present_pos
        absent_neg: int = neg_count - present_neg
        absent_total: int = absent_pos + absent_neg

        H_present: float = entropy(present_pos, present_neg)
        H_absent: float = entropy(absent_pos, absent_neg)

        # Information Gain: reduction in entropy when splitting on word presence.
        IG: float = H_Y - ((present_total / total_docs) * H_present + (absent_total / total_docs) * H_absent)
        info_gain_scores[word] = IG

    # Select the m words with the highest information gain.
    sorted_words: list[tuple[str, float]] = sorted(info_gain_scores.items(), key=lambda x: x[1], reverse=True)
    selected_words: list[str] = [word for word, score in sorted_words[:m]]

    # Dictionary mapping selected words to their feature index.
    vocab: Dict[str, int] = {word: i for i, word in enumerate(selected_words)}
    return vocab


def vectorize_texts(texts: list[str], vocab: Dict[str, int]) -> torch.Tensor:
    """
    Converts a list of texts into a binary feature matrix.
    For each vocabulary word, the corresponding entry is 1 if the text contains the word,
    and 0 otherwise.
    """
    # `X` is a binary feature matrix of shape (n_texts, len(vocab))
    X: torch.Tensor = torch.zeros((len(texts), len(vocab)), dtype=torch.float32, device="cuda" if torch.cuda.is_available() else "cpu")

    # Loop over each review
    for i, text in enumerate(texts):
        # Tokenize the review
        tokens: set = set(tokenize(text))

        for token in tokens:
            # If the word is in vocabulary set corresponding feature to 1
            if (token in vocab):
                X[i, vocab[token]] = 1
    return X
