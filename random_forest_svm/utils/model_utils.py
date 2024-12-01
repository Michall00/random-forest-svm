import numpy as np


def entropy(y: np.ndarray) -> float:
    """Calculate the entropy of a dataset."""
    unique_calsses, counts = np.unique(y, return_counts=True)
    probabilities = counts / len(y)
    return -np.sum(probabilities * np.log2(probabilities))


def information_gain(X: np.ndarray, y: np.ndarray, feature: int) -> float:
    """Calculate the information gain of a feature."""
    total_entropy = entropy(y)
    values, counts = np.unique(X[:, feature], return_counts=True)
    weighted_entropy = np.sum((counts[i] / np.sum(counts)) * entropy(y[X[:, feature] == values[i]]) for i in range(len(values)))
    return total_entropy - weighted_entropy
