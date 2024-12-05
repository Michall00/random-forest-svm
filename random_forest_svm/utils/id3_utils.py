import numpy as np


def entropy(y: np.ndarray) -> float:
    """Calculate the entropy of a dataset."""
    unique_classes, counts = np.unique(y, return_counts=True)
    probabilities = counts / len(y)
    return -np.sum(probabilities * np.log2(probabilities))


def information_gain(X: np.ndarray, y: np.ndarray, feature: int, split: float = None) -> float:
    """Calculate the information gain of a feature."""
    total_entropy = entropy(y)
    if split is not None:
        left_mask = X[:, feature] <= split
        right_mask = X[:, feature] > split
        left_entropy = entropy(y[left_mask])
        right_entropy = entropy(y[right_mask])
        weighted_entropy = (np.sum(left_mask) / len(y)) * left_entropy + (
            np.sum(right_mask) / len(y)
        ) * right_entropy
    else:
        values, counts = np.unique(X[:, feature], return_counts=True)
        weighted_entropy = np.sum(
            (counts[i] / np.sum(counts)) * entropy(y[X[:, feature] == values[i]])
            for i in range(len(values))
        )
    return total_entropy - weighted_entropy


def find_best_split(X: np.ndarray, y: np.ndarray, feature: int) -> float:
    """Find the best split point for a continuous feature."""
    sorted_indices = np.argsort(X[:, feature])
    X_sorted = X[sorted_indices]

    best_split = None
    best_gain = -np.inf

    for i in range(1, len(X_sorted)):
        if X_sorted[i, feature] != X_sorted[i - 1, feature]:
            split = (X_sorted[i, feature] + X_sorted[i - 1, feature]) / 2
            gain = information_gain(X, y, feature, split)
            if gain > best_gain:
                best_gain = gain
                best_split = split

    return best_split
