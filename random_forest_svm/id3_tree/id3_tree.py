import numpy as np
from typing import Any, Union
from random_forest_svm.utils.model_utils import information_gain
from collections import defaultdict


class ID3Tree:
    def __init__(self):
        """Initialize the ID3 decision tree."""
        self.tree = {}

    def fit(self, X: np.ndarray, y: np.ndarray, features: list[int]) -> None:
        """Fit the ID3 decision tree to the data."""
        self.tree = self._id3(X, y, features)

    def _id3(self, X: np.ndarray, y: np.ndarray, features: list[int]) -> Union[dict[int, Any], Any]:
        """Recursively build the ID3 decision tree."""
        unique_classes, counts = np.unique(y, return_counts=True)
        if len(unique_classes) == 1:
            return unique_classes[0]
        if len(features) == 0:
            return unique_classes[np.argmax(counts)]

        gains = [information_gain(X, y, feature) for feature in features]
        best_feature = features[np.argmax(gains)]

        tree = defaultdict(dict)
        remaining_features = [f for f in features if f != best_feature]

        for value in np.unique(X[:, best_feature]):
            subtree = self._id3(X[X[:, best_feature] == value], y[X[:, best_feature] == value], remaining_features)
            tree[best_feature][value] = subtree

        return dict(tree)

    def predict(self, X: np.ndarray) -> list[Any]:
        """Predict class labels for the given feature matrix."""
        return [self._predict_single(x, self.tree) for x in X]

    def _predict_single(self, x: np.ndarray, tree: Union[dict[int, Any], Any]) -> Any:
        """Predict the class label for a single example."""
        while isinstance(tree, dict):
            feature = next(iter(tree))
            value = x[feature]
            tree = tree.get(feature, {}).get(value, tree.get('default'))
        return tree


if __name__ == "__main__":
    import pandas as pd
    data = pd.DataFrame({
        'Outlook': ['Sunny', 'Sunny', 'Overcast', 'Rain', 'Rain', 'Rain', 'Overcast', 'Sunny', 'Sunny', 'Rain', 'Sunny', 'Overcast', 'Overcast', 'Rain'],
        'Temperature': ['Hot', 'Hot', 'Hot', 'Mild', 'Cool', 'Cool', 'Cool', 'Mild', 'Cool', 'Mild', 'Mild', 'Mild', 'Hot', 'Mild'],
        'Humidity': ['High', 'High', 'High', 'High', 'Normal', 'Normal', 'Normal', 'High', 'Normal', 'Normal', 'Normal', 'High', 'Normal', 'High'],
        'Wind': ['Weak', 'Strong', 'Weak', 'Weak', 'Weak', 'Strong', 'Strong', 'Weak', 'Weak', 'Weak', 'Strong', 'Strong', 'Weak', 'Strong'],
        'PlayTennis': ['No', 'No', 'Yes', 'Yes', 'Yes', 'No', 'Yes', 'No', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'No']
    })

    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values
    features = list(range(X.shape[1]))

    id3 = ID3Tree()
    id3.fit(X, y, features)
    print(id3.tree)