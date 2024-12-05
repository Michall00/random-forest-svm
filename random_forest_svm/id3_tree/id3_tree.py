import numpy as np
from typing import Any, Union
from random_forest_svm.utils.id3_utils import find_best_split, information_gain
from collections import defaultdict


class ID3:
    def __init__(self, feature_names: list[str]) -> None:
        """Initialize the ID3 decision tree."""
        self.tree = {}
        self.feature_names = feature_names

    def fit(self, X: np.ndarray, y: np.ndarray, features: list[int]) -> None:
        """Fit the ID3 decision tree to the data."""
        self.tree = self._id3(X, y, features)

    def _id3(self, X: np.ndarray, y: np.ndarray, features: list[int]) -> Union[dict[str, Any], Any]:
        """Recursively build the ID3 decision tree."""
        unique_classes, counts = np.unique(y, return_counts=True)
        if len(unique_classes) == 1:
            return unique_classes[0]
        if len(features) == 0:
            return unique_classes[np.argmax(counts)]

        best_feature = None
        best_split = None
        best_gain = -np.inf

        for feature in features:
            if np.issubdtype(X[:, feature].dtype, np.number):
                split = find_best_split(X, y, feature)
                if split is not None:
                    gain = information_gain(X, y, feature, split)
                    if gain > best_gain:
                        best_gain = gain
                        best_feature = feature
                        best_split = split
            else:
                gain = information_gain(X, y, feature)
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature

        if best_feature is None:
            return unique_classes[np.argmax(counts)]

        tree = defaultdict(dict)
        remaining_features = [f for f in features if f != best_feature]
        feature_name = self.feature_names[best_feature]

        if best_split is not None:
            left_mask = X[:, best_feature] <= best_split
            right_mask = X[:, best_feature] > best_split

            tree[feature_name]['<='] = self._id3(X[left_mask], y[left_mask], remaining_features)
            tree[feature_name]['>'] = self._id3(X[right_mask], y[right_mask], remaining_features)
        else:
            for value in np.unique(X[:, best_feature]):
                subtree = self._id3(X[X[:, best_feature] == value], y[X[:, best_feature] == value], remaining_features)
                tree[feature_name][value] = subtree

        return dict(tree)

    def predict(self, X: np.ndarray) -> list[Any]:
        """Predict class labels for the given feature matrix."""
        return [self._predict_single(x, self.tree) for x in X]

    def _predict_single(self, x: np.ndarray, tree: Union[dict[int, Any], Any]) -> Any:
        """Predict the class label for a single example."""
        while isinstance(tree, dict):
            feature = next(iter(tree))
            subtree = tree[feature]

            if '<=' in subtree and '>' in subtree:
                split_value = list(subtree.keys())[0]
                if x[feature] <= split_value:
                    tree = subtree['<=']
                else:
                    tree = subtree['>']
            else:
                value = x[feature]
                tree = subtree.get(value, subtree.get('default'))

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

    feature_names = data.columns[:-1].tolist()
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values
    features = list(range(X.shape[1]))

    id3 = ID3(feature_names=feature_names)
    id3.fit(X, y, features)
    print("Decision Tree for dicreate values:")
    print(id3.tree)
