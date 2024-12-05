import numpy as np
from typing import Any, Union, Optional
from random_forest_svm.utils.id3_utils import find_best_split, information_gain
from collections import defaultdict


class ID3:
    def __init__(self,
                 max_depth: int = None,
                 min_samples_split: int = 2,
                 min_samples_leaf: int = 1):
        """Initialize the ID3 decision tree with hyperparameters."""
        self.tree = {}
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf

    def fit(self,
            X: np.ndarray,
            y: np.ndarray) -> None:
        """Fit the ID3 decision tree to the data."""
        self._all_features = list(range(X.shape[1]))
        self.tree = self._id3(X, y, depth=0)

    def _id3(self,
             X: np.ndarray,
             y: np.ndarray,
             depth: int,
             features: Optional[list[int]] = None,
             ) -> Union[dict[str, Any], Any]:
        """Recursively build the ID3 decision tree."""
        if features is None:
            features = self._all_features
        unique_classes, counts = np.unique(y, return_counts=True)
        if len(unique_classes) == 1:
            return unique_classes[0]
        if len(features) == 0 or (self.max_depth is not None and depth >= self.max_depth):
            return unique_classes[np.argmax(counts)]
        if len(y) < self.min_samples_split or np.min(counts) < self.min_samples_leaf:
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

        if best_split is not None:
            left_mask = X[:, best_feature] <= best_split
            right_mask = X[:, best_feature] > best_split

            tree[best_feature][f'<= {best_split}'] = self._id3(X[left_mask], y[left_mask], depth + 1, remaining_features)
            tree[best_feature][f'> {best_split}'] = self._id3(X[right_mask], y[right_mask], depth + 1, remaining_features)
        else:
            for value in np.unique(X[:, best_feature]):
                subtree = self._id3(X[X[:, best_feature] == value], y[X[:, best_feature] == value], depth + 1, remaining_features)
                tree[best_feature][value] = subtree

        return dict(tree)

    def predict(self,
                X: np.ndarray) -> list[Any]:
        """Predict class labels for the given feature matrix."""
        return [self._predict_single(x, self.tree) for x in X]

    def _predict_single(self,
                        x: np.ndarray,
                        tree: Union[dict[str, Any], Any]) -> Any:
        """Predict the class label for a single example."""
        while isinstance(tree, dict):
            feature = next(iter(tree))
            subtree = tree[feature]

            if any(op in key for key in subtree.keys() for op in ['<=', '>']):
                split_key = next(iter(subtree))
                split_value = float(split_key.split()[1])
                if x[self._all_features[feature]] <= split_value:
                    tree = subtree[f'<= {split_value}']
                else:
                    tree = subtree[f'> {split_value}']
            else:
                value = x[self._all_features[feature]]
                tree = subtree.get(value, subtree.get('default'))

        return tree


if __name__ == "__main__":
    data = np.array([
        ['Overcast', 'Hot', 'High', 'Weak', 'No'],
        ['Sunny', 'Hot', 'High', 'Strong', 'No'],
        ['Overcast', 'Hot', 'High', 'Weak', 'Yes'],
        ['Rain', 'Mild', 'High', 'Weak', 'Yes'],
        ['Rain', 'Cool', 'Normal', 'Weak', 'Yes'],
        ['Rain', 'Cool', 'Normal', 'Strong', 'No'],
        ['Overcast', 'Cool', 'Normal', 'Strong', 'Yes'],
        ['Overcast', 'Mild', 'High', 'Weak', 'No'],
        ['Overcast', 'Cool', 'Normal', 'Weak', 'Yes'],
        ['Rain', 'Mild', 'Normal', 'Weak', 'Yes'],
        ['Overcast', 'Mild', 'Normal', 'Strong', 'Yes'],
        ['Overcast', 'Mild', 'High', 'Strong', 'Yes'],
        ['Overcast', 'Hot', 'Normal', 'Weak', 'Yes'],
        ['Rain', 'Mild', 'High', 'Strong', 'No']
    ])

    feature_names = ['Outlook', 'Temperature', 'Humidity', 'Wind']
    X = data[:, :-1]
    y = data[:, -1]
    features = list(range(X.shape[1]))

    id3 = ID3(max_depth=4, min_samples_split=2, min_samples_leaf=2)
    id3.fit(X, y)
    print("Decision Tree for discrete values:")
    print(id3.tree)

    data_continuous = np.array([
        [2.5, 1.1, 'A'],
        [3.6, 2.2, 'B'],
        [1.2, 3.3, 'A'],
        [4.8, 4.4, 'B'],
        [3.3, 5.5, 'A'],
        [2.1, 1.2, 'A'],
        [5.0, 2.3, 'B'],
        [1.8, 3.4, 'A'],
        [3.7, 4.5, 'B'],
        [2.9, 5.6, 'A']
    ])

    feature_names_continuous = ['Feature1', 'Feature2']
    X_continuous = data_continuous[:, :-1].astype(float)
    y_continuous = data_continuous[:, -1]
    features_continuous = list(range(X_continuous.shape[1]))

    id3_continuous = ID3()
    id3_continuous.fit(X_continuous, y_continuous)
    print("\nDecision Tree for continuous values:")
    print(id3_continuous.tree)

    predictions_continuous = id3_continuous.predict(X_continuous)
    print("Predictions for continuous data:")
    print(predictions_continuous)
