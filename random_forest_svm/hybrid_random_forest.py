# AUTHOR: Mateusz Ostaszewski
import numpy as np
from sklearn.svm import SVC
from random_forest_svm.id3_tree.id3_tree import ID3
from scipy.stats import mode

class MockSVC:
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self.cl = np.unique(y)[0]

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.full((X.shape[0],), self.cl, dtype=int)

class HybridRandomForest:
    def __init__(
        self,
        n_classifiers: int,
        svm_params: dict,
        id3_params: dict,
        proportion_svm: float = 0.5,
        subsample: float = 1,
    ) -> None:
        self.n_classifiers = n_classifiers
        self.proportion_svm = proportion_svm
        self.classifiers: list = []
        self._validate_parameters(subsample)
        self.subsample = subsample
        self.id3_params = id3_params
        self.svm_params = svm_params

    def _validate_parameters(self, subsample: float) -> None:
        if not (0 <= subsample <= 1):
            raise ValueError("Parameter subsample must be in range [0, 1].")

    def fit(
        self, X_svm: np.ndarray, y_svm: np.ndarray, X_id3: np.ndarray, y_id3: np.ndarray
    ) -> None:
        n_samples, _ = X_svm.shape
        n_svm = int(self.n_classifiers * self.proportion_svm)
        n_selected_samples = int(n_samples * self.subsample)

        for cls_idx in range(self.n_classifiers):
            sample_idxs = np.random.choice(n_samples, n_selected_samples, replace=False)
            X, y = (
                (X_svm[sample_idxs, :], y_svm[sample_idxs])
                if cls_idx < n_svm
                else (X_id3[sample_idxs, :], y_id3[sample_idxs])
            )

            if cls_idx < n_svm:
                if len(np.unique(y)) == 1:
                    cls = MockSVC()
                else:
                    cls = SVC(**self.svm_params)
            else:
                cls = ID3(**self.id3_params)

            cls.fit(X, y)
            self.classifiers.append(cls)

    def predict(self, X_svm: np.ndarray, X_id3: np.ndarray) -> np.ndarray:
        n_samples = X_svm.shape[0]
        predictions = np.zeros((self.n_classifiers, n_samples))

        for cls_idx, clf in enumerate(self.classifiers):
            if isinstance(clf, SVC):
                predictions[cls_idx] = clf.predict(X_svm)
            else:
                predictions[cls_idx] = clf.predict(X_id3)

        return mode(predictions, axis=0).mode


if __name__ == "__main__":
    from random_forest_svm.data.load_data import load_churn, load_wine_quality

    X_svm, y_svm, X_id3, y_id3 = load_wine_quality()

    svm_params = {"C": 1.0, "kernel": "rbf", "gamma": "scale"}

    id3_params = {"max_depth": None, "min_samples_split": 2, "min_samples_leaf": 1}

    hybrid_rf = HybridRandomForest(n_classifiers=10, svm_params=svm_params, id3_params=id3_params)
    hybrid_rf.fit(X_svm, y_svm, X_id3, y_id3)

    print(hybrid_rf.predict(X_svm, X_id3))
