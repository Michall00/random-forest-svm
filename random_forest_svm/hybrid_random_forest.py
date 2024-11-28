import numpy as np
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier as ID3
from scipy.stats import mode


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

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        n_samples, _ = X.shape
        n_svm = int(self.n_classifiers * self.proportion_svm)
        n_selected_samples = int(n_samples * self.subsample)

        for cls_idx in range(self.n_classifiers):
            sample_idxs = np.random.choice(n_samples, n_selected_samples, replace=False)
            _X, _y = X[sample_idxs, :], y[sample_idxs]

            cls = (
                SVC(**self.svm_params, probability=True)
                if cls_idx < n_svm
                else ID3(**self.id3_params, criterion="entropy")
            )
            cls.fit(_X, _y)
            self.classifiers.append(cls)

    def predict(self, X: np.ndarray) -> np.ndarray:
        predictions = np.array([clf.predict(X) for clf in self.classifiers])
        return mode(predictions, axis=0).mode[0]
