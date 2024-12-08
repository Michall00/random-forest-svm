import numpy as np
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.metrics import (
    make_scorer,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
)
from random_forest_svm.hybrid_random_forest import HybridRandomForest
from typing import Dict, Union, Type, Optional


def evaluate_classifier(
    classifier_class: Type,
    n_splits: int,
    classifier_params: dict,
    X_id3: np.ndarray,
    y_id3: np.ndarray,
    X_svm: Optional[np.ndarray] = None,
    y_svm: Optional[np.ndarray] = None,
) -> Dict[str, Union[float, np.ndarray]]:

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    metrics = {
        "accuracy": make_scorer(accuracy_score),
        "f1": make_scorer(f1_score, average="weighted"),
        "precision": make_scorer(precision_score, average="weighted"),
        "recall": make_scorer(recall_score, average="weighted"),
    }
    results = {metric: [] for metric in metrics}

    y_true_all = []
    y_pred_all = []

    for train_idx, test_idx in skf.split(X_id3, y_id3):
        if classifier_class == HybridRandomForest:
            cls = classifier_class(**classifier_params)
            cls.fit(X_svm[train_idx], y_svm[train_idx], X_id3[train_idx], y_id3[train_idx])
            y_pred = cls.predict(X_svm[test_idx], X_id3[test_idx])
        else:
            cls = classifier_class(**classifier_params)
            cls.fit(X_id3[train_idx], y_id3[train_idx])
            y_pred = cls.predict(X_id3[test_idx])

        y_true = y_svm[test_idx]
        y_true_all.extend(y_true)
        y_pred_all.extend(y_pred)
        for metric in metrics:
            results[metric].append(metrics[metric]._score_func(y_true, y_pred))

    for metric in results:
        results[metric] = np.mean(results[metric])

    results["confusion_matrix"] = confusion_matrix(y_true_all, y_pred_all)
    return results
