import numpy as np
from sklearn.model_selection import StratifiedKFold
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
from functools import wraps
import mlflow
import seaborn as sns
import matplotlib.pyplot as plt


def create_cf_heatmap(confusion_matrix: np.ndarray) -> None:
    plt.figure(figsize=(10, 7))
    sns.heatmap(confusion_matrix, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    heatmap_path = "reports/figures/confusion_matrix.png"
    plt.savefig(heatmap_path)
    plt.close()


def mlflow_logger(func):
    @wraps(func)
    def wrapper(*args, **kwargs) -> Dict[str, Union[float, np.ndarray]]:
        enable_mlflow = kwargs.get("enable_mlflow", False)
        if not enable_mlflow:
            return func(*args, **kwargs)

        experiment_name = kwargs.get("experiment_name", "default")
        dataset_name = kwargs.get("dataset_name", "unknown")
        classifier_class = kwargs.get("classifier_class", None)

        mlflow.set_experiment(experiment_name)
        with mlflow.start_run():
            mlflow.log_param("dataset_name", dataset_name)
            mlflow.log_param("classifier_class", classifier_class.__name__)
            results = func(*args, **kwargs)
            for key, value in results.items():
                if key == "confusion_matrix":
                    create_cf_heatmap(value)
                    mlflow.log_artifact("reports/figures/confusion_matrix.png")
                else:
                    mlflow.log_metric(key, value)
            return results

    return wrapper


@mlflow_logger
def evaluate_classifier(
    classifier_class: Type,
    n_splits: int,
    classifier_params: dict,
    X_id3: np.ndarray,
    y_id3: np.ndarray,
    X_svm: Optional[np.ndarray] = None,
    y_svm: Optional[np.ndarray] = None,
    enable_mlflow: bool = False,
    experiment_name: str = "default",
    dataset_name: str = "unknown",
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
