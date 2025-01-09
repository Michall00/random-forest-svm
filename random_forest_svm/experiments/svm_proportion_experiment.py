# AUTHOR: Michał Sadowski
from random_forest_svm.hybrid_random_forest import HybridRandomForest
from random_forest_svm.data.load_data import load_iris, load_wine_quality, load_churn
from random_forest_svm.utils.training_utils import evaluate_classifier
from itertools import product
import numpy as np


def main():
    svm_proportions = np.arange(0.00, 1.01, 0.01)
    hybrid_forest_params = {
        "n_classifiers": 100,
        "svm_params": {"C": 1.0, "kernel": "rbf", "gamma": "scale"},
        "id3_params": {"max_depth": None, "min_samples_split": 2, "min_samples_leaf": 1},
        "subsample": 0.5,
    }
    models = [HybridRandomForest]
    datasets = [load_wine_quality, load_churn, load_iris]

    experiments = list(product(models, datasets, svm_proportions))

    for experiment in experiments:
        model, load_data, proportion_svm = experiment
        X_svm, y_svm, X_id3, y_id3 = load_data()
        hybrid_forest_params["proportion_svm"] = proportion_svm
        evaluate_classifier(
            classifier_class=model,
            classifier_params=hybrid_forest_params,
            n_splits=5,
            X_svm=X_svm,
            y_svm=y_svm,
            X_id3=X_id3,
            y_id3=y_id3,
            enable_mlflow=True,
            experiment_name="Proportion SVM",
            dataset_name=load_data.__name__,
        )


if __name__ == "__main__":
    main()
