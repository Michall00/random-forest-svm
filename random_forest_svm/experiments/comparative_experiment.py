from random_forest_svm.hybrid_random_forest import HybridRandomForest
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from random_forest_svm.data.load_data import load_iris, load_wine_quality, load_churn
from random_forest_svm.utils.training_utils import evaluate_classifier
from itertools import product


def main():
    hybrid_forest_params = {
        "n_classifiers": 10,
        "svm_params": {"C": 1.0, "kernel": "rbf", "gamma": "scale"},
        "id3_params": {"max_depth": None, "min_samples_split": 2, "min_samples_leaf": 1},
        "proportion_svm": 0.5,
        "subsample": 0.5,
    }
    models = [SVC]
    datasets = [load_iris, load_wine_quality, load_churn]

    experiments = list(product(models, datasets))

    for experiment in experiments:
        model, load_data = experiment
        X_svm, y_svm, X_id3, y_id3 = load_data()
        evaluate_classifier(
            classifier_class=model,
            classifier_params=hybrid_forest_params if model == HybridRandomForest else {},
            n_splits=5,
            X_svm=X_svm,
            y_svm=y_svm,
            X_id3=X_id3,
            y_id3=y_id3,
            enable_mlflow=True,
            experiment_name=f"Comparative Experiment",
            dataset_name=load_data.__name__,
        )


if __name__ == "__main__":
    main()
