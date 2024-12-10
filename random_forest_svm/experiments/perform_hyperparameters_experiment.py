import optuna
from optuna.trial import Trial
from random_forest_svm.utils.training_utils import evaluate_classifier, load_dataset
from random_forest_svm.hybrid_random_forest import HybridRandomForest
from functools import partial


def objective(trial: Trial,
              dataset_name: str,
              n_splits: int,
              metric: str):
    X_svm, y_svm, X_id3, y_id3 = load_dataset(dataset_name)

    C = trial.suggest_float("C", 0.1, 100, log=True)
    gamma = trial.suggest_float("gamma", 0.0001, 10, log=True)
    max_depth = trial.suggest_int("max_depth", 0, 20)
    min_samples_split = trial.suggest_int("min_samples_split", 2, 20)
    min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 20)
    n_classifiers = trial.suggest_int("n_classifiers", 10, 200)
    proportion_svm = trial.suggest_float("proportion_svm", 0, 1)
    subsample = trial.suggest_float("subsample", 0, 1)

    max_depth = None if max_depth == 0 else max_depth
    kernel = "rbf"

    svm_params = {
        "C": C,
        "kernel": kernel,
        "gamma": gamma
    }

    id3_params = {
        "max_depth": max_depth,
        "min_samples_split": min_samples_split,
        "min_samples_leaf": min_samples_leaf
    }

    classifier_params = {
        "n_classifiers": n_classifiers,
        "svm_params": svm_params,
        "id3_params": id3_params,
        "proportion_svm": proportion_svm,
        "subsample": subsample
    }

    results = evaluate_classifier(
        classifier_class=HybridRandomForest,
        n_splits=n_splits,
        classifier_params=classifier_params,
        X_svm=X_svm,
        y_svm=y_svm,
        X_id3=X_id3,
        y_id3=y_id3,
        enable_mlflow=True,
        experiment_name="Hyperparameters Experiment",
        dataset_name=dataset_name,
    )

    return results[metric]


def main():
    partial_objective = partial(objective, dataset_name="WineQuality", n_splits=5, metric="f1")
    study = optuna.create_study(
        study_name="hyperparameters_experiment_wine_quality",
        direction="maximize",
        load_if_exists=True,
        storage="sqlite:///hyperparameters.db",)
    study.optimize(partial_objective, n_trials=1)


if __name__ == "__main__":
    main()
