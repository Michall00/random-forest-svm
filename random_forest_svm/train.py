from random_forest_svm.data.load_data import load_iris, load_wine_quality
from random_forest_svm.utils.training_utils import evaluate_classifier
from random_forest_svm.hybrid_random_forest import HybridRandomForest


def main():
    X_svm, y_svm, X_id3, y_id3 = load_iris()

    svm_params = {"C": 1.0, "kernel": "rbf", "gamma": "scale"}
    id3_params = {"max_depth": None, "min_samples_split": 2, "min_samples_leaf": 1}
    cls_params = {
        "n_classifiers": 10,
        "svm_params": svm_params,
        "id3_params": id3_params,
        "proportion_svm": 0.5,
        "subsample": 0.5,
    }

    results = evaluate_classifier(
        X_svm=X_svm,
        y_svm=y_svm,
        X_id3=X_id3,
        y_id3=y_id3,
        n_splits=5,
        classifier_params=cls_params,
        classifier_class=HybridRandomForest,
    )
    print(results)


if __name__ == "__main__":
    main()
