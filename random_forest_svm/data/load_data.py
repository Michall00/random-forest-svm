# AUTHOR: Mateusz Ostaszewski
import pandas as pd
import numpy as np


def load_data(
    dataset_name: str, column_to_drop: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load the dataset and return the data in format (X_svm, y_svm, X_id3, y_id3).
    """
    svm_df = pd.read_csv(f"data/processed/{dataset_name}/SVM.csv")
    id3_df = pd.read_csv(f"data/processed/{dataset_name}/ID3.csv")

    X_svm = svm_df.drop(column_to_drop, axis=1)
    y_svm = svm_df[column_to_drop]
    X_id3 = id3_df.drop(column_to_drop, axis=1)
    y_id3 = id3_df[column_to_drop]

    return X_svm.to_numpy(), y_svm.to_numpy(), X_id3.to_numpy(), y_id3.to_numpy()


def load_iris() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    return load_data("iris", "class")


def load_wine_quality() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    return load_data("wine_quality", "quality")


def load_churn() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    return load_data("churn", "Churn")
