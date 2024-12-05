import os
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


class DataProcessor(ABC):
    def __init__(self, raw_data_path: Path, processed_data_dir: Path):
        self.data = pd.read_csv(raw_data_path)
        self.processed_data_dir = processed_data_dir

    def process_data(self) -> None:
        self.svm_df = self.data.copy()
        self.id3_df = self.data.copy()
        self.apply_standardization()

    def save_data(self) -> None:
        os.makedirs(self.processed_data_dir, exist_ok=True)
        self.id3_df.to_csv(self.processed_data_dir / "ID3.csv", index=False)
        self.svm_df.to_csv(self.processed_data_dir / "SVM.csv", index=False)

    @abstractmethod
    def apply_standardization(self) -> None: ...


class IrisDataProcessor(DataProcessor):
    def __init__(self, raw_data_path: Path, processed_data_dir: Path):
        super().__init__(raw_data_path, processed_data_dir)

    def process_data(self) -> None:
        self.data["class"] = self.data["class"].apply(lambda x: 1 if x == "Iris-setosa" else 0)
        super().process_data()

    def apply_standardization(self) -> None:
        scaler = StandardScaler()
        self.svm_df[self.svm_df.columns[:-1]] = scaler.fit_transform(
            self.svm_df[self.svm_df.columns[:-1]]
        )


class WineQualityDataProcessor(DataProcessor):
    def __init__(self, raw_data_path: Path, processed_data_dir: Path):
        super().__init__(raw_data_path, processed_data_dir)

    def process_data(self) -> None:
        self.data["quality"] = self.data["quality"].apply(lambda x: 1 if x >= 6 else 0)
        super().process_data()

    def apply_standardization(self) -> None:
        self.svm_df["color"] = self.svm_df["color"].apply(lambda x: 1 if x == "red" else 0)
        self.id3_df["color"] = self.id3_df["color"].apply(lambda x: 1 if x == "red" else 0)
        columns_to_log = [
            "residual_sugar",
            "chlorides",
            "free_sulfur_dioxide",
            "total_sulfur_dioxide",
        ]
        columns_to_standardize = [
            "fixed_acidity",
            "volatile_acidity",
            "citric_acid",
            "density",
            "pH",
            "sulphates",
            "alcohol",
        ]

        for column in columns_to_log:
            self.svm_df[column] = np.log1p(self.svm_df[column])

        scaler = StandardScaler()
        self.svm_df[columns_to_standardize] = scaler.fit_transform(
            self.svm_df[columns_to_standardize]
        )


class ChurnDataProcessor(DataProcessor):
    def __init__(self, raw_data_path: Path, processed_data_dir: Path):
        super().__init__(raw_data_path, processed_data_dir)

    def apply_standardization(self) -> None:
        columns_to_log = [
            "Call  Failure",
            "Seconds of Use",
            "Frequency of use",
            "Frequency of SMS",
            "Customer Value",
            "Charge  Amount",
            "Distinct Called Numbers",
        ]
        columns_to_standardize = ["Subscription  Length", "Age"]

        for column in columns_to_log:
            self.svm_df[column] = self.svm_df[column].apply(lambda x: np.log1p(x))

        scaler = StandardScaler()
        self.svm_df[columns_to_standardize] = scaler.fit_transform(
            self.svm_df[columns_to_standardize]
        )
