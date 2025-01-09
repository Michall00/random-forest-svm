# AUTHOR: Mateusz Ostaszewski
from pathlib import Path

from random_forest_svm.data.config import (
    CHURN_RAW_PATH,
    IRIS_RAW_PATH,
    PROCESSED_DATA_DIR,
    WINE_QUALITY_RAW_PATH,
)
from random_forest_svm.data.data_processor import (
    ChurnDataProcessor,
    IrisDataProcessor,
    WineQualityDataProcessor,
)


def process_data():
    iris_processor = IrisDataProcessor(Path(IRIS_RAW_PATH), Path(PROCESSED_DATA_DIR) / "iris")
    wine_quality_processor = WineQualityDataProcessor(
        Path(WINE_QUALITY_RAW_PATH), Path(PROCESSED_DATA_DIR) / "wine_quality"
    )
    churn_processor = ChurnDataProcessor(Path(CHURN_RAW_PATH), Path(PROCESSED_DATA_DIR) / "churn")
    processors = [iris_processor, wine_quality_processor, churn_processor]

    for processor in processors:
        processor.process_data()
        processor.save_data()


def main():
    process_data()


if __name__ == "__main__":
    main()
