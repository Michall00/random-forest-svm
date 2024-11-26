from ucimlrepo import fetch_ucirepo 
from random_forest_svm.data.config import IRIS_ID, WINE_QUALITY_ID, CHURN_ID, IRIS_RAW_PATH, WINE_QUALITY_RAW_PATH, CHURN_RAW_PATH


DATASETS = [
    {"id": IRIS_ID, "path": IRIS_RAW_PATH},
    {"id": WINE_QUALITY_ID, "path": WINE_QUALITY_RAW_PATH},
    {"id": CHURN_ID, "path": CHURN_RAW_PATH},
]

def download_and_save_dataset(dataset_id, file_path):
    dataset = fetch_ucirepo(id=dataset_id)
    data = dataset['data']['original']
    data.to_csv(file_path, index=False)

def main():
    for dataset in DATASETS:
        download_and_save_dataset(dataset["id"], dataset["path"])

if __name__ == '__main__':
    main()
