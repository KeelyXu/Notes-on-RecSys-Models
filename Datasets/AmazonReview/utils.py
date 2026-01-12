from pathlib import Path
import pandas as pd
import json
import gzip
from Utils.io import find_project_root

PROJECT_ROOT = find_project_root()

def _check_path(data_dir: Path) -> Path:
    data_dir = PROJECT_ROOT / 'Data' / 'AmazonReviewData(2018)' / data_dir
    if not data_dir.exists():
        raise FileNotFoundError(f"{data_dir} does not exist")
    return data_dir

def read_json_in_dir(data_dir: Path):
    data_dir = _check_path(data_dir)
    json_file = next(data_dir.glob("*.json.gz"))
    g = gzip.open(json_file, 'rb')
    for l in g:
        yield json.loads(l)

def read_csv_in_dir(data_dir: Path) -> pd.DataFrame:
    data_dir = _check_path(data_dir)
    csv_file = next(data_dir.glob("*.csv"))
    csv_data = pd.read_csv(csv_file, header=None, names=['itemId', 'userId', 'rating', 'timestamp'])
    return csv_data


if __name__ == "__main__":
    csv_data = read_csv_in_dir(Path("All Beauty"))
    json_data = read_json_in_dir(Path("All Beauty"))
    item_data = [data for data in json_data]
    print("DATA SAMPLES")
    print(csv_data.head(), '\n')
    print("SUMMARY")
    print(f"  Number of interactions: {len(csv_data)}")
    print(f"  Number of items: {len(item_data)}")
    print(f"  Number of users: {csv_data['userId'].nunique()}")
    print(f"  Items appearing: {csv_data['itemId'].nunique()}")
