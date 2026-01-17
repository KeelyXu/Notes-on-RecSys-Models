import pandas as pd
import numpy as np
import torch
from Utils.io import find_project_root
from Datasets.MovieLens.Dataset import RecallDataset
from sklearn.preprocessing import LabelEncoder

PROJECT_ROOT = find_project_root()

def read_all_data(split_genre: bool=True, verbose: bool=False) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    data_dir = PROJECT_ROOT / 'Data' / 'MovieLens_1M'

    if verbose:
        print(f"Start reading data from {data_dir}...")

    users = pd.read_csv(
        data_dir / 'users.dat',
        sep="::",
        engine="python",
        names=["user_id", "gender", "age", "occupation", "zip_code"],
        encoding="latin-1"
    )

    movies = pd.read_csv(
        data_dir / 'movies.dat',
        sep="::",
        engine="python",
        names=["movie_id", "title", "genres"],
        encoding="latin-1"
    )

    if split_genre:
        genre_dummies = movies["genres"].str.get_dummies(sep="|")
        movies = pd.concat(
            [movies.drop(columns=["genres"]), genre_dummies],
            axis=1
        )

    ratings = pd.read_csv(
        data_dir / 'ratings.dat',
        sep="::",
        engine="python",
        names=["user_id", "movie_id", "rating", "timestamp"],
        encoding="latin-1"
    )

    return users, movies, ratings

def preprocess(
        users: pd.DataFrame,
        movies: pd.DataFrame,
        ratings: pd.DataFrame
):
    print(f"Start preprocessing...")

    # Encode UserIDs and MovieIDs (1-based mapping, 0 is reserved for padding)
    user_le = LabelEncoder()
    # Fit on all possible IDs
    users['user_id'] = user_le.fit_transform(users['user_id']) + 1
    ratings['user_id'] = user_le.transform(ratings['user_id']) + 1

    item_le = LabelEncoder()
    movies['movie_id'] = item_le.fit_transform(movies['movie_id']) + 1
    ratings['movie_id'] = item_le.transform(ratings['movie_id']) + 1

    users = users.sort_values(by=['user_id'])
    movies = movies.sort_values(by=['movie_id'])

    num_items = ratings['movie_id'].max()
    print(f"Num Users: {len(user_le.classes_)}, Num Items: {num_items}")

    users['gender'] = users['gender'].map({'F': 0, 'M': 1})
    users['age'] = LabelEncoder().fit_transform(users['age'])

    # Drop useless columns
    movies = movies.drop(columns=['title'])
    users = users.drop(columns=['zip_code'])    # zip code is of high cardinality which means it's sparse and contains few information

    return users, movies, ratings

def convert_item_feats_into_tensor(movies: pd.DataFrame) -> (torch.Tensor, list[str]):
    feat_cols = movies.columns.tolist()
    feat_cols.remove('movie_id')    # movie_id should not be included in features
    item_pool_numpy = movies[feat_cols].values.astype(int)
    # Do not forget to padding the first row of the tensor
    padding_row = np.zeros((1, item_pool_numpy.shape[1]), dtype=int)
    item_pool_numpy = np.concatenate([padding_row, item_pool_numpy], axis=0)

    item_pool_tensor = torch.LongTensor(item_pool_numpy)
    return item_pool_tensor, feat_cols

def make_samples(all_histories: dict[str, list[int]], ratios: list[float]|None = None, min_len: int=5):
    train_samples = []
    val_samples = []
    test_samples = []

    # Normalize rations
    if ratios:
        sum_ratios = sum(ratios)
        ratios = [ratio / sum_ratios for ratio in ratios]

    print(f"Splitting data (Mode: {'Leave-One-Out' if ratios is None else 'Ratio ' + str(ratios)})...")
    for user_id, history in all_histories.items():
        L = len(history)
        if L < min_len:
            continue

        if ratios is None:
            num_test = 1
            num_val = 1
        else:
            num_test = max(1, int(L * ratios[2]))
            num_val = max(1, int(L * ratios[1]))

        # make sure every train sample has history of at least one item
        if L - num_test - num_val < 2:
            continue

        # history: [i_1, i_2, ..., i_T-2, i_T-1, i_T]
        # idx is the index of the TARGET item in the history list

        # Test samples: [L - num_test, L)
        for i in range(L - num_test, L):
            test_samples.append({
                "user_id": user_id,
                "item_id": history[i],
                "idx": i
            })

        # Validation Samples: [L - num_test - num_val, L - num_test)
        for i in range(L - num_test - num_val, L - num_test):
            val_samples.append({
                "user_id": user_id,
                "item_id": history[i],
                "idx": i
            })

        # Train Samples: [1, L - num_test - num_val)
        for i in range(1, L - num_test - num_val):
            train_samples.append({
                "user_id": user_id,
                "item_id": history[i],
                "idx": i
            })

    print(f"Train samples: {len(train_samples)}")
    print(f"Val samples:   {len(val_samples)}")
    print(f"Test samples:  {len(test_samples)}")

    return train_samples, val_samples, test_samples

def make_datasets(
        users: pd.DataFrame,
        ratings: pd.DataFrame,
        max_seq_len: int = 50,
        num_negatives: int = 0,
        min_len: int = 5,
        ratios: list[float]|None = None,
        seed: int = 42
) -> tuple[RecallDataset, RecallDataset, RecallDataset]:
    """
    Loads MovieLens 1M data and creates Train, Validation, and Test datasets

    Args:
        max_seq_len: Maximum sequence length for padding.
        num_negatives: Number of negative samples per positive sample.
        min_len: Users with fewer interactions than this will be filtered out.
        ratios: Splitting ratio of [train, val, test], e.g. [0.8, 0.1, 0.1]. If None, split by Leave-One-Out.
        seed: Random seed.

    Returns:
        train_dataset, val_dataset, test_dataset
    """
    # Build User Profiles (dict: user_id -> features)
    user_profiles = users.set_index('user_id').to_dict(orient='index')

    # Construct User Histories and Split
    # Sort by User and Timestamp
    ratings = ratings.sort_values(by=['user_id', 'timestamp'])

    # Group interactions by user
    # Result: user_id -> list of movie_ids
    all_histories = ratings.groupby('user_id')['movie_id'].apply(list).to_dict()

    train_samples, val_samples, test_samples = make_samples(all_histories, ratios, min_len)

    # Instantiate Datasets
    # Note: We pass the SAME final_histories to all datasets.
    # The 'idx' in the sample determines how much of the history is seen.

    train_dataset = RecallDataset(
        user_histories=all_histories,
        samples=train_samples,
        user_profiles=user_profiles,
        max_item_id=ratings['movie_id'].max(),
        max_seq_len=max_seq_len,
        num_negatives=num_negatives,
        seed=seed
    )

    val_dataset = RecallDataset(
        user_histories=all_histories,
        samples=val_samples,
        user_profiles=user_profiles,
        max_item_id=ratings['movie_id'].max(),
        max_seq_len=max_seq_len,
        num_negatives=num_negatives,  # Usually 100 for metric calculation, but keeping consistent
        seed=seed
    )

    test_dataset = RecallDataset(
        user_histories=all_histories,
        samples=test_samples,
        user_profiles=user_profiles,
        max_item_id=ratings['movie_id'].max(),
        max_seq_len=max_seq_len,
        num_negatives=num_negatives,
        serving_mode=True,
        seed=seed
    )

    return train_dataset, val_dataset, test_dataset


if __name__ == "__main__":
    # Pipeline
    users, movies, ratings = read_all_data(verbose=True)
    users, movies, ratings = preprocess(users, movies, ratings)
    item_feats_tensor, feat_names = convert_item_feats_into_tensor(movies)
    train_set, val_set, test_set = make_datasets(users, ratings, ratios=[0.8, 0.0, 0.2])
