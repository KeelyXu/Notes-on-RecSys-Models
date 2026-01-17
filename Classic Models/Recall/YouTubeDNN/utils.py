import pandas as pd
import re
from collections import defaultdict
import torch
from Datasets.MovieLens.Dataset import RecallDataset


def build_vocab_from_titles(titles: list[str]) -> dict[str, int]:
    """
    Build vocabulary from movie titles using unigram and bigram.

    Args:
        titles: list of movie titles

    Returns:
        vocab: dict mapping token (unigram or bigram) to unique id (1-based, 0 is padding)
    """
    vocab = defaultdict(int)

    for title in titles:
        if pd.isna(title):
            continue
        # Convert to lowercase and remove special characters, keep only alphanumeric and spaces
        title = re.sub(r'[^a-zA-Z0-9\s]', ' ', str(title).lower())
        words = title.split()

        # Add unigrams
        for word in words:
            if word:
                vocab[word] += 1

        # Add bigrams
        for i in range(len(words) - 1):
            bigram = f"{words[i]}_{words[i + 1]}"
            vocab[bigram] += 1

    # Create mapping: token -> id (1-based, 0 reserved for padding)
    vocab_dict = {token: idx + 1 for idx, token in enumerate(sorted(vocab.keys()))}

    return vocab_dict


def title_to_tokens(title: str, vocab: dict[str, int], max_tokens: int = None) -> list[int]:
    """
    Convert a movie title to a list of token IDs (unigram and bigram).

    Args:
        title: movie title string
        vocab: vocabulary dict mapping token to id
        max_tokens: maximum number of tokens to return (None for all)

    Returns:
        list of token IDs
    """
    if pd.isna(title):
        return []

    tokens = []
    # Convert to lowercase and remove special characters
    title = re.sub(r'[^a-zA-Z0-9\s]', ' ', str(title).lower())
    words = title.split()

    # Add unigrams
    for word in words:
        if word and word in vocab:
            tokens.append(vocab[word])

    # Add bigrams
    for i in range(len(words) - 1):
        bigram = f"{words[i]}_{words[i + 1]}"
        if bigram in vocab:
            tokens.append(vocab[bigram])

    if max_tokens is not None:
        tokens = tokens[:max_tokens]

    return tokens


def titles_to_tokens_batch(titles: pd.Series, vocab: dict[str, int]) -> pd.Series:
    """
    Batch convert movie titles to token ID lists using vectorized operations.
    This is much faster than iterating row by row.

    Args:
        titles: pandas Series of movie titles
        vocab: vocabulary dict mapping token to id

    Returns:
        pandas Series of token ID lists
    """
    # Pre-compile regex for better performance
    pattern = re.compile(r'[^a-zA-Z0-9\s]')

    # Vectorized string operations (much faster than per-row processing)
    titles_clean = titles.fillna('').astype(str).str.lower()
    titles_clean = titles_clean.str.replace(pattern, ' ', regex=True)

    # Split into words (vectorized)
    words_series = titles_clean.str.split()

    # Process each title - using apply but on already split words
    def process_title(words):
        if not words or len(words) == 0:
            return []
        tokens = []
        # Add unigrams
        for word in words:
            if word and word in vocab:
                tokens.append(vocab[word])
        # Add bigrams
        for i in range(len(words) - 1):
            bigram = f"{words[i]}_{words[i + 1]}"
            if bigram in vocab:
                tokens.append(vocab[bigram])
        return tokens

    # Apply on words_series
    return words_series.apply(process_title)


class RecallDataset_for_YouTubeDNN(RecallDataset):

    def __init__(self,
                 user_histories: dict,
                 search_histories: dict,  # dict: user_id -> list of token_id_lists
                 samples: list[dict],
                 user_profiles: dict,
                 max_item_id: int,
                 max_seq_len: int,
                 max_search_len: int,
                 timestamps: dict,  # dict: user_id -> list of timestamps
                 train_time_anchor: float,  # timestamp anchor for training (max timestamp in training set)
                 example_age_min: float = None,  # min example_age for normalization
                 example_age_max: float = None,  # max example_age for normalization
                 num_negatives: int = 0,
                 serving_mode: bool = False,
                 seed: int = 42):
        super().__init__(user_histories, samples, user_profiles,
                         max_item_id, max_seq_len, num_negatives, serving_mode, seed)
        self.search_histories = search_histories
        self.max_search_len = max_search_len
        self.timestamps = timestamps
        self.train_time_anchor = train_time_anchor
        self.example_age_min = example_age_min
        self.example_age_max = example_age_max

    def __getitem__(self, idx):
        sample = super().__getitem__(idx)
        user_id = self.samples[idx]["user_id"]
        interaction_idx = self.samples[idx]["idx"]

        # Get search history tokens (flatten all tokens before this interaction)
        search_token_lists = self.search_histories[user_id][:interaction_idx]
        search_tokens = []
        for token_list in search_token_lists:
            search_tokens.extend(token_list)
        search_tokens, _ = self.pad_sequence(search_tokens, self.max_search_len)

        # Get example age
        if self.serving_mode:
            # In serving mode, example age is 0 (current time)
            # This should be normalized to 0 (corresponding to the newest sample in training)
            example_age = 0.0
        else:
            # In training mode, example age = train_time_anchor - sample_timestamp
            sample_timestamp = self.timestamps[user_id][interaction_idx]
            example_age = (self.train_time_anchor - sample_timestamp) / (3600 * 24)  # Convert to days

        # Normalize example_age to [0, 1] using min-max normalization
        if self.example_age_min is not None and self.example_age_max is not None:
            if self.example_age_max > self.example_age_min:
                # Clamp example_age to [example_age_min, example_age_max] to ensure normalization in [0, 1]
                example_age_clamped = max(self.example_age_min, min(example_age, self.example_age_max))
                example_age_norm = (example_age_clamped - self.example_age_min) / (self.example_age_max - self.example_age_min)
            else:
                example_age_norm = 0.0
        else:
            example_age_norm = example_age  # No normalization if stats not provided
        
        # example_age_sq is the square of normalized example_age, which is already in [0, 1]
        # So example_age_sq will also be in [0, 1]
        example_age_sq = example_age_norm ** 2
        
        # Add example age and its square to user features (normalized to [0, 1])
        user_features = sample["user_features"].copy()
        user_features["example_age"] = torch.tensor(example_age_norm)
        user_features["example_age_sq"] = torch.tensor(example_age_sq)

        sample["search_history"] = torch.tensor(search_tokens, dtype=torch.long)
        sample["user_features"] = user_features

        return sample
