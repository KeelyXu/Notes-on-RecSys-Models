from abc import ABC, abstractmethod
import random
import torch
from torch.utils.data import Dataset


class BaseRecallDataset(Dataset, ABC):
    """
    Base class for recall / matching datasets.

    Each sample should conceptually correspond to:
        (user features,
         user behavior sequence,
         target item,
         label)
    """

    def __init__(
        self,
        max_seq_len: int,
        num_negatives: int = 0,
        seed: int = 42,
    ):
        """
        Args:
            max_seq_len: maximum length of user behavior sequence
            num_negatives: number of negative samples per positive
            seed: random seed for reproducibility
        """
        self.max_seq_len = max_seq_len
        self.num_negatives = num_negatives
        self.rng = random.Random(seed)

        # subclasses should populate these
        # sample = {
        #     "user_id": u,
        #     "item_id": i,
        #     "idx": interaction_idx,   # unique position index in case a user interacts with the same item for multiple times
        # }
        self.samples = []  # indexable training units

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """
        Returns a training instance.
        Must be implemented in terms of abstract methods below.
        """
        sample = self.samples[idx]

        user_id = sample["user_id"]
        pos_item_id = sample["item_id"]

        user_features = self.get_user_features(user_id)
        behavior_seq = self.get_user_sequence(user_id, pos_item_id)
        behavior_seq, seq_len = self.pad_sequence(behavior_seq)

        items = [pos_item_id]
        labels = [1]

        if self.num_negatives > 0:
            neg_items = self.sample_negatives(user_id, pos_item_id)
            items.extend(neg_items)
            labels.extend([0] * len(neg_items))

        return {
            "user_features": self.to_tensor(user_features),
            "behavior_seq": torch.tensor(behavior_seq, dtype=torch.long),
            "seq_len": torch.tensor(seq_len, dtype=torch.long),
            "item_ids": torch.tensor(items, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.float),
        }

    # ------------------------------------------------------------------
    # Abstract methods (dataset-specific)
    # ------------------------------------------------------------------

    @abstractmethod
    def get_user_features(self, user_id: int) -> dict:
        """
        Return static user features (can be empty dict).
        """
        raise NotImplementedError

    @abstractmethod
    def get_user_sequence(
        self, user_id: int, idx: int
    ) -> list[int]:
        """
        Return user behavior sequence before the target interaction.
        """
        raise NotImplementedError

    @abstractmethod
    def sample_negatives(
        self, user_id: int, pos_item_id: int
    ) -> list[int]:
        """
        Sample negative item ids.
        """
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Shared utility methods
    # ------------------------------------------------------------------

    def pad_sequence(self, seq: list[int]):
        """
        Right-pad sequence with 0, truncate from the left if too long.
        """
        seq = seq[-self.max_seq_len :]
        seq_len = len(seq)

        if seq_len < self.max_seq_len:
            seq = seq + [0] * (self.max_seq_len - seq_len)

        return seq, seq_len

    def to_tensor(self, features: dict) -> dict[str, torch.Tensor]:
        """
        Convert feature dict to tensor dict.
        """
        tensor_feats = {}
        for k, v in features.items():
            if isinstance(v, torch.Tensor):
                tensor_feats[k] = v
            else:
                tensor_feats[k] = torch.tensor(v)
        return tensor_feats
