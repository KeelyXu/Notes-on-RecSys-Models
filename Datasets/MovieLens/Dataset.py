from Datasets.BaseDatasets import BaseRecallDataset


class RecallDataset(BaseRecallDataset):

    # PS: We ignore ratings and only consider the interactions

    def __init__(self,
                 user_histories: dict,    # dict: u -> [i1, i2, ...] (time-ordered)
                 samples: list[dict],     # list of {"user_id", "item_id", "idx"}
                 user_profiles: dict,     # dict: u -> dict of features
                 max_item_id: int,
                 max_seq_len: int,
                 num_negatives: int = 0,
                 serving_mode: bool = False,
                 seed: int = 42):
        super().__init__(max_seq_len, num_negatives, serving_mode, seed)
        self.user_histories = user_histories
        self.user_profiles = user_profiles
        self.samples = samples
        self.max_item_id = max_item_id

    def get_user_features(self, user_id: int) -> dict:
        return self.user_profiles[user_id]

    def get_user_sequence(self, user_id: int, idx: int) -> list[int]:
        return self.user_histories[user_id][:idx]

    def sample_negatives(self, user_id, pos_item_id):
        negs = set()
        clicked = {i for _, i in self.user_histories[user_id]}

        while len(negs) < self.num_negatives:
            neg = self.rng.randint(1, self.max_item_id)
            if neg != pos_item_id and neg not in clicked:
                negs.add(neg)

        return list(negs)

    def get_serving_targets_of_user(self, user_id: int, idx: int) -> list[int]:
        return self.user_histories[user_id][idx:]
