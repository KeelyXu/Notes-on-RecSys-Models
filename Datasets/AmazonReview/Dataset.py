from Datasets.BaseDatasets import BaseRecallDataset


class RecallDataset(BaseRecallDataset):

    def __init__(self,
                 user_histories: dict,    # dict: u -> [i1, i2, ...] (time-ordered)
                 samples: list[dict],     # list of {"user_id", "item_id", "idx"}
                 num_items: int,
                 max_seq_len: int,
                 num_negatives: int = 0,
                 seed: int = 42):
        super().__init__(max_seq_len, num_negatives, seed)
        self.user_histories = user_histories
        self.samples = samples
        self.num_items = num_items

    def get_user_features(self, user_id: int) -> dict:
        return {'user_id': user_id}   # Amazon Review did not provide user profile, only user ID is available

    def get_user_sequence(self, user_id: int, idx: int) -> list[int]:
        return self.user_histories[user_id][:idx]

    def sample_negatives(self, user_id, pos_item_id):
        negs = set()
        clicked = {i for _, i in self.user_histories[user_id]}

        while len(negs) < self.num_negatives:
            neg = self.rng.randint(1, self.num_items)
            if neg != pos_item_id and neg not in clicked:
                negs.add(neg)

        return list(negs)
