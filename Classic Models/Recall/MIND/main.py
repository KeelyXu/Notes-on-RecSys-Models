import random
from tqdm import tqdm
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from Utils.metrics import recall_at_k, hit_rate_at_k
from Datasets.AmazonReview.utils import read_csv_in_dir
from Datasets.AmazonReview.Dataset import RecallDataset
from model import MIND


def train(model, train_set):
    device = model.device
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model.train()
    for epoch in range(1, n_epochs + 1):
        total_loss = 0.0
        for batch in tqdm(train_set):
            history = batch["behavior_seq"].to(device)
            user_profile = {feat_name: feat.to(device) for feat_name, feat in batch["user_features"].items()}
            item_ids = batch["item_ids"].to(device)

            optimizer.zero_grad()
            caps = model(history, user_profile)
            logits, labels = model.sampled_softmax(caps, item_ids, history)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch} | Loss {total_loss:.3f}")

def test(model, test_set, top_k=30):
    device = model.device
    model = model.to(device)

    model.eval()
    with torch.no_grad():
        item_embeds = model.itemEmbeds.weight  # [num_items, d]
        # user-wise recall(0~1) and hit (0 or 1)
        recalls, hit_rates = [], []
        for batch in tqdm(test_set):
            history = batch["behavior_seq"].to(device)
            user_profile = {feat_name: feat.to(device) for feat_name, feat in batch["user_features"].items()}
            user_ids = batch["user_features"]["user_id"].detach().long().numpy()
            target_items = [multi_targets[user_id] for user_id in user_ids]

            caps = model(history, user_profile)     # [batch_size, K, d]
            logits = torch.matmul(caps, item_embeds.T)          # [batch_size, K, num_items]
            max_logits = logits.max(dim=1).values.detach().numpy()     # [batch_size, num_items]
            # get top_k with the highest logits
            recall_items = np.argpartition(max_logits, kth=num_items - top_k, axis=1)[:, -top_k:]  # [batch_size, top_k]
            recall_items = recall_items.tolist()

    print(f"recall@{top_k}: {recall_at_k(target_items, recall_items, top_k)}, "
          f"hitRate@{top_k}: {hit_rate_at_k(target_items, recall_items, top_k)}")


if __name__ == "__main__":
    # configuration
    d = 8
    K = 3
    iter_num = 3
    num_neg = 10
    min_item_freq = 10
    min_user_freq = 3
    dataset = "All Beauty"
    train_user_frac = 0.8
    max_seq_len = 20
    train_batch_size = 512
    test_batch_size = 16
    n_epochs = 100
    lr = 1e-3

    # prepare data
    df = read_csv_in_dir(Path(dataset))

    # filter items and users with low frequency
    df = df.groupby('itemId').filter(lambda x: len(x) >= min_item_freq)
    df = df.groupby('userId').filter(lambda x: len(x) >= min_user_freq)

    # assign integer index to each item and user
    # note: item idx start from 1 because 0 represents padding here
    id_user = list(enumerate(df['userId'].unique()))
    id_item = list(enumerate(df['itemId'].unique()))
    id2user = {idx: user for idx, user in id_user}
    user2id = {user: idx for idx, user in id_user}
    id2item = {idx + 1: item for idx, item in id_item}
    item2id = {item: idx + 1 for idx, item in id_item}

    df['userId'] = df['userId'].map(user2id)
    df['itemId'] = df['itemId'].map(item2id)
    df = df.sort_values(by=['timestamp']).reset_index(drop=True)

    # split data by split users into train and test users for simplicity
    train_users, test_users = set(), set()
    for userId in range(len(id_user)):
        if random.random() < train_user_frac:
            train_users.add(userId)
        else:
            test_users.add(userId)

    # get histories for each user
    history = df.groupby('userId')['itemId'].apply(list).to_dict()
    print(f"Avg history length: {df.groupby('userId')['itemId'].size().mean()}")

    train_samples = []
    for userId in train_users:
        user_hist = history[userId]
        # for train set, we use sliding windows (y from idx=1 to idx=len(user_hist) - 1) for each user
        for i in range(1, len(user_hist)):
            train_samples.append({"user_id": userId,
                                  "item_id": user_hist[i],
                                  "idx": i})

    test_samples = []
    multi_targets = dict()
    for userId in test_users:
        # for test set, generate only one sample for each user (when we get 80% history)
        user_hist = history[userId]
        pred_idx = int(0.8 * len(user_hist))
        test_samples.append({"user_id": userId,
                             "item_id": user_hist[pred_idx],
                             "idx": pred_idx})
        multi_targets[userId] = history[userId][pred_idx:]

    # make datasets
    num_items = len(id_item) + 1
    train_set = DataLoader(
        RecallDataset(history, train_samples, num_items, max_seq_len),
        batch_size=train_batch_size,
        shuffle=True
    )
    test_set = DataLoader(
        RecallDataset(history, test_samples, num_items, max_seq_len),
        batch_size=test_batch_size,
        shuffle=True
    )

    print("Start training...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = MIND(d=d, K=K, user_profile_embed_dim={'user_id': (len(id_user), d)},
                 num_items=num_items, max_len=max_seq_len, num_neg=num_neg, iter_num=iter_num, device=device)
    train(model, train_set)
    print("Start testing...")
    test(model, test_set)
