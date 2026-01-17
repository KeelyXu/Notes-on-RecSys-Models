from tqdm import tqdm
import torch.nn as nn
from Utils.metrics import recall_at_k, hit_rate_at_k
from Datasets.MovieLens.utils import *
from model import MIND


def train(model: MIND, train_set, lr, n_epochs):
    device = model.device
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model.train()
    for epoch in range(1, n_epochs + 1):
        total_loss = 0.0
        total_sample = 0
        for batch in tqdm(train_set, desc=f"Epoch {epoch}"):
            history = batch["behavior_seq"].to(device)
            user_profile = {feat_name: feat.to(device) for feat_name, feat in batch["user_features"].items()}
            item_ids = batch["item_ids"].to(device)

            optimizer.zero_grad()
            caps, cap_mask = model(history, user_profile)
            logits, labels = model.sampled_softmax(caps, cap_mask, item_ids, history)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            batch_size = history.shape[0]
            total_loss += loss.item() * batch_size
            total_sample += batch_size

        print(f"Epoch {epoch} | Avg Loss {total_loss / total_sample:.3f}")


def test(model: MIND, test_set, item_pool_size, top_k=20):
    device = model.device
    model = model.to(device)

    model.eval()
    all_target_items = []
    all_recall_items = []
    with torch.no_grad():
        item_embeds = model.get_item_embedding(torch.arange(item_pool_size, device=device))  # [num_items, d]
        for batch in tqdm(test_set, desc="Testing"):
            history = batch["behavior_seq"].to(device)
            user_profile = {feat_name: feat.to(device) for feat_name, feat in batch["user_features"].items()}
            target_items = batch["item_ids"]

            caps, cap_mask = model(history, user_profile)     # [batch_size, K, d], [batch_size, K]
            logits = torch.matmul(caps, item_embeds.T)          # [batch_size, K, num_items]
            logits = logits.masked_fill(cap_mask.unsqueeze(2), -1e9)         # [batch_size, K, num_items]
            logits[:, :, 0] = -1e9                                           # mask padding item
            max_logits, _ = logits.max(dim=1)     # [batch_size, num_items]
            # get top_k with the highest logits
            _, top_indices = torch.topk(max_logits, k=top_k, dim=1)     # [batch_size, top_k]
            recall_items = top_indices.cpu().numpy().tolist()
            all_target_items.extend(target_items)
            all_recall_items.extend(recall_items)

    print(f"recall@{top_k}: {recall_at_k(all_target_items, all_recall_items, top_k)}, "
          f"hitRate@{top_k}: {hit_rate_at_k(all_target_items, all_recall_items, top_k)}")


if __name__ == "__main__":
    # configuration
    d = 8
    K = 3
    iter_num = 3
    num_neg = 10
    max_seq_len = 30
    train_batch_size = 512
    test_batch_size = 32
    n_epochs = 100
    lr = 1e-3

    # prepare data
    users, movies, ratings = read_all_data(verbose=True)
    users, movies, ratings = preprocess(users, movies, ratings)
    item_feats_tensor, feat_names = convert_item_feats_into_tensor(movies)
    train_set, _, test_set = make_datasets(users, ratings,
                                           max_seq_len=max_seq_len,
                                           ratios=[0.8, 0.0, 0.2])

    # user id is viewed as a common user feature in MIND, so we add it as a feature in user profile
    user_profiles = train_set.user_profiles
    for uid, profile in user_profiles.items():
        profile['user_id'] = uid

    train_loader = train_set.to_dataloader(batch_size=train_batch_size)
    test_loader = test_set.to_dataloader(batch_size=test_batch_size)

    user_profile_embed_dim = {'user_id': (users['user_id'].max() + 1, d),
                              'gender': (users['gender'].nunique(), d),
                              'age': (users['age'].nunique(), d),
                              'occupation': (users['occupation'].nunique(), d)}
    item_profile_n_cat = {genre: 2 for genre in movies.columns if genre != 'movie_id'}

    # Training
    print("Start training...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    item_pool_size = movies['movie_id'].max() + 1
    model = MIND(d=d, K=K,
                 user_feats_embed_dim=user_profile_embed_dim,
                 item_feats_n_cat=item_profile_n_cat,
                 item_feats=item_feats_tensor,
                 item_feat_names=feat_names,
                 item_pool_size=item_pool_size,
                 max_len=max_seq_len, num_neg=num_neg, iter_num=iter_num, device=device)
    train(model, train_loader, lr, n_epochs)

    # Testing
    print("Start testing...")
    test(model, test_loader, item_pool_size)
