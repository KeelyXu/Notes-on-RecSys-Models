from tqdm import tqdm
import torch.nn as nn
from Utils.metrics import recall_at_k, hit_rate_at_k
from Datasets.MovieLens.utils import *
from utils import *
from model import YouTubeDNN


def train(model: YouTubeDNN, train_set, lr, n_epochs):
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
            search_history = batch["search_history"].to(device)
            
            # Separate categorical and simple features
            cat_feats = {}
            simple_feats_list = []
            for feat_name, feat in batch["user_features"].items():  # shape of feat: [batch_size]
                feat = feat.to(device)
                if feat_name in ["example_age", "example_age_sq", "gender"]:
                    feat = feat.unsqueeze(1)  # [batch_size, 1] for concat later
                    simple_feats_list.append(feat)
                else:
                    cat_feats[feat_name] = feat
            
            if simple_feats_list:
                simple_feats = torch.cat(simple_feats_list, dim=1)  # [batch_size, num_simple_feats]
            else:
                simple_feats = torch.empty(history.shape[0], 0, device=device)
            
            # Get target item (positive item)
            target_item = batch["item_ids"].to(device)
            
            optimizer.zero_grad()
            user_embeds = model(history, search_history, cat_feats, simple_feats)
            logits, labels = model.sampled_softmax(user_embeds, target_item, history)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            batch_size = history.shape[0]
            total_loss += loss.item() * batch_size
            total_sample += batch_size

        print(f"Epoch {epoch} | Loss {total_loss / total_sample:.3f}")


def test(model: YouTubeDNN, test_set, item_pool_size, top_k=20):
    device = model.device
    model = model.to(device)

    model.eval()
    all_target_items = []
    all_recall_items = []
    
    with torch.no_grad():
        # Get all item embeddings
        item_embeds = model.itemEmbeds(torch.arange(item_pool_size, device=device))  # [num_items, d_item]
        
        for batch in tqdm(test_set, desc="Testing"):
            history = batch["behavior_seq"].to(device)
            search_history = batch["search_history"].to(device)
            
            # Separate categorical and simple features
            cat_feats = {}
            simple_feats_list = []
            for feat_name, feat in batch["user_features"].items():
                feat = feat.to(device)
                if feat_name in ["example_age", "example_age_sq", "gender"]:
                    feat = feat.unsqueeze(1)  # [batch_size, 1] for concat later
                    simple_feats_list.append(feat)
                else:
                    cat_feats[feat_name] = feat
            
            if simple_feats_list:
                simple_feats = torch.cat(simple_feats_list, dim=1)
            else:
                simple_feats = torch.empty(history.shape[0], 0, device=device)
            
            target_items = batch["item_ids"]  # list of lists
            
            # Get user embeddings
            user_embeds = model(history, search_history, cat_feats, simple_feats)  # [batch_size, d_item]
            
            # Calculate logits: [batch_size, num_items]
            logits = torch.matmul(user_embeds, item_embeds.T)
            logits[:, 0] = -1e9  # mask padding item
            
            # Get top_k items
            _, top_indices = torch.topk(logits, k=top_k, dim=1)  # [batch_size, top_k]
            recall_items = top_indices.cpu().numpy().tolist()
            
            all_target_items.extend(target_items)
            all_recall_items.extend(recall_items)

    print(f"recall@{top_k}: {recall_at_k(all_target_items, all_recall_items, top_k)}, "
          f"hitRate@{top_k}: {hit_rate_at_k(all_target_items, all_recall_items, top_k)}")


if __name__ == "__main__":
    # configuration
    d_item = 256
    d_token = 256
    num_neg = 10
    max_len = 30
    max_search_len = 50  # max number of search tokens
    train_batch_size = 512
    test_batch_size = 32
    n_epochs = 100
    lr = 1e-3
    dense_layer_sizes = [512, 256]  # sizes of dense layers

    # prepare data
    print("Loading data...")
    users, movies, ratings = read_all_data(verbose=True, split_genre=False)  # Keep title for search history
    
    # Build vocabulary from all movie titles
    print("Building vocabulary from movie titles...")
    all_titles = movies['title'].tolist()
    vocab = build_vocab_from_titles(all_titles)
    vocab_size = len(vocab) + 1  # +1 for padding token (id=0)
    print(f"Vocabulary size: {vocab_size}")
    
    # Add title to ratings before preprocessing (so we can use it after preprocessing)
    ratings = ratings.merge(movies[['movie_id', 'title']], on='movie_id', how='left')
    
    # Preprocess data (this will remap IDs, but title column in ratings is preserved)
    users, movies, ratings = preprocess(users, movies, ratings)
    
    # Build search histories: user_id -> list of token_id_lists (one list per interaction)
    print("Building search histories...")
    ratings_sorted = ratings.sort_values(by=['user_id', 'timestamp']).copy()
    
    # Batch process titles to tokens (much faster than iterrows)
    print("  Converting titles to tokens (batch processing)...")
    token_lists = titles_to_tokens_batch(ratings_sorted['title'], vocab)
    ratings_sorted['tokens'] = token_lists
    
    # Group by user_id and aggregate tokens and timestamps
    print("  Grouping by user_id...")
    search_histories = ratings_sorted.groupby('user_id')['tokens'].apply(list).to_dict()
    timestamps_dict = ratings_sorted.groupby('user_id')['timestamp'].apply(list).to_dict()
    all_histories = ratings_sorted.groupby('user_id')['movie_id'].apply(list).to_dict()

    # Create samples
    train_samples, _, test_samples = make_samples(all_histories, ratios=[0.8, 0.0, 0.2])
    
    # Find max timestamp in training samples
    train_time_anchor = 0.0
    for sample in train_samples:
        user_id = sample["user_id"]
        idx = sample["idx"]
        if idx < len(timestamps_dict[user_id]):
            train_time_anchor = max(train_time_anchor, timestamps_dict[user_id][idx])
    
    # Calculate example_age statistics on training set for normalization
    print("Calculating example_age statistics for normalization...")
    example_ages = []
    for sample in train_samples:
        user_id = sample["user_id"]
        idx = sample["idx"]
        if idx < len(timestamps_dict[user_id]):
            sample_timestamp = timestamps_dict[user_id][idx]
            example_age = (train_time_anchor - sample_timestamp) / (3600 * 24)  # Convert to days
            example_ages.append(example_age)
    
    example_age_min = min(example_ages) if example_ages else 0.0
    example_age_max = max(example_ages) if example_ages else 1.0
    print(f"  example_age range: [{example_age_min:.2f}, {example_age_max:.2f}] days")
    
    # Build user profiles
    user_profiles = users.set_index('user_id').to_dict(orient='index')
    
    # Create datasets
    print("Creating datasets...")
    max_item_id = ratings['movie_id'].max()
    
    train_set = RecallDataset_for_YouTubeDNN(
        user_histories=all_histories,
        search_histories=search_histories,
        samples=train_samples,
        user_profiles=user_profiles,
        max_item_id=max_item_id,
        max_seq_len=max_len,
        max_search_len=max_search_len,
        timestamps=timestamps_dict,
        train_time_anchor=train_time_anchor,
        example_age_min=example_age_min,
        example_age_max=example_age_max,
        num_negatives=0,  # negative sampling is done in sampled_softmax
        serving_mode=False,
        seed=42
    )
    
    test_set = RecallDataset_for_YouTubeDNN(
        user_histories=all_histories,
        search_histories=search_histories,
        samples=test_samples,
        user_profiles=user_profiles,
        max_item_id=max_item_id,
        max_seq_len=max_len,
        max_search_len=max_search_len,
        timestamps=timestamps_dict,
        train_time_anchor=train_time_anchor,
        example_age_min=example_age_min,
        example_age_max=example_age_max,
        num_negatives=0,
        serving_mode=True,
        seed=42
    )
    
    train_loader = train_set.to_dataloader(batch_size=train_batch_size, shuffle=True)
    test_loader = test_set.to_dataloader(batch_size=test_batch_size, shuffle=False)
    
    # Calculate user profile dimension
    user_profile_dim = 0
    user_profile_embed_dim = {}
    for feat_name in ['age', 'occupation']:
        n_cat = users[feat_name].nunique()
        embed_dim = d_item  # Use same dimension as item embedding
        user_profile_embed_dim[feat_name] = (n_cat, embed_dim)
        user_profile_dim += embed_dim
    
    # Add dimensions for search tokens, item vectors, and simple features (example_age, example_age_sq)
    user_profile_dim = d_token + d_item + user_profile_dim + 3  # +3 for example_age, example_age_sq and gender
    
    # Training
    print("Start training...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    item_pool_size = movies['movie_id'].max() + 1
    
    model = YouTubeDNN(
        d_item=d_item,
        d_token=d_token,
        user_feats_embed_dim=user_profile_embed_dim,
        user_profile_dim=user_profile_dim,
        item_pool_size=item_pool_size,
        max_len=max_len,
        max_search_len=max_search_len,
        vocab_size=vocab_size,
        num_neg=num_neg,
        dense_layer_sizes=dense_layer_sizes,
        device=device
    )
    
    train(model, train_loader, lr, n_epochs)
    
    # Testing
    print("Start testing...")
    test(model, test_loader, item_pool_size, top_k=20)
