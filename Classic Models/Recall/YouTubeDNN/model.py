import torch
import torch.nn as nn
from Utils.sample import generate_negative_samples


class YouTubeDNN(nn.Module):

    def __init__(self, d_item: int, d_token: int,
                 user_feats_embed_dim: dict[str, tuple[int, int]],
                 user_profile_dim: int,
                 item_pool_size: int,
                 max_len: int, max_search_len: int, vocab_size: int,
                 num_neg: int,
                 dense_layer_sizes: list[int],
                 device: str='cpu'):
        """
        Args:
            d_item: embed dim of items
            d_token: embed dim of search tokens
            user_feats_embed_dim:  a dict specifying n_cat and embed dim of categorical feature in user profile
            user_profile_dim: dim of user profile after concatenate embedded features and simple features
            item_pool_size: size of item pool (include padding item)
            max_len: max length of user behavior sequence
            max_search_len: max length of token sequence
            vocab_size: size of vocabulary size (unigram and bigram)
            num_neg: number of negative samples per positive
            dense_layer_sizes: list of dense layer sizes
            device: device to use
        """
        super().__init__()
        self.item_pool_size = item_pool_size
        self.max_len = max_len
        self.max_search_item = max_search_len
        self.num_neg = num_neg
        self.device = device
        self.itemEmbeds = nn.Embedding(item_pool_size, d_item, padding_idx=0)
        self.tokenEmbeds = nn.Embedding(vocab_size, d_token, padding_idx=0)

        self.userEmbeds = nn.ModuleDict()
        for feat_name, (n_cat, embed_dim) in user_feats_embed_dim.items():
            self.userEmbeds[feat_name] = nn.Embedding(n_cat, embed_dim)

        # Fully connected layers
        dense_layers = []
        prev_size = user_profile_dim
        for layer_size in dense_layer_sizes:
            dense_layers.append(nn.Linear(prev_size, layer_size))
            dense_layers.append(nn.ReLU())
            prev_size = layer_size
        dense_layers.append(nn.Linear(prev_size, d_item))
        self.dense_layers = nn.Sequential(*dense_layers)

    def forward(self, history, search_history, cat_feats, simple_feats):
        """
        Args:
            history: [batch_size, max_len]
            search_history: [batch_size, max_search_len]
            cat_feats: a dict, each value is of shape [batch_size]
            simple_feats: [batch_size, num_simple_feats]
        Returns:
            tensor: [batch_size, d_item]
        """
        batch_size = history.shape[0]
        # Item embedding & pooling
        item_embeds = self.itemEmbeds(history)  # [batch_size, max_len, d_item]
        mask = (history != 0).float()   # [batch_size, max_len]
        lengths = mask.sum(dim=1, keepdim=True)  # [batch_size, 1]
        item_vectors = (item_embeds * mask.unsqueeze(-1)).sum(dim=1) / lengths    # [batch_size, d_item]
        
        # Search token embedding & pooling
        token_embeds = self.tokenEmbeds(search_history)     # [batch_size, max_search_len, d_token]
        mask = (search_history != 0).float()  # [batch_size, max_search_len]
        lengths = mask.sum(dim=1, keepdim=True)  # [batch_size, 1]
        token_vectors = (token_embeds * mask.unsqueeze(-1)).sum(dim=1) / lengths  # [batch_size, d_token]

        # Other categorical feature embedding
        cat_embeds = torch.empty(batch_size, 0, device=self.device)
        for cat_feat_name, cat_feat in cat_feats.items():
            cat_embeds = torch.concat([cat_embeds, self.userEmbeds[cat_feat_name](cat_feat)], dim=1)

        # Concatenate user features
        user_embeds = torch.concat([token_vectors, item_vectors, cat_embeds, simple_feats], dim=1)   # [batch_size, d_token + d_item + d_other_feats]

        # FFN
        user_embeds = self.dense_layers(user_embeds)    # [batch_size, d_item]

        return user_embeds
        
    def sampled_softmax(self, user_embeds, target_item, history=None):
        """
        Args:
            user_embeds: [batch_size, d_item]
            target_item: [batch_size, 1]
        Returns:
            tensor, tensor: [batch_size, self.num_neg + 1], [batch_size, self.num_neg + 1]
        """
        batch_size = user_embeds.shape[0]
        sampled_items, mask = generate_negative_samples(target_item=target_item,
                                                        item_pool_size=self.item_pool_size,
                                                        num_neg=self.num_neg,
                                                        device=self.device,
                                                        history=history)

        # Item embedding
        sampled_items_embeds = self.itemEmbeds(sampled_items)        # [batch_size, num_neg + 1, d_item]

        # Calculate logits
        logits = (user_embeds.unsqueeze(1) * sampled_items_embeds).sum(dim=-1)            # [batch_size, num_neg + 1]
        logits_masked = torch.masked_fill(logits, mask, -1e9)

        # Softmax
        labels = torch.zeros(batch_size, dtype=torch.long).to(self.device)  # [batch_size]

        return logits_masked, labels
