import torch
import torch.nn as nn
import torch.nn.functional as F
from Utils.sample import generate_negative_samples


class MIND(nn.Module):

    def __init__(self, d: int, K: int,
                 user_feats_embed_dim: dict[str, tuple[int, int]],
                 item_feats_n_cat: dict[str, int],
                 item_feats: torch.Tensor,
                 item_feat_names: list[str],
                 item_pool_size: int, 
                 max_len: int, num_neg: int,
                 dynamic_K: bool=True,
                 iter_num: int=3, device: str='cpu'):
        """
        Args:
            d: dimension of item vec
            K: max (when dynamic_K=True) number of capsules
            user_feats_embed_dim: a dict specifying n_cat and embed dim of each feature in user profile (include user ID)
            item_feats_n_cat: a dict specifying the n_cat of each feature in item profile
            item_feats: a tensor of shape [item_pool_size, num_feats]
            item_feat_names: a list, the i-th element specify the name of i-th feature in item_feats
            item_pool_size: size of item pool (include padding item)
            max_len: max length of user behavior sequence
            num_neg: number of negative samples per positive
            dynamic_K: whether K is dynamic (can be adjusted according to the user) or not
            iter_num: number of iterations in capsule calculation
            device: device to use
        """
        super().__init__()
        self.K = K
        self.item_pool_size = item_pool_size
        self.max_len = max_len
        self.iter_num = iter_num
        self.num_neg = num_neg
        self.dynamic_K = dynamic_K

        self.register_buffer('item_feats', item_feats)
        self.item_feat_names = item_feat_names
        self.device = device

        # Embedding layer
        self.itemIdEmbeds = nn.Embedding(item_pool_size, d, padding_idx=0)
        self.itemEmbeds = nn.ModuleDict()
        for feat_name, n_cat in item_feats_n_cat.items():
            self.itemEmbeds[feat_name] = nn.Embedding(n_cat, d)

        self.userEmbeds = nn.ModuleDict()
        self.user_profile_dim = 0
        for feat_name, (n_cat, embed_dim) in user_feats_embed_dim.items():
            self.userEmbeds[feat_name] = nn.Embedding(n_cat, embed_dim)
            self.user_profile_dim += embed_dim
        
        # Fully connected layers
        self.dense1 = nn.Linear(self.user_profile_dim + d, 4 * d)   # self.user_profile_dim + d: concatenate user profile and interest cap
        self.dense2 = nn.Linear(4 * d, d)

        # Shared bilinear mapping matrix
        self.S = nn.Parameter(torch.empty(d, d, device=device))
        nn.init.normal_(self.S, mean=0.0, std=1.0)

    def get_item_embedding(self, item_ids):
        """
        Helper function to get pooled item embeddings from IDs.
        Args:
            item_ids: [batch_size, num_items] or [num_items]
        Returns:
            tensor: [batch_size, num_items, d]
        """
        # Lookup side features from the buffer using item_ids
        batch_feats = self.item_feats[item_ids]     # [batch_size, num_items, num_feats] or [num_items, num_feats]

        embed_list = [self.itemIdEmbeds(item_ids)]  # Embedding item IDs to [batch_size, num_items, d] or [num_items, d]

        for i, feat_name in enumerate(self.item_feat_names):
            feat_val = batch_feats[..., i].long()   # [batch_size, num_items] or [num_items]
            embed_list.append(self.itemEmbeds[feat_name](feat_val))

        # Stack and Average Pooling
        feats = torch.stack(embed_list, dim=0)  # [num_feats+1, batch, num_items, d] or [num_feats+1, num_items, d]
        return torch.mean(feats, dim=0)  # [batch, num_items, d] or [num_items, d]

    def _get_capsule_mask(self, history):
        """
        Args:
            history: [batch_size, max_len]
        Returns:
            tensor: [batch_size, K]
        """
        batch_size = history.shape[0]
        cap_mask = torch.zeros(batch_size, self.K, device=self.device)
        if self.dynamic_K:
            interest_num = (history != 0).sum(dim=1)    # [batch_size]
            K = torch.clamp(torch.log2(interest_num.float()), min=1, max=self.K).long()    # [batch_size]
            pos = torch.arange(self.K, device=self.device).unsqueeze(0) # [1, self.K]
            cap_mask = pos >= K.unsqueeze(1)
        return cap_mask

    def forward(self, history, user_profile):
        """
        Args:
            history: [batch_size, max_len]
            user_profile: a dict, each value is of shape [batch_size, 1]
        Returns:
            tensor: [batch_size, K, d], [batch_size, K]
        """
        batch_size = history.shape[0]
        # initialize logits of capsule pairs
        B = torch.empty(self.K, self.max_len, device=self.device)       # [K, max_len]
        nn.init.normal_(B, mean=0.0, std=1.0)
        B = B.repeat(batch_size, 1, 1)   # [batch_size, K, max_len]
        # masking: ensure padding item capsules do not affect high-level capsules
        item_mask = (history == 0).unsqueeze(1)      # [batch_size, 1, max_len]
        cap_mask = self._get_capsule_mask(history).unsqueeze(2)   # [batch_size, K, 1]
        mask = item_mask | cap_mask     # [batch_size, K, max_len]

        # get embeddings of items
        item_embeds = self.get_item_embedding(history) # [batch, max_len, d]

        # avoid calculate this repeatedly
        Sc = torch.matmul(item_embeds, self.S)  # [batch_size, max_len, d]

        # update capsules in a iterative way
        for i in range(self.iter_num):
            B_masked = B.masked_fill(mask, -1e9)    # [batch_size, K, max_len]
            W = F.softmax(B_masked, dim=1)          # [batch_size, K, max_len]
            caps = torch.matmul(W, Sc)      # [batch_size, K, d]
            caps = self.squash(caps)        # [batch_size, K, d]
            # update B
            B += torch.matmul(caps, Sc.transpose(1, 2))     # [batch_size, K, max_len]

        # user embedding
        user_embeds = torch.empty(batch_size, 0, device=self.device)
        for feat in user_profile:
            user_embeds = torch.cat([user_embeds, self.userEmbeds[feat](user_profile[feat])], dim=1)    # [batch_size, self.user_profile_dim]

        # concatenate user profile vec with caps
        caps = torch.cat([caps, user_embeds.unsqueeze(1).repeat(1, self.K, 1)], dim=-1)
        caps = self.dense2(F.relu(self.dense1(caps)))    # [batch_size, K, d]

        return caps, cap_mask.squeeze(2)

    def squash(self, caps):
        """
        Args:
            caps: [batch_size, K, d]
        Returns:
            tensor: [batch_size, K, d]
        """
        norm = torch.norm(caps, dim=2, keepdim=True)    # [batch_size, K, 1]
        squared_norm = torch.pow(norm, 2)               # [batch_size, K, 1]

        return (squared_norm / ((1 + squared_norm) * norm + 1e-9)) * caps

    def layer_aware_attention(self, caps, cap_mask, target_items, p=2.0):
        """
        Args:
            caps: [batch_size, K, d]
            cap_mask: [batch_size, K]
            target_items: [batch_size, self.num_neg + 1]
        Returns:
            tensor: [batch_size, self.num_neg + 1, d]
        """
        target_item_embeds = self.get_item_embedding(target_items)    # [batch_size, self.num_neg + 1, d]
        attn_score = torch.matmul(target_item_embeds, caps.transpose(1, 2))  # [batch_size, self.num_neg + 1, K]
        attn_score = torch.pow(attn_score, p)   # [batch_size, self.num_neg + 1, K]
        attn_score = attn_score.masked_fill(cap_mask.unsqueeze(1), -1e9)

        v_user = torch.matmul(F.softmax(attn_score, dim=2), caps)  # [batch_size, self.num_neg + 1, d]

        score = (v_user * target_item_embeds).sum(dim=-1)          # [batch_size, self.num_neg + 1]

        return score

    def sampled_softmax(self, caps, cap_mask, target_item, history=None, p=2.0):
        """
        Args:
            caps: [batch_size, K, d]
            cap_mask: [batch_size, K]
            target_item: [batch_size, 1]
            history: tensor of shape [batch_size, max_len] (when passed, no negative sample will be in the history) or None
            p: float
        Returns:
            tensor, tensor: [batch_size, self.num_neg + 1], [batch_size, self.num_neg + 1]
        """
        batch_size = caps.shape[0]
        sampled_items, mask = generate_negative_samples(target_item=target_item,
                                                        item_pool_size=self.item_pool_size,
                                                        num_neg=self.num_neg,
                                                        device=self.device,
                                                        history=history)

        # calculate logits
        logits = self.layer_aware_attention(caps, cap_mask, sampled_items, p=p)    # [batch_size, self.num_neg + 1]
        logits_masked = logits.masked_fill(mask, -1e9)                  # [batch_size, self.num_neg + 1]
        labels = torch.concat([torch.ones(batch_size, 1), torch.zeros(batch_size, self.num_neg)], dim=1).to(self.device)  # [batch_size, self.num_neg + 1]

        return logits_masked, labels
