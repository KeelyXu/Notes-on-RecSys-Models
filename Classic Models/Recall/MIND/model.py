import torch
import torch.nn as nn
import torch.nn.functional as F


class MIND(nn.Module):

    def __init__(self, d: int, K: int, user_profile_embed_dim: dict,
                 num_items: int, max_len: int, num_neg: int,
                 iter_num: int=3, device: str='cpu'):
        """
        Args
            d: dimension of item vec
            K: number of capsules
            user_profile_embed_dim: a dict specifying the embedding dimension of each feature in user profile
            num_items: size of item pool
            max_len: max length of user behavior sequence
            num_neg: number of negative samples per positive
            iter_num: number of iterations in capsule calculation
            device: device to use
        """
        super().__init__()
        self.K = K
        self.num_items = num_items
        self.max_len = max_len
        self.iter_num = iter_num
        self.num_neg = num_neg
        self.device = device

        # Embedding layer
        self.itemEmbeds = nn.Embedding(num_items, d, padding_idx=0)
        self.userEmbeds = nn.ModuleDict()
        self.user_profile_dim = 0
        for feat_name, (n_cat, embed_dim) in user_profile_embed_dim.items():
            self.userEmbeds[feat_name] = nn.Embedding(n_cat, embed_dim)
            self.user_profile_dim += embed_dim
        self.dense1 = nn.Linear(self.user_profile_dim + d, 4 * d)
        self.dense2 = nn.Linear(4 * d, d)

        # Shared bilinear mapping matrix
        self.S = nn.Parameter(torch.empty(d, d, device=device))
        nn.init.normal_(self.S, mean=0.0, std=1.0)

    def forward(self, history, user_profile):
        """
        Args:
            history: [batch_size, max_len]
            user_profile: a dict, each value is of shape [batch_size, 1]
        Returns:
            tensor: [batch_size, K, d]
        """
        batch_size = history.shape[0]
        # initialize logits of capsule pairs
        B = torch.empty(self.K, self.max_len, device=self.device)       # [K, max_len]
        nn.init.normal_(B, mean=0.0, std=1.0)
        B = B.repeat(batch_size, 1, 1)   # [batch_size, K, max_len]
        # masking: ensure padding item capsules do not affect high-level capsules
        mask = (history == 0).unsqueeze(1)      # [batch_size, 1, max_len]

        # perform embedding
        item_embeds = self.itemEmbeds(history)  # [batch_size, max_len, d]

        # avoid calculate this repeatedly
        Sc = torch.matmul(item_embeds, self.S)  # [batch_size, max_len, d]

        # update capsules in a iterative way
        for i in range(self.iter_num):
            B_masked = B.masked_fill(mask, -1e9)    # [batch_size, K, max_len]
            W = F.softmax(B_masked, dim=1)          # [batch_size, K, max_len]
            caps = torch.matmul(W, Sc)      # [batch_size, K, d]
            caps = self.squash(caps)        # [batch_size, K, d]
            # update B
            B = torch.matmul(caps, Sc.transpose(1, 2))     # [batch_size, K, max_len]

        # user embedding
        user_embeds = torch.empty(batch_size, 0, device=self.device)
        for feat in user_profile:
            user_embeds = torch.cat([user_embeds, self.userEmbeds[feat](user_profile[feat])], dim=1)

        # concatenate user profile vec with caps
        caps = torch.cat([caps, user_embeds.unsqueeze(1).repeat(1, 3, 1)], dim=-1)
        caps = self.dense2(F.relu(self.dense1(caps)))    # [batch_size, K, d]

        return caps

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

    def layer_aware_attention(self, caps, target_items, p=2.0):
        """
        Args:
            caps: [batch_size, K, d]
            target_items: [batch_size, self.num_neg + 1]
        Returns:
            tensor: [batch_size, self.num_neg + 1, d]
        """
        target_item_embeds = self.itemEmbeds(target_items)    # [batch_size, self.num_neg + 1, d]
        attn_score = torch.matmul(target_item_embeds, caps.transpose(1, 2))  # [batch_size, self.num_neg + 1, K]
        attn_score = torch.pow(attn_score, p)   # [batch_size, self.num_neg + 1, K]

        v_user = torch.matmul(F.softmax(attn_score, dim=2), caps)  # [batch_size, self.num_neg + 1, d]

        score = (v_user * target_item_embeds).sum(dim=-1)          # [batch_size, self.num_neg + 1]

        return score

    def sampled_softmax(self, caps, target_item, history, p=2.0):
        """
        Args:
            caps: [batch_size, K, d]
            target_item: [batch_size, 1]
            history: [batch_size, max_len]
            p: float
        Returns:
            tensor, tensor: [batch_size, self.num_neg + 1], [batch_size, self.num_neg + 1]
        """
        batch_size = caps.shape[0]
        # negative items should not be in the history as well as be the target item
        # to avoid loop and do calculation efficiently, when the unwanted negative items are included, mask them
        neg_items = torch.randint(1, self.num_items, (batch_size, self.num_neg), device=self.device)    # [batch_size, self.num_neg]
        target_items = torch.cat([target_item, neg_items], dim=1)   # [batch_size, self.num_neg + 1]

        # check validity of "negative" samples
        # history.unsqueeze(1): [batch_size, 1, max_len]
        # neg_items.unsqueeze(-1): [batch_size, self.num_neg, 1]
        history_hit = (history.unsqueeze(1) == neg_items.unsqueeze(-1))     # [batch_size, self.num_neg, max_len]
        mask = history_hit.any(dim=-1) | (neg_items == target_item)     # [batch_size, self.num_neg]
        mask = torch.concat([torch.zeros((batch_size, 1), dtype=torch.bool, device=mask.device), mask], dim=1)  # [batch_size, self.num_neg + 1]

        # calculate logits
        logits = self.layer_aware_attention(caps, target_items, p=p)    # [batch_size, self.num_neg + 1]
        logits_masked = logits.masked_fill(mask, -1e9)                  # [batch_size, self.num_neg + 1]
        labels = torch.concat([torch.ones(batch_size, 1), torch.zeros(batch_size, self.num_neg)], dim=1)  # [batch_size, self.num_neg + 1]

        return logits_masked, labels
