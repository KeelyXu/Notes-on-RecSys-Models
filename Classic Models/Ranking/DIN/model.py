import torch
import torch.nn as nn
import torch.nn.functional as F


class DIN(nn.Module):

    def __init__(self,
                 d_model: int,
                 user_feats_n_cat: dict[str, int],
                 item_feats_n_cat: dict[str, int],
                 ctx_feats_n_cat: dict[str, int],
                 act_func: str='Dice',
                 device: str='cpu'):
        """
        Args:
            d_model: dimensionality of embedding vector
            user_feats_n_cat: a dict specifying the n_cat of each feature in user profile
            item_feats_n_cat: a dict specifying the n_cat of each feature in item profile
            ctx_feats_n_cat: a dict specifying the n_cat of each feature in context
            act_func: Dice or PReLU
            device: device to use
        """
        super().__init__()
        self.device = device

        # Embedding layers
        self.userEmbeds = nn.ModuleDict()
        for feat_name, n_cat in user_feats_n_cat.items():
            self.userEmbeds[feat_name] = nn.Embedding(n_cat, d_model)
        self.user_embed_dim = len(user_feats_n_cat) * d_model

        self.itemEmbeds = nn.ModuleDict()
        for feat_name, n_cat in item_feats_n_cat.items():
            self.itemEmbeds[feat_name] = nn.Embedding(n_cat, d_model)
        self.item_embed_dim = len(item_feats_n_cat) * d_model

        self.ctxEmbeds = nn.ModuleDict()
        for feat_name, n_cat in ctx_feats_n_cat.items():
            self.ctxEmbeds[feat_name] = nn.Embedding(n_cat, d_model)
        self.ctx_embed_dim = len(ctx_feats_n_cat) * d_model

        # FFN in Local activation unit
        self.act_unit = nn.Sequential(
            nn.Linear(4 * self.item_embed_dim, 36),
            Dice(36, dim=3) if act_func == 'Dice' else nn.PReLU(),     # input dim: [batch_size, seq_len, d]
            nn.Linear(36, 1)
        )

        # FC layers
        input_dim = self.user_embed_dim + 2 * self.item_embed_dim + self.ctx_embed_dim
        self.fc_layers = nn.Sequential(
            nn.Linear(input_dim, 200),
            Dice(200) if act_func == 'Dice' else nn.PReLU(),    # input dim: [batch_size, d]
            nn.Linear(200, 80),
            Dice(80) if act_func == 'Dice' else nn.PReLU(),     # input dim: [batch_size, d]
            nn.Linear(80, 2)
        )

    def forward(self, user_profiles, user_behaviors, candidate, ctxs, mask=None):
        """
        Args:
            user_profiles: a dict of user_feat_name -> [batch_size]
            user_behaviors: a dict of item_feat_name -> [batch_size, seq_len]
            candidate: a dict of item_feat_name -> [batch_size]
            ctxs: a dict of ctx_feat_name -> [batch_size]
            mask: [batch_size, seq_len] or None (0 for padding)
        Returns:
            tensor: [batch_size, 2]
        """
        # Embedding & Concat
        user_embeds = []
        for feat in user_profiles:
            user_embeds.append(self.userEmbeds[feat](user_profiles[feat]))
        user_embeds = torch.cat(user_embeds, dim=-1)  # [batch_size, self.user_embed_dim]

        item_embeds = []
        for feat in user_behaviors:
            item_embeds.append(self.itemEmbeds[feat](user_behaviors[feat])) # [batch_size, seq_len, d_model]
        item_embeds = torch.cat(item_embeds, dim=-1)  # [batch_size, seq_len, self.item_embed_dim]

        candidate_embeds = []
        for feat in candidate:
            candidate_embeds.append(self.itemEmbeds[feat](candidate[feat]))
        candidate_embeds = torch.cat(candidate_embeds, dim=-1) # [batch_size, self.item_embed_dim]
        candidate_embeds_exp = candidate_embeds.unsqueeze(1).expand(-1, item_embeds.size(1), -1)    # [batch_size, seq_len, self.item_embed_dim]

        ctx_embeds = []
        for feat in ctxs:
            ctx_embeds.append(self.ctxEmbeds[feat](ctxs[feat]))
        ctx_embeds = torch.cat(ctx_embeds, dim=-1)   # [batch_size, self.ctx_embed_dim]

        # Activation Unit
        attn_inputs = torch.cat([item_embeds, candidate_embeds_exp, candidate_embeds_exp - item_embeds, candidate_embeds_exp * item_embeds],
                           dim=-1)  # [batch_size, seq_len, 4 * self.item_embed_dim]
        attn_weights = self.act_unit(attn_inputs)    # [batch_size, seq_len, 1]
        attn_weights = attn_weights * mask.unsqueeze(-1)
        item_pooling = torch.sum(attn_weights * item_embeds, dim=1)    # [batch_size, self.item_embed_dim]

        # FC
        fc_inputs = torch.cat([user_embeds, item_pooling, candidate_embeds, ctx_embeds], dim=-1)    # [batch_size, self.user_embed_dim + 2 * self.item_embed_dim + self.ctx_embed_dim]
        outputs = self.fc_layers(fc_inputs) # [batch_size, 2]

        return F.softmax(outputs, dim=-1)


class Dice(nn.Module):

    def __init__(self, d: int, dim: int=2, epsilon: float=1e-8):
        super().__init__()
        self.dim = dim
        self.bn = nn.BatchNorm1d(d, eps=epsilon)
        self.sigmoid = nn.Sigmoid()

        if self.dim == 2:
            self.alpha = nn.Parameter(torch.zeros(d))
        else:
            self.alpha = nn.Parameter(torch.zeros(d, 1))

    def forward(self, s):
        """
        Args:
            s: [batch_size, seq_len, d] or [batch_size, d]
        Returns:
            tensor: in the same shape of s
        """
        assert s.dim() == self.dim
        if self.dim == 2:
            p_s = self.sigmoid(self.bn(s))
            out = p_s * s + (1 - p_s) * self.alpha * s
        else:
            s = s.transpose(1, 2)   # [batch_size, d, seq_len]
            p_s = self.sigmoid(self.bn(s))
            out = p_s * s + (1 - p_s) * self.alpha * s
            out = out.transpose(1, 2)

        return out
