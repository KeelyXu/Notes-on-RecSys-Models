import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.nn.functional as F

from Utils.sample import generate_negative_samples


class SDM(nn.Module):

    def __init__(self,
                 d_model: int, num_heads: int,
                 user_feats_embed_dim: dict[str, tuple[int, int]],
                 item_feats_n_cat: dict[str, int],
                 item_feats: torch.Tensor,
                 item_feat_names: list[str],
                 lstm_layer_num: int,
                 item_pool_size: int,
                 num_neg: int,
                 device: str='cpu'):
        """
        Note: all features of item use idx=0 as padding index.

        Args:
            d_model: "d" in paper, most vectors are in dimension `d`
            num_heads: number of attention heads
            user_feats_embed_dim: a dict specifying n_cat and embed dim of each feature in user profile (include user ID)
            item_feats_n_cat: a dict specifying the n_cat of each feature in item profile
            item_feats: a tensor of shape [item_pool_size, num_feats]
            item_feat_names: a list, the i-th element specify the name of i-th feature in item_feats
            lstm_layer_num: number of LSTM layers
            item_pool_size: size of item pool (include padding item)
            num_neg: number of negative samples per positive
            device: device to use
        """
        super().__init__()
        self.register_buffer('item_feats', item_feats)
        self.item_feat_names = item_feat_names
        self.item_pool_size = item_pool_size
        self.num_neg = num_neg
        self.device = device

        # Embedding Layers
        self.userEmbeds = nn.ModuleDict()
        self.user_embed_dim = 0
        for feat_name, (n_cat, embed_dim) in user_feats_embed_dim.items():
            self.userEmbeds[feat_name] = nn.Embedding(n_cat, embed_dim)
            self.user_embed_dim += embed_dim

        self.itemEmbeds = nn.ModuleDict()
        self.item_embed_dim = 0
        for feat_name, n_cat in item_feats_n_cat.items():
            self.itemEmbeds[feat_name] = nn.Embedding(n_cat, d_model)
            self.item_embed_dim += d_model

        self.user_proj = nn.Linear(self.user_embed_dim, d_model)
        self.item_proj = nn.Linear(self.item_embed_dim, d_model)

        # LSTM Layers (layers are stored separately because we will add residuals between them)
        self.lstm_layers = nn.ModuleList()
        for _ in range(lstm_layer_num - 1):
            self.lstm_layers.append(nn.LSTM(d_model, d_model, batch_first=True))
            self.lstm_layers.append(nn.Dropout(0.2))

        self.lstm_layers.append(nn.LSTM(d_model, d_model, batch_first=True))
        self.lstm_layers.append(nn.Identity())

        # Short-term modeling
        self.self_mha = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
        self.dropout = nn.Dropout(0.1)
        self.layer_norm = nn.LayerNorm(d_model)

        # Long-term modeling
        self.dense = nn.Sequential(nn.Linear(self.item_embed_dim, d_model), nn.Tanh())

        # Gating
        self.W1 = nn.Linear(d_model, d_model)
        self.W2 = nn.Linear(d_model, d_model)
        self.W3 = nn.Linear(d_model, d_model)

    def get_user_embedding(self, user_profile):
        """
        Args:
            user_profile: a dict, each value is of shape [batch_size]
        Returns:
            tensor: [batch_size, d]
        """
        batch_size = list(user_profile.values())[0].shape[0]
        user_embeds = torch.empty(batch_size, 0, device=self.device)
        for feat in user_profile:
            user_embeds = torch.cat([user_embeds, self.userEmbeds[feat](user_profile[feat])],
                                    dim=1)  # [batch_size, self.user_embed_dim]
        user_embeds_proj = self.user_proj(user_embeds)  # [batch_size, d_model]

        return user_embeds_proj

    def get_item_embedding(self, item_ids):
        """
        Helper function to get pooled item embeddings from IDs.
        Args:
            item_ids: [...]
        Returns:
            tensor: [..., d]
        """
        # Lookup side features from the buffer using item_ids
        batch_feats = self.item_feats[item_ids]  # [..., num_feats]

        embed_list = []

        for i, feat_name in enumerate(self.item_feat_names):
            feat_val = batch_feats[..., i].long()  # [...]
            embed_list.append(self.itemEmbeds[feat_name](feat_val))

        # Concatenation
        item_embeds = torch.cat(embed_list, dim=-1)     # [..., self.item_embed_dim]

        item_embeds_proj = self.item_proj(item_embeds)  # [..., d_model]

        return item_embeds_proj

    def lstm(self, short_term_seq_embedded, lengths):
        """
        Args:
            short_term_seq_embedded: [batch_size, seq_len, d_model]
            lengths: [batch_size]
        Returns:
            tensor: [batch_size, seq_len, d_model]
        """
        out = short_term_seq_embedded

        for i in range(0, len(self.lstm_layers), 2):
            lstm = self.lstm_layers[i]
            dropout = self.lstm_layers[i + 1]

            input_tensor = out

            # pack
            packed_input = pack_padded_sequence(
                input_tensor,
                lengths.cpu(),
                batch_first=True,
                enforce_sorted=False
            )

            packed_out, _ = lstm(packed_input)

            # unpack
            out, _ = pad_packed_sequence(
                packed_out,
                batch_first=True,
                total_length=input_tensor.size(1)
            )

            # residual
            out = dropout(out) + input_tensor

        return out

    def user_attn(self, seqs, user_embeds, mask):
        """
        Args:
            seqs: [batch_size, seq_len, d_model]
            user_embeds: [batch_size, d_model]
            mask: [batch_size, seq_len]
        Returns:
            tensor: [batch_size, d_model]
        """

        user_attn_score = torch.matmul(seqs, user_embeds.unsqueeze(-1)).squeeze(-1)  # [batch_size, seq_len]
        user_attn_score = torch.masked_fill(user_attn_score, mask, -1e9)  # [batch_size, seq_len]

        attn_weights = F.softmax(user_attn_score, dim=-1).unsqueeze(-1)  # [batch_size, seq_len, 1]
        V = (attn_weights * seqs).sum(dim=1)  # [batch_size, d_model]

        return V

    def short_term_representation(self, short_term_seq, lengths, user_embeds):
        """
        Args:
            short_term_seq: [batch_size, seq_len]
            lengths: [batch_size]
            user_embeds: [batch_size, d_model]
        Returns:
            tensor: [batch_size, d_model]
        """
        mask = (short_term_seq == 0)    # [batch_size, seq_len]
        # Item embedding
        seq_embeds = self.get_item_embedding(short_term_seq)    # [batch_size, seq_len, d_model]

        # LSTM
        hidden_states_1 = self.lstm(seq_embeds, lengths)   # [batch_size, seq_len, d_model]

        # Self-attention
        hidden_states_2, _ = self.self_mha(hidden_states_1, hidden_states_1, hidden_states_1,
                                           key_padding_mask=mask)  # [batch_size, seq_len, d_model]

        # Residual Dropout & Layer Norm
        hidden_states = self.layer_norm(hidden_states_1 + self.dropout(hidden_states_2))

        # User attention
        S = self.user_attn(hidden_states, user_embeds, mask)

        return S

    def long_term_representation(self, long_term_seqs, user_embeds):
        """
        Args:
            long_term_seqs: a dict of item_feat_name -> [batch_size, seq_len]
            user_embeds: [batch_size, d_model]
        Returns:
            tensor: [batch_size, d_model]
        """
        long_term_seqs_out = {}

        # Feat embedding
        for feat_name in self.itemEmbeds:
            long_term_seqs_out[feat_name] = self.itemEmbeds[feat_name](long_term_seqs[feat_name])  # [batch_size, seq_len, d_model]

        # User attention
        for feat_name in self.itemEmbeds:
            # Create mask for long-term sequences (padding is 0)
            long_term_mask = (long_term_seqs[feat_name] == 0)  # [batch_size, seq_len]
            long_term_seqs_out[feat_name] = self.user_attn(long_term_seqs_out[feat_name], user_embeds, long_term_mask)  # [batch_size, d_model]

        # Concatenation
        seqs_rep = torch.cat(list(long_term_seqs_out.values()), dim=1)     # [batch_size, self.item_embed_dim]

        # Dense layer
        L = self.dense(seqs_rep)    # [batch_size, d_model]

        return L

    def gate_fusion(self, user_embeds, S, L):
        """
        Args:
            user_embeds: [batch_size, d_model]
            S: [batch_size, d_model]
            L: [batch_size, d_model]
        Returns:
            tensor: [batch_size, d_model]
        """
        G = F.sigmoid(self.W1(user_embeds) + self.W2(S) + self.W3(L))   # [batch_size, d_model]

        O = (1 - G) * L + G * S     # [batch_size, d_model]

        return O

    def forward(self, user_profile, short_term_seq, lengths, long_term_seqs):
        """
        Args:
            user_profile: user_profile: a dict, each value is of shape [batch_size]
            short_term_seq: [batch_size, seq_len]
            lengths: [batch_size]
            long_term_seqs: a dict of item_feat_name -> [batch_size, seq_len]
        Returns:
            tensor: [batch_size, d_model]
        """
        user_embeds = self.get_user_embedding(user_profile)
        S = self.short_term_representation(short_term_seq, lengths, user_embeds)
        L = self.long_term_representation(long_term_seqs, user_embeds)

        O = self.gate_fusion(user_embeds, S, L)

        return O

    def sampled_softmax(self, user_embeds, target_item):
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
                                                        device=self.device)

        # Item embedding
        sampled_items_embeds = self.get_item_embedding(sampled_items)  # [batch_size, num_neg + 1, d_model]

        # Calculate logits
        logits = (user_embeds.unsqueeze(1) * sampled_items_embeds).sum(dim=-1)  # [batch_size, num_neg + 1]
        logits_masked = torch.masked_fill(logits, mask, -1e9)

        # Softmax
        labels = torch.zeros(batch_size, dtype=torch.long).to(self.device)  # [batch_size]

        return logits_masked, labels
