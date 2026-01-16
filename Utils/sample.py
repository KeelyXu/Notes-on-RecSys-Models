import torch

def generate_negative_samples(target_item,
                              item_pool_size: int, 
                              num_neg: int, device: str, 
                              history=None):
    """
    Args:
        target_item: tensor of shape [batch_size, 1]
        item_pool_size: size of item pool (include padding item)
        num_neg: number of negative samples per positive
        device: device to use
        history: tensor of shape [batch_size, max_len] (when passed, no negative sample will be in the history) or None
    Returns:
        sampled_items: tensor of shape [batch_size, num_neg + 1]
        mask: tensor of shape [batch_size, num_neg + 1]
    """
    batch_size = target_item.shape[0]
    neg_items = torch.randint(1, item_pool_size, (batch_size, num_neg), device=device)  # [batch_size, num_neg]
    sampled_items = torch.cat([target_item, neg_items], dim=1)  # [batch_size, num_neg + 1]
    mask = (neg_items == target_item)
    if history is not None:
        # negative items should not be in the history as well as be the target item
        # to avoid loop and do calculation efficiently, when the unwanted negative items are included, mask them
        # check validity of "negative" samples
        # history.unsqueeze(1): [batch_size, 1, max_len]
        # neg_items.unsqueeze(-1): [batch_size, num_neg, 1]
        history_hit = (history.unsqueeze(1) == neg_items.unsqueeze(-1))  # [batch_size, num_neg, max_len]
        mask = history_hit.any(dim=-1) | mask  # [batch_size, num_neg]

    mask = torch.concat([torch.zeros((batch_size, 1), dtype=torch.bool, device=mask.device), mask],
                        dim=1)  # [batch_size, num_neg + 1]

    return sampled_items, mask
