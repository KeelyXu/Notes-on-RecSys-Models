import numpy as np


def recall_at_k(actual: list[list], predicted: list[list], k=10):
    """
    Computes the mean Recall at K (Recall@K).

    Recall@K measures the proportion of relevant items found in the top-k recommendations.
    Formula: Recall@K = |Relevant_Items_in_Top_K| / |Total_Relevant_Items|

    Args:
        actual (list of list): A list of lists where each inner list contains the
                               ground truth (relevant) item IDs for a user.
        predicted (list of list): A list of lists where each inner list contains the
                                  predicted item IDs for a user (sorted by relevance score).
        k (int): The number of top recommendations to consider. Default is 10.

    Returns:
        float: The mean Recall@K score across all users.
    """
    if len(actual) != len(predicted):
        raise ValueError("The lengths of actual and predicted lists must be the same.")

    recalls = []

    for true_items, pred_items in zip(actual, predicted):
        pred_items_k = pred_items[:k]

        true_set = set(true_items)
        pred_set = set(pred_items_k)

        # Calculate the number of hits (items present in both ground truth and top-k predictions)
        hits = len(true_set & pred_set)

        # Calculate Recall: hits / total_relevant_items
        # Handle the edge case where a user has no relevant items (ground truth is empty)
        if len(true_set) > 0:
            recalls.append(hits / len(true_set))
        else:
            # If there are no relevant items, recall is strictly 0.0 (or undefined, treated as 0 here)
            recalls.append(0.0)

    return np.mean(recalls)


def hit_rate_at_k(actual: list[list], predicted: list[list], k=10):
    """
    Computes the mean Hit Rate at K (HR@K).

    HR@K measures whether the test item is present in the top-k recommendations.
    It is a binary metric: 1 if at least one relevant item is in top-k, else 0.

    Args:
        actual (list of list): A list of lists containing ground truth item IDs.
        predicted (list of list): A list of lists containing predicted item IDs.
        k (int): The number of top recommendations to consider. Default is 10.

    Returns:
        float: The mean Hit Rate@K score across all users.
    """
    if len(actual) != len(predicted):
        raise ValueError("The lengths of actual and predicted lists must be the same.")

    hits = []

    for true_items, pred_items in zip(actual, predicted):
        pred_items_k = pred_items[:k]

        true_set = set(true_items)
        pred_set = set(pred_items_k)

        # If there is at least one common item, it's a hit (1.0), otherwise miss (0.0)
        if len(true_set & pred_set) > 0:
            hits.append(1.0)
        else:
            hits.append(0.0)

    return np.mean(hits)