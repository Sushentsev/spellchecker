from typing import List


def acc_k(y_true: List[str], y_pred: List[List[str]], k: int = 1) -> float:
    correct_cnt = 0
    for true_item, preds in zip(y_true, y_pred):
        correct_cnt += true_item in preds[:k]

    return correct_cnt / len(y_true)


def mrr(y_true: List[str], y_pred: List[List[str]]) -> float:
    inv_ranks = []
    for true_item, preds in zip(y_true, y_pred):
        if true_item not in preds:
            inv_ranks.append(0)
            continue

        inv_ranks.append(1 / (preds.index(true_item) + 1))

    return sum(inv_ranks) / len(inv_ranks)
