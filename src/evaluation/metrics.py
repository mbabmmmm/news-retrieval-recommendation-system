"""Evaluation metrics for retrieval, recommendation and hybrid system."""
from typing import Dict, List, Set

import numpy as np


def precision_at_k(relevant: Set[str], retrieved: List[str], k: int) -> float:
    top = retrieved[:k]
    if not top:
        return 0.0
    return len(set(top) & relevant) / len(top)


def recall_at_k(relevant: Set[str], retrieved: List[str], k: int) -> float:
    if not relevant:
        return 0.0
    top = retrieved[:k]
    # 确保至少返回一个文档
    if len(top) == 0:
        return 0.0
    return len(set(top) & relevant) / len(relevant)


def f1_at_k(relevant: Set[str], retrieved: List[str], k: int) -> float:
    p = precision_at_k(relevant, retrieved, k)
    r = recall_at_k(relevant, retrieved, k)
    if p + r == 0:
        return 0.0
    return 2 * p * r / (p + r)


def average_precision(relevant: Set[str], retrieved: List[str]) -> float:
    if not relevant or not retrieved:
        return 0.0
    score = 0.0
    hits = 0
    for i, doc in enumerate(retrieved, 1):
        if doc in relevant:
            hits += 1
            score += hits / i
    return score / len(relevant)


def ndcg_at_k(relevant: Set[str], retrieved: List[str], k: int) -> float:
    if not retrieved:
        return 0.0
    dcg = 0.0
    for i, doc in enumerate(retrieved[:k], 1):
        rel = 1.0 if doc in relevant else 0.0
        dcg += rel / np.log2(i + 1)
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / np.log2(i + 1) for i in range(1, ideal_hits + 1))
    return dcg / idcg if idcg > 0 else 0.0


def mae(y_true: List[float], y_pred: List[float]) -> float:
    if not y_true:
        return 0.0
    return float(np.mean(np.abs(np.array(y_true) - np.array(y_pred))))


def rmse(y_true: List[float], y_pred: List[float]) -> float:
    if not y_true:
        return 0.0
    return float(np.sqrt(np.mean((np.array(y_true) - np.array(y_pred)) ** 2)))


def hit_ratio(relevant: Set[str], retrieved: List[str], k: int) -> float:
    top = retrieved[:k]
    return 1.0 if set(top) & relevant else 0.0


def evaluate_retrieval(queries: Dict[str, Set[str]], results: Dict[str, List[str]], k: int = 10):
    metrics = {"precision": [], "recall": [], "f1": [], "map": [], "ndcg": []}
    for qid, rel in queries.items():
        ret = results.get(qid, [])
        metrics["precision"].append(precision_at_k(rel, ret, k))
        metrics["recall"].append(recall_at_k(rel, ret, k))
        metrics["f1"].append(f1_at_k(rel, ret, k))
        metrics["map"].append(average_precision(rel, ret))
        metrics["ndcg"].append(ndcg_at_k(rel, ret, k))
    return {m: float(np.mean(v)) for m, v in metrics.items()}


def evaluate_recommendation(
    test_data: Dict[str, Set[str]],
    predictions: Dict[str, List[str]],
    rating_true: Dict[str, float],
    rating_pred: Dict[str, float],
    k: int = 10,
):
    p_list, r_list, f1_list, hit_list = [], [], [], []
    for uid, rel in test_data.items():
        pred = predictions.get(uid, [])
        p_list.append(precision_at_k(rel, pred, k))
        r_list.append(recall_at_k(rel, pred, k))
        f1_list.append(f1_at_k(rel, pred, k))
        hit_list.append(hit_ratio(rel, pred, k))

    common_keys = set(rating_true) & set(rating_pred)
    y_true = [rating_true[k] for k in common_keys]
    y_pred = [rating_pred[k] for k in common_keys]

    return {
        "precision": float(np.mean(p_list)) if p_list else 0.0,
        "recall": float(np.mean(r_list)) if r_list else 0.0,
        "f1": float(np.mean(f1_list)) if f1_list else 0.0,
        "mae": mae(y_true, y_pred),
        "rmse": rmse(y_true, y_pred),
        "hit_ratio": float(np.mean(hit_list)) if hit_list else 0.0,
    }