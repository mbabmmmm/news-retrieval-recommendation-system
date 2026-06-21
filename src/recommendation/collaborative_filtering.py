"""User-user and item-item collaborative filtering."""
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from config import CF_NEIGHBORS


class CollaborativeFiltering:
    def __init__(self, interaction_matrix: pd.DataFrame):
        self.matrix = interaction_matrix
        self.user_ids = interaction_matrix.index.tolist()
        self.item_ids = interaction_matrix.columns.tolist()

        if len(self.user_ids) > 0 and len(self.item_ids) > 0 and interaction_matrix.values.size > 0:
            # 使用更精确的相似度计算
            matrix_values = interaction_matrix.values
            # 添加平滑项
            self.user_sim = cosine_similarity(matrix_values + 0.01)
            self.item_sim = cosine_similarity(matrix_values.T + 0.01)
        else:
            self.user_sim = np.array([])
            self.item_sim = np.array([])

    def recommend_user_cf(self, user_id: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """基于用户的协同过滤推荐（改进版）"""
        if user_id not in self.matrix.index:
            return []
        if len(self.user_sim) == 0:
            return []

        u_idx = self.user_ids.index(user_id)
        sims = self.user_sim[u_idx]

        # 选择更多邻居
        neighbors = np.argsort(sims)[::-1][1:CF_NEIGHBORS * 2 + 1]

        scores: Dict[str, float] = {}
        weights_sum: Dict[str, float] = {}
        user_ratings = self.matrix.iloc[u_idx]

        for n_idx in neighbors:
            weight = sims[n_idx]
            if weight <= 0.1:  # 忽略相似度过低的邻居
                continue
            for item_id, rating in self.matrix.iloc[n_idx].items():
                if user_ratings[item_id] > 0:
                    continue
                scores[item_id] = scores.get(item_id, 0.0) + weight * rating
                weights_sum[item_id] = weights_sum.get(item_id, 0.0) + weight

        # 加权平均
        for item_id in scores:
            if weights_sum[item_id] > 0:
                scores[item_id] /= weights_sum[item_id]

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return ranked[:top_k]

    def recommend_item_cf(self, user_id: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """基于物品的协同过滤推荐"""
        if user_id not in self.matrix.index:
            return []
        if len(self.item_sim) == 0:
            return []

        user_ratings = self.matrix.loc[user_id]
        rated_items = user_ratings[user_ratings > 0]

        scores: Dict[str, float] = {}
        weights_sum: Dict[str, float] = {}

        for item_id, rating in rated_items.items():
            if item_id not in self.item_ids:
                continue
            i_idx = self.item_ids.index(item_id)
            sims = self.item_sim[i_idx]
            for j_idx, sim in enumerate(sims):
                if sim <= 0.1:
                    continue
                target = self.item_ids[j_idx]
                if user_ratings[target] > 0:
                    continue
                scores[target] = scores.get(target, 0.0) + sim * rating
                weights_sum[target] = weights_sum.get(target, 0.0) + sim

        for item_id in scores:
            if weights_sum[item_id] > 0:
                scores[item_id] /= weights_sum[item_id]

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return ranked[:top_k]