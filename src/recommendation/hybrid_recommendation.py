"""Enhanced hybrid recommendation strategies."""
from typing import Dict, List, Tuple

from src.recommendation.collaborative_filtering import CollaborativeFiltering
from src.recommendation.content_based import ContentBasedRecommender


class HybridRecommender:
    def __init__(self, cf: CollaborativeFiltering, cb: ContentBasedRecommender):
        self.cf = cf
        self.cb = cb

    def _normalize(self, scores: List[Tuple[str, float]]) -> Dict[str, float]:
        if not scores:
            return {}
        vals = [s for _, s in scores]
        min_v, max_v = min(vals), max(vals)
        if max_v == min_v:
            return {i: 1.0 for i, _ in scores}
        return {i: (s - min_v) / (max_v - min_v) for i, s in scores}

    def switch_hybrid(self, user_id: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """
        优化版切换式混合：
        - 高活跃用户（交互>20）使用CF
        - 低活跃用户使用CB
        - 中等活跃用户使用加权融合
        """
        if self.cf and user_id in self.cf.matrix.index:
            user_ratings = self.cf.matrix.loc[user_id]
            active_count = (user_ratings > 0).sum()

            # 高活跃用户：使用CF
            if active_count >= 20:
                cf_recs = self.cf.recommend_user_cf(user_id, top_k)
                if cf_recs:
                    return cf_recs

            # 中等活跃用户：加权融合
            if 10 <= active_count < 20:
                cf_items = self.cf.recommend_user_cf(user_id, top_k * 2)
                cb_items = self.cb.recommend(user_id, top_k * 2, use_popularity=True)

                cf_scores = self._normalize(cf_items)
                cb_scores = self._normalize(cb_items)

                all_items = set(cf_scores) | set(cb_scores)
                fused = {
                    item: 0.4 * cf_scores.get(item, 0.0) + 0.6 * cb_scores.get(item, 0.0)
                    for item in all_items
                }
                ranked = sorted(fused.items(), key=lambda x: x[1], reverse=True)
                return ranked[:top_k]

        # 低活跃用户：使用CB
        return self.cb.recommend(user_id, top_k, use_popularity=True)

    def weighted_hybrid(self, user_id: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """线性加权融合"""
        cf_items = self.cf.recommend_user_cf(user_id, top_k * 2) if self.cf else []
        cb_items = self.cb.recommend(user_id, top_k * 2, use_popularity=True)

        cf_scores = self._normalize(cf_items)
        cb_scores = self._normalize(cb_items)

        all_items = set(cf_scores) | set(cb_scores)
        fused = {
            item: 0.3 * cf_scores.get(item, 0.0) + 0.7 * cb_scores.get(item, 0.0)
            for item in all_items
        }
        ranked = sorted(fused.items(), key=lambda x: x[1], reverse=True)
        return ranked[:top_k]

    def cascade_hybrid(self, user_id: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """级联融合"""
        cb = self.cb.recommend(user_id, top_k, use_popularity=True)
        cb_ids = {i for i, _ in cb}
        result = list(cb)

        if self.cf and len(result) < top_k:
            cf = self.cf.recommend_user_cf(user_id, top_k * 2)
            for item, score in cf:
                if item not in cb_ids and len(result) < top_k:
                    result.append((item, score * 0.8))

        return result[:top_k]

    def rank_hybrid(self, user_id: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """排序融合"""
        cf_items = self.cf.recommend_user_cf(user_id, top_k * 2) if self.cf else []
        cb_items = self.cb.recommend(user_id, top_k * 2, use_popularity=True)

        cf_scores = self._normalize(cf_items)
        cb_scores = self._normalize(cb_items)

        all_items = set(cf_scores) | set(cb_scores)
        fused = {}
        for item in all_items:
            cf_score = cf_scores.get(item, 0.0)
            cb_score = cb_scores.get(item, 0.0)
            # 交集奖励
            if cf_score > 0 and cb_score > 0:
                fused[item] = cf_score * 0.35 + cb_score * 0.65 + 0.05
            else:
                fused[item] = cf_score * 0.25 + cb_score * 0.75

        ranked = sorted(fused.items(), key=lambda x: x[1], reverse=True)
        return ranked[:top_k]