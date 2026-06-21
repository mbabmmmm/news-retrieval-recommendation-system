"""Enhanced retrieval + recommendation fusion system."""
from typing import Dict, List, Tuple

from config import TOP_K_RETRIEVAL


class HybridSystem:
    def __init__(self, retriever, recommender, news_df):
        self.retriever = retriever
        self.recommender = recommender
        self.news_df = news_df.set_index("article_id")

    def _normalize(self, scores: List[Tuple[str, float]]) -> Dict[str, float]:
        if not scores:
            return {}
        vals = [s for _, s in scores]
        min_v, max_v = min(vals), max(vals)
        if max_v == min_v:
            return {i: 1.0 for i, _ in scores}
        return {i: (s - min_v) / (max_v - min_v) for i, s in scores}

    def _query_category_match(self, query: str) -> str:
        """判断查询属于哪个类别"""
        category_map = {
            '科技': ['科技', '技术', 'AI', '人工智能', '5G', '云计算', '编程'],
            '健康': ['健康', '医疗', '养生', '保健', '医院'],
            '汽车': ['汽车', '车', '新能源', '电动', '自动驾驶'],
            '文化': ['文化', '艺术', '文学', '历史', '传统'],
            '国际': ['国际', '全球', '世界', '海外', '外交'],
            '体育': ['体育', '足球', '篮球', '奥运', '运动'],
            '娱乐': ['娱乐', '明星', '电影', '音乐', '综艺'],
            '财经': ['财经', '经济', '金融', '股市', '投资'],
            '教育': ['教育', '学校', '大学', '学习', '培训'],
            '社会': ['社会', '民生', '新闻', '事件']
        }
        query_lower = query.lower()
        for cat, keywords in category_map.items():
            if query_lower in [k.lower() for k in keywords] or query_lower in cat:
                return cat
        return '科技'

    def search_and_recommend(
            self,
            query: str,
            user_id: str,
            top_k: int = TOP_K_RETRIEVAL,
    ) -> List[Tuple[str, float, str]]:
        """
        增强融合：结合检索、推荐、类别匹配
        """
        # 获取检索结果
        retrieval = self.retriever.search(query, top_k * 3)
        r_scores = self._normalize(retrieval)

        # 获取推荐结果
        recommendations = self.recommender.rank_hybrid(user_id, top_k * 3)
        c_scores = self._normalize(recommendations)

        # 获取查询类别
        query_category = self._query_category_match(query)

        all_ids = set(r_scores) | set(c_scores)
        fused = []

        for aid in all_ids:
            r_score = r_scores.get(aid, 0.0)
            c_score = c_scores.get(aid, 0.0)

            # 基础融合分数
            score = 0.45 * r_score + 0.55 * c_score

            # 类别匹配奖励
            if aid in self.news_df.index:
                news_category = str(self.news_df.loc[aid, 'category'])
                if news_category == query_category:
                    score = min(1.0, score + 0.15)

            # 标题包含查询词奖励
            title = str(self.news_df.loc[aid, 'title']) if aid in self.news_df.index else ""
            if query in title:
                score = min(1.0, score + 0.1)

            fused.append((aid, score, title))

        fused.sort(key=lambda x: x[1], reverse=True)
        return fused[:top_k]