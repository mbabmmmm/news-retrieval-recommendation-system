"""Content-based recommendation with enhanced features."""
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler


class ContentBasedRecommender:
    def __init__(self, news_df: pd.DataFrame, user_df: pd.DataFrame):
        self.news_df = news_df.set_index("article_id")
        self.user_df = user_df.set_index("user_id")

        # 构建增强的文档特征
        texts = []
        for idx, row in news_df.iterrows():
            parts = []
            # 类别：重复3次
            if row.get('category') and str(row['category']).strip():
                parts.append(str(row['category']) * 3)
            # 关键词
            if row.get('keywords') and isinstance(row['keywords'], list):
                parts.append(' '.join(str(kw) for kw in row['keywords']))
            # 标题：重复2次
            if row.get('title') and str(row['title']).strip():
                parts.append(str(row['title']) * 2)
            # 内容摘要
            if row.get('content') and str(row['content']).strip():
                parts.append(str(row['content'])[:200])
            texts.append(' '.join(parts))

        self.vectorizer = TfidfVectorizer(
            token_pattern=r'(?u)\b\w+\b',
            min_df=2,
            max_df=0.85,
            ngram_range=(1, 2),
            sublinear_tf=True,
        )
        self.item_features = self.vectorizer.fit_transform(texts)
        self.article_ids = news_df["article_id"].tolist()

        # 计算综合流行度分数
        self.popularity_scores = self._compute_popularity_scores(news_df)

        # 计算质量分数（基于内容长度）
        self.quality_scores = self._compute_quality_scores(news_df)

        print(f"    内容推荐器词汇表大小: {len(self.vectorizer.vocabulary_)}")

    def _compute_popularity_scores(self, news_df):
        """计算增强的流行度分数"""
        max_views = news_df['views'].max()
        max_likes = news_df['likes'].max()
        if max_views == 0:
            max_views = 1
        if max_likes == 0:
            max_likes = 1

        # 组合浏览量、点赞数，并添加平滑
        scores = (news_df['views'] / max_views) * 0.5 + (news_df['likes'] / max_likes) * 0.3 + 0.2
        scaler = MinMaxScaler()
        return scaler.fit_transform(scores.values.reshape(-1, 1)).flatten()

    def _compute_quality_scores(self, news_df):
        """计算内容质量分数（基于标题和内容长度）"""
        title_len = news_df['title'].str.len()
        content_len = news_df['content'].str.len()
        max_title = title_len.max() if title_len.max() > 0 else 1
        max_content = content_len.max() if content_len.max() > 0 else 1

        scores = (title_len / max_title) * 0.4 + (content_len / max_content) * 0.4 + 0.2
        scaler = MinMaxScaler()
        return scaler.fit_transform(scores.values.reshape(-1, 1)).flatten()

    def _user_profile(self, user_id: str):
        """构建增强的用户画像"""
        if user_id not in self.user_df.index:
            return None

        user = self.user_df.loc[user_id]
        pref_cols = [c for c in self.user_df.columns if c.startswith("pref_")]

        profile_parts = []
        # 按偏好分数排序，优先考虑高偏好类别
        prefs = []
        for col in pref_cols:
            if user[col] > 0.1:
                category = col.replace('pref_', '')
                prefs.append((category, user[col]))

        # 按偏好分数降序排列
        prefs.sort(key=lambda x: x[1], reverse=True)

        for category, score in prefs:
            # 高权重类别重复更多次
            weight = int(score * 12)
            profile_parts.extend([category] * max(1, weight))

        if not profile_parts:
            return None

        profile_text = ' '.join(profile_parts)
        return self.vectorizer.transform([profile_text])

    def recommend(self, user_id: str, top_k: int = 10, use_popularity: bool = True) -> List[Tuple[str, float]]:
        """基于内容的推荐（综合内容、流行度、质量）"""
        profile = self._user_profile(user_id)
        if profile is None:
            return []

        # 计算内容相似度
        content_scores = cosine_similarity(profile, self.item_features).flatten()

        # 综合打分
        if use_popularity:
            final_scores = content_scores * 0.5 + self.popularity_scores * 0.3 + self.quality_scores * 0.2
        else:
            final_scores = content_scores

        ranked = np.argsort(final_scores)[::-1][:top_k]
        return [(self.article_ids[i], float(final_scores[i])) for i in ranked if final_scores[i] > 0.05]