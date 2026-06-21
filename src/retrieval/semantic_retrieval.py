"""Enhanced semantic retrieval using Word2Vec embeddings."""
from typing import List, Tuple

import numpy as np
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity

from src.preprocessing import tokenize


class SemanticRetriever:
    def __init__(self, tokenized_docs: List[str], article_ids: List[str], vector_size: int = 150):
        self.article_ids = article_ids
        self.token_lists = [doc.split() for doc in tokenized_docs]

        # 过滤空文档
        valid_sentences = [tokens for tokens in self.token_lists if len(tokens) > 0]

        if len(valid_sentences) >= 3:
            self.model = Word2Vec(
                sentences=valid_sentences,
                vector_size=vector_size,
                window=5,
                min_count=2,  # 提高min_count
                workers=1,
                seed=42,
                epochs=15,  # 增加训练轮数
                sg=1,  # 使用Skip-gram
            )
            self.doc_vectors = np.array([self._doc_vector(tokens) for tokens in self.token_lists])
        else:
            self.model = None
            self.doc_vectors = np.zeros((len(tokenized_docs), vector_size))

    def _doc_vector(self, tokens: List[str]) -> np.ndarray:
        if self.model is None:
            return np.zeros(150)
        vecs = [self.model.wv[t] for t in tokens if t in self.model.wv]
        if not vecs:
            return np.zeros(self.model.vector_size)
        # 使用加权平均
        return np.average(vecs, axis=0)

    def search(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        if self.model is None:
            return []

        q_tokens = tokenize(query)
        if not q_tokens:
            return []

        q_vec = self._doc_vector(q_tokens).reshape(1, -1)
        if np.allclose(q_vec, 0):
            return []

        scores = cosine_similarity(q_vec, self.doc_vectors).flatten()

        # 添加一个小的平滑项
        scores = scores + 0.01

        ranked = np.argsort(scores)[::-1][:top_k]
        return [(self.article_ids[i], float(scores[i])) for i in ranked if scores[i] > 0.01]