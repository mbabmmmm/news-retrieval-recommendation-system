"""BM25 检索（优化版）。"""
from typing import List, Tuple
import numpy as np
from config import BM25_K1, BM25_B
from src.preprocessing import tokenize


class BM25Retriever:
    def __init__(self, vectorizer, dtm, article_ids: List[str]):
        self.vectorizer = vectorizer
        self.dtm = dtm
        self.article_ids = article_ids
        self.N = dtm.shape[0]
        self.avgdl = float(dtm.sum(axis=1).mean())
        self.doc_lens = np.array(dtm.sum(axis=1)).flatten()
        df = np.array((dtm > 0).sum(axis=0)).flatten()
        # 改进的IDF计算
        self.idf = np.log((self.N - df + 0.5) / (df + 0.5) + 1.0)

    def _score(self, q_terms, doc_idx):
        score = 0.0
        doc_vec = self.dtm[doc_idx].toarray().flatten()
        dl = self.doc_lens[doc_idx]
        for term in q_terms:
            if term not in self.vectorizer.vocabulary_:
                continue
            tid = self.vectorizer.vocabulary_[term]
            tf = doc_vec[tid]
            if tf == 0:
                continue
            # BM25公式
            denom = tf + BM25_K1 * (1 - BM25_B + BM25_B * dl / self.avgdl)
            score += self.idf[tid] * (tf * (BM25_K1 + 1)) / denom
        return score

    def search(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        terms = tokenize(query)
        if not terms:
            return []
        scores = [self._score(terms, i) for i in range(self.N)]
        ranked = np.argsort(scores)[::-1][:top_k]
        return [(self.article_ids[i], float(scores[i])) for i in ranked if scores[i] > 0]