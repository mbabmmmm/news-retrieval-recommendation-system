"""TF-IDF 检索。"""
from typing import List, Tuple
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from src.preprocessing import tokenize


class TFIDFRetriever:
    def __init__(self, vectorizer, tfidf_matrix, article_ids: List[str]):
        self.vectorizer = vectorizer
        self.matrix = tfidf_matrix
        self.article_ids = article_ids

    def search(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        q = " ".join(tokenize(query))
        q_vec = self.vectorizer.transform([q])
        scores = cosine_similarity(q_vec, self.matrix).flatten()
        ranked = np.argsort(scores)[::-1][:top_k]
        return [(self.article_ids[i], float(scores[i])) for i in ranked if scores[i] > 0]