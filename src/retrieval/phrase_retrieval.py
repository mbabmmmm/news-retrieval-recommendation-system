"""短语检索（位置索引）- 增强版。"""
from typing import List, Tuple
from src.preprocessing import tokenize


class PhraseRetriever:
    def __init__(self, tokenized_docs: List[str], article_ids: List[str]):
        self.article_ids = article_ids
        self.token_lists = [d.split() for d in tokenized_docs]
        self.pos_index = self._build_index()

    def _build_index(self):
        index = {}
        for doc_id, tokens in enumerate(self.token_lists):
            for pos, term in enumerate(tokens):
                index.setdefault(term, {}).setdefault(doc_id, []).append(pos)
        return index

    def search(self, phrase: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """增强短语搜索，支持精确匹配和模糊匹配"""
        terms = tokenize(phrase)
        if len(terms) == 0:
            return []

        # 单次查询
        if len(terms) == 1:
            term = terms[0]
            if term not in self.pos_index:
                return []
            matched = [(self.article_ids[doc_id], 1.0)
                       for doc_id in self.pos_index[term].keys()]
            matched.sort(key=lambda x: x[0])
            return matched[:top_k]

        # 精确短语匹配（连续）
        exact_matches = []
        if terms[0] in self.pos_index:
            for doc_id, positions in self.pos_index[terms[0]].items():
                doc_tokens = self.token_lists[doc_id]
                for start in positions:
                    # 检查是否连续
                    if start + len(terms) - 1 >= len(doc_tokens):
                        continue
                    match = True
                    for offset, t in enumerate(terms[1:], 1):
                        if doc_tokens[start + offset] != t:
                            match = False
                            break
                    if match:
                        exact_matches.append((self.article_ids[doc_id], 1.0))
                        break

        if exact_matches:
            return exact_matches[:top_k]

        # 宽松匹配1：所有词都在文档中（顺序无关）
        all_terms_docs = None
        for term in terms:
            if term in self.pos_index:
                term_docs = set(self.pos_index[term].keys())
                if all_terms_docs is None:
                    all_terms_docs = term_docs
                else:
                    all_terms_docs &= term_docs

        if all_terms_docs:
            # 计算匹配分数（基于位置接近度）
            results = []
            for doc_id in all_terms_docs:
                # 获取所有词的位置
                positions = []
                for term in terms:
                    positions.extend(self.pos_index[term].get(doc_id, []))
                positions.sort()
                # 计算位置分散度（越集中分数越高）
                if len(positions) >= len(terms):
                    # 找到最小窗口
                    min_window = float('inf')
                    for i in range(len(positions) - len(terms) + 1):
                        window = positions[i + len(terms) - 1] - positions[i]
                        if window < min_window:
                            min_window = window
                    # 分数 = 1 - (窗口大小 / 文档长度)
                    score = 1.0 - min(min_window / max(len(self.token_lists[doc_id]), 1), 0.9)
                    results.append((self.article_ids[doc_id], 0.7 + score * 0.2))
                else:
                    results.append((self.article_ids[doc_id], 0.7))

            results.sort(key=lambda x: x[1], reverse=True)
            return results[:top_k]

        # 宽松匹配2：至少包含2个词
        if len(terms) >= 2:
            two_terms_docs = None
            for term in terms[:2]:
                if term in self.pos_index:
                    term_docs = set(self.pos_index[term].keys())
                    if two_terms_docs is None:
                        two_terms_docs = term_docs
                    else:
                        two_terms_docs &= term_docs

            if two_terms_docs:
                results = [(self.article_ids[doc_id], 0.5) for doc_id in two_terms_docs]
                return results[:top_k]

        return []