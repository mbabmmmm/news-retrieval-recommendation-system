"""布尔检索：AND / OR / NOT（增强版）。"""
from typing import List, Set, Tuple
from src.preprocessing import tokenize


class BooleanRetriever:
    def __init__(self, vectorizer, dtm, article_ids: List[str]):
        self.vectorizer = vectorizer
        self.dtm = dtm
        self.article_ids = article_ids
        self.index = {}
        for term, tid in vectorizer.vocabulary_.items():
            docs = set(dtm[:, tid].nonzero()[0].tolist())
            if docs:
                self.index[term] = docs

    def search(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """
        增强布尔检索，支持复杂表达式
        """
        query = query.strip()
        tokens = tokenize(query)

        if not tokens:
            return []

        # 简单查询（无布尔操作符）
        if " and " not in query.lower() and " or " not in query.lower() and " not " not in query.lower():
            docs = set()
            for term in tokens:
                docs.update(self.index.get(term, set()))
            # 按文档ID排序保证结果稳定
            results = [(self.article_ids[i], 1.0) for i in sorted(docs)]
            return results[:top_k]

        # 处理 NOT
        if " not " in query.lower():
            parts = query.lower().split(" not ")
            left_parts = parts[0].strip()
            right_parts = parts[1].strip() if len(parts) > 1 else ""

            left_terms = tokenize(left_parts)
            right_terms = tokenize(right_parts)

            left_docs = set()
            for term in left_terms:
                left_docs.update(self.index.get(term, set()))

            right_docs = set()
            for term in right_terms:
                right_docs.update(self.index.get(term, set()))

            docs = left_docs - right_docs
            results = [(self.article_ids[i], 1.0) for i in sorted(docs)]
            return results[:top_k]

        # 处理 AND 和 OR（支持复杂嵌套）
        # 优先处理 AND
        if " and " in query.lower():
            # 分割 AND 表达式
            and_parts = []
            current = query.lower()
            while " and " in current:
                idx = current.find(" and ")
                and_parts.append(current[:idx])
                current = current[idx + 5:]
            and_parts.append(current)

            # 处理每个部分中的 OR
            all_and_sets = []
            for part in and_parts:
                if " or " in part:
                    or_parts = part.split(" or ")
                    or_docs = set()
                    for or_part in or_parts:
                        or_terms = tokenize(or_part.strip())
                        for term in or_terms:
                            or_docs.update(self.index.get(term, set()))
                    all_and_sets.append(or_docs)
                else:
                    part_terms = tokenize(part.strip())
                    part_docs = set()
                    for term in part_terms:
                        part_docs.update(self.index.get(term, set()))
                    all_and_sets.append(part_docs)

            # 取交集
            if all_and_sets:
                docs = all_and_sets[0].copy()
                for ds in all_and_sets[1:]:
                    docs &= ds
            else:
                docs = set()

        # 处理 OR（无AND的情况）
        elif " or " in query.lower():
            or_parts = query.lower().split(" or ")
            docs = set()
            for part in or_parts:
                part_terms = tokenize(part.strip())
                for term in part_terms:
                    docs.update(self.index.get(term, set()))
        else:
            docs = set()

        results = [(self.article_ids[i], 1.0) for i in sorted(docs)]
        return results[:top_k]