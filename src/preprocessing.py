"""Text preprocessing: tokenization, stopwords, matrices."""
import re
from typing import List

import jieba
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from config import STOPWORDS


def tokenize(text: str, min_word_len: int = 1) -> List[str]:
    """Chinese tokenization with stopword removal."""
    if not isinstance(text, str) or not text.strip():
        return []
    tokens = jieba.lcut(text)
    cleaned = []
    for token in tokens:
        token = token.strip()
        if len(token) < min_word_len:
            continue
        if token in STOPWORDS:
            continue
        if re.match(r'^[\W_]+$', token) and not re.match(r'[\u4e00-\u9fff]', token):
            continue
        cleaned.append(token)
    return cleaned


def preprocess_corpus(news_df, use_full_text: bool = True) -> List[str]:
    """Tokenize news corpus with enhanced text representation."""
    tokenized_docs = []

    for idx, row in news_df.iterrows():
        parts = []
        # 标题权重最高（重复3次）
        if row.get('title') and str(row['title']).strip():
            parts.append(str(row['title']) * 3)
        # 类别权重高（重复2次）
        if row.get('category') and str(row['category']).strip():
            parts.append(str(row['category']) * 2)
        # 关键词
        if row.get('keywords') and isinstance(row['keywords'], list):
            parts.append(' '.join(str(kw) for kw in row['keywords']))
        # 内容前300字
        if use_full_text and row.get('content') and str(row['content']).strip():
            parts.append(str(row['content'])[:300])

        text = ' '.join(parts)
        tokens = tokenize(text)

        if not tokens:
            tokens = [c for c in text if c.strip() and c not in STOPWORDS and len(c.strip()) >= 1]
        tokenized_docs.append(" ".join(tokens))

    empty_count = sum(1 for doc in tokenized_docs if not doc.strip())
    if empty_count > 0:
        print(f"    警告: {empty_count}/{len(tokenized_docs)} 篇文档分词后为空")

    return tokenized_docs


def build_tfidf_matrix(tokenized_docs: List[str]):
    """Build TF-IDF document-term matrix with n-grams."""
    vectorizer = TfidfVectorizer(
        token_pattern=r'(?u)\b\w+\b',
        min_df=2,
        max_df=0.9,
        ngram_range=(1, 2),
        sublinear_tf=True,
    )
    tfidf_matrix = vectorizer.fit_transform(tokenized_docs)
    print(f"    TF-IDF词汇表大小: {len(vectorizer.vocabulary_)}")
    return vectorizer, tfidf_matrix


def build_doc_term_matrix(tokenized_docs: List[str]):
    """Build count document-term matrix for BM25/boolean search."""
    vectorizer = CountVectorizer(
        token_pattern=r'(?u)\b\w+\b',
        min_df=2,
        max_df=0.9,
        ngram_range=(1, 2),
    )
    dtm = vectorizer.fit_transform(tokenized_docs)
    print(f"    CountVectorizer词汇表大小: {len(vectorizer.vocabulary_)}")
    return vectorizer, dtm