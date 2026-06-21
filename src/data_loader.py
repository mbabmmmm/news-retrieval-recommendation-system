"""Data loading and preprocessing."""
import json
from typing import Dict, Tuple

import pandas as pd

from config import NEWS_DATA_PATH, USER_DATA_PATH


def load_news_data(news_path=NEWS_DATA_PATH) -> pd.DataFrame:
    """Load JSONL news data and merge metadata with content."""
    articles: Dict[str, dict] = {}
    contents: Dict[str, str] = {}

    with open(news_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            record_type = record.get("type")
            article_id = record.get("article_id")

            if record_type == "article":
                articles[article_id] = record
            elif record_type == "article_content":
                contents[article_id] = record.get("content", "")

    rows = []
    for article_id, meta in articles.items():
        # 确保keywords是列表
        keywords = meta.get("keywords", [])
        if isinstance(keywords, str):
            keywords = [k.strip() for k in keywords.split(',') if k.strip()]

        # 确保subcategories是列表
        subcategories = meta.get("subcategories", [])
        if isinstance(subcategories, str):
            subcategories = [s.strip() for s in subcategories.split(',') if s.strip()]

        content = contents.get(article_id, "")

        rows.append(
            {
                "article_id": article_id,
                "title": meta.get("title", ""),
                "category": meta.get("category", ""),
                "subcategories": subcategories,
                "keywords": keywords,
                "publish_date": meta.get("publish_date", ""),
                "author": meta.get("author", ""),
                "views": meta.get("views", 0),
                "likes": meta.get("likes", 0),
                "content": content,
            }
        )

    df = pd.DataFrame(rows)

    # 构建full_text，确保有足够内容
    df["full_text"] = (
            df["title"].fillna("") + " " +
            df["category"].fillna("") + " " +
            df["keywords"].apply(lambda x: " ".join(x) if isinstance(x, list) and x else "") + " " +
            df["content"].fillna("") + " " +
            df["subcategories"].apply(lambda x: " ".join(x) if isinstance(x, list) and x else "")
    )

    # 调试信息
    print(f"    加载新闻数量: {len(df)}")
    print(f"    有内容的新闻: {(df['content'] != '').sum()}")
    print(f"    有标题的新闻: {(df['title'] != '').sum()}")
    print(f"    有关键词的新闻: {(df['keywords'].apply(lambda x: len(x) > 0 if isinstance(x, list) else False)).sum()}")

    return df


def load_user_data(user_path=USER_DATA_PATH) -> pd.DataFrame:
    """Load user preference JSON data."""
    with open(user_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    records = []

    # 处理不同的JSON格式
    if isinstance(data, dict):
        # 如果data是字典，尝试多种可能的键名
        users = data.get("users", [])
        if not users:
            # 可能用户数据直接在最外层
            for key, value in data.items():
                if isinstance(value, dict) and "category_preferences" in value:
                    value["user_id"] = key
                    users.append(value)
    elif isinstance(data, list):
        users = data
    else:
        users = []

    for user in users:
        # 获取category_preferences
        prefs = user.get("category_preferences", {})
        if not prefs:
            # 尝试其他可能的键名
            for key in ["preferences", "prefs", "category_prefs"]:
                if key in user:
                    prefs = user[key]
                    break

        records.append(
            {
                "user_id": user.get("user_id"),
                "username": user.get("username", ""),
                "registration_date": user.get("registration_date", ""),
                "activity_level": user.get("activity_level", 1),
                **{f"pref_{k}": v for k, v in prefs.items() if isinstance(v, (int, float))},
            }
        )

    df = pd.DataFrame(records)
    print(f"    加载用户数量: {len(df)}")
    return df


def build_user_item_matrix(news_df: pd.DataFrame, user_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Build implicit user-item matrix from category preferences."""
    max_popularity = (news_df["views"] + news_df["likes"]).max()
    if max_popularity == 0:
        max_popularity = 1

    article_part = news_df[["article_id", "category", "views", "likes"]].copy()
    article_part["popularity"] = (article_part["views"] + article_part["likes"]) / max_popularity

    pref_cols = [c for c in user_df.columns if c.startswith("pref_")]
    user_part = user_df[["user_id"] + pref_cols].copy()

    interactions = []
    for cat in article_part["category"].unique():
        pref_col = f"pref_{cat}"
        if pref_col not in user_part.columns:
            continue
        sub_articles = article_part[article_part["category"] == cat]
        for _, user in user_part.iterrows():
            pref_score = user[pref_col]
            if pref_score <= 0.15:
                continue
            ratings = pref_score * (0.7 + 0.3 * sub_articles["popularity"].values)
            mask = ratings > 0.15
            if not mask.any():
                continue
            for aid, rating in zip(sub_articles.loc[mask, "article_id"], ratings[mask]):
                interactions.append(
                    {
                        "user_id": user["user_id"],
                        "article_id": aid,
                        "rating": round(float(rating), 4),
                        "category": cat,
                    }
                )

    interaction_df = pd.DataFrame(interactions)
    matrix = interaction_df.pivot_table(
        index="user_id",
        columns="article_id",
        values="rating",
        fill_value=0.0,
    )
    return interaction_df, matrix