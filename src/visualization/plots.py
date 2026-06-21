"""Matplotlib visualization for evaluation and recommendations."""
from typing import Dict, List

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from config import OUTPUT_DIR

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False


def plot_retrieval_comparison(metrics_dict: Dict[str, Dict[str, float]], save_name: str = "retrieval_metrics.png"):
    df = pd.DataFrame(metrics_dict).T
    # 选择要显示的指标
    display_metrics = ['precision', 'recall', 'f1', 'ndcg']
    df_display = df[display_metrics]

    fig, ax = plt.subplots(figsize=(12, 6))
    df_display.plot(kind="bar", ax=ax, rot=45, width=0.7)
    ax.set_title("检索系统评估指标对比", fontsize=14)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_xlabel("检索算法", fontsize=12)
    ax.legend(loc="upper right", title="指标")
    ax.set_ylim(0, 1.1)
    ax.grid(axis='y', alpha=0.3)

    # 添加数值标签
    for container in ax.containers:
        ax.bar_label(container, fmt='%.3f', fontsize=8)

    plt.tight_layout()
    path = OUTPUT_DIR / save_name
    plt.savefig(path, dpi=150)
    plt.close()
    return path


def plot_recommendation_comparison(metrics_dict: Dict[str, Dict[str, float]],
                                   save_name: str = "recommendation_metrics.png"):
    df = pd.DataFrame(metrics_dict).T
    display_metrics = ['precision', 'recall', 'hit_ratio']
    df_display = df[display_metrics]

    fig, ax = plt.subplots(figsize=(10, 6))
    df_display.plot(kind="bar", ax=ax, rot=0, width=0.6)
    ax.set_title("推荐系统评估指标对比", fontsize=14)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_xlabel("推荐算法", fontsize=12)
    ax.legend(loc="upper right", title="指标")
    ax.set_ylim(0, 1.1)
    ax.grid(axis='y', alpha=0.3)

    for container in ax.containers:
        ax.bar_label(container, fmt='%.3f', fontsize=9)

    plt.tight_layout()
    path = OUTPUT_DIR / save_name
    plt.savefig(path, dpi=150)
    plt.close()
    return path


def plot_user_recommendations(user_id: str, items: List[tuple], save_name: str = None):
    """Visualize personalized recommendations for one user."""
    if save_name is None:
        save_name = f"user_{user_id}_recommendations.png"

    titles = []
    scores = []
    for _, score, title in items[:10]:
        # 截取标题长度
        short_title = title[:25] + "..." if len(title) > 25 else title
        titles.append(short_title)
        scores.append(score)

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(range(len(titles)), scores, color='steelblue')
    ax.set_yticks(range(len(titles)))
    ax.set_yticklabels(titles)
    ax.set_xlabel("融合得分", fontsize=12)
    ax.set_title(f"用户 {user_id} 个性化推荐结果 (Top-10)", fontsize=14)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.1)

    # 添加数值标签
    for i, bar in enumerate(bars):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                f'{scores[i]:.3f}', va='center', fontsize=9)

    plt.tight_layout()
    path = OUTPUT_DIR / save_name
    plt.savefig(path, dpi=150)
    plt.close()
    return path


def plot_data_analysis(news_df, user_df, save_name: str = "data_analysis.png"):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 新闻类别分布
    cat_counts = news_df["category"].value_counts()
    axes[0, 0].bar(range(len(cat_counts)), cat_counts.values, color='coral')
    axes[0, 0].set_title("新闻类别分布", fontsize=12)
    axes[0, 0].set_xticks(range(len(cat_counts)))
    axes[0, 0].set_xticklabels(cat_counts.index, rotation=45, ha='right')
    axes[0, 0].set_ylabel("新闻数量")

    # 用户平均类别偏好
    pref_cols = [c for c in user_df.columns if c.startswith("pref_")]
    if pref_cols:
        avg_prefs = user_df[pref_cols].mean()
        avg_prefs.index = [c.replace("pref_", "") for c in pref_cols]
        axes[0, 1].bar(range(len(avg_prefs)), avg_prefs.values, color='seagreen')
        axes[0, 1].set_title("用户平均类别偏好", fontsize=12)
        axes[0, 1].set_xticks(range(len(avg_prefs)))
        axes[0, 1].set_xticklabels(avg_prefs.index, rotation=45, ha='right')
        axes[0, 1].set_ylabel("平均偏好分数")

    # 活动等级分布
    if 'activity_level' in user_df.columns:
        activity_counts = user_df['activity_level'].value_counts().sort_index()
        axes[1, 0].bar(activity_counts.index, activity_counts.values, color='cornflowerblue')
        axes[1, 0].set_title("用户活跃等级分布", fontsize=12)
        axes[1, 0].set_xlabel("活跃等级")
        axes[1, 0].set_ylabel("用户数量")

    # 新闻浏览量分布（取对数）
    if 'views' in news_df.columns:
        views_log = np.log10(news_df['views'] + 1)
        axes[1, 1].hist(views_log, bins=30, color='purple', alpha=0.7)
        axes[1, 1].set_title("新闻浏览量分布 (log10)", fontsize=12)
        axes[1, 1].set_xlabel("log10(浏览量+1)")
        axes[1, 1].set_ylabel("新闻数量")

    plt.tight_layout()
    path = OUTPUT_DIR / save_name
    plt.savefig(path, dpi=150)
    plt.close()
    return path