"""Web展示界面 - Flask应用（与api_server.py功能一致）"""
import sys
from pathlib import Path
import json
from datetime import datetime
import random

import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify

# 添加项目路径
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from config import RANDOM_SEED, TOP_K_RETRIEVAL, TOP_K_RECOMMEND
from src.data_loader import build_user_item_matrix, load_news_data, load_user_data
from src.preprocessing import build_doc_term_matrix, build_tfidf_matrix, preprocess_corpus
from src.retrieval.tfidf_retrieval import TFIDFRetriever
from src.retrieval.bm25_retrieval import BM25Retriever
from src.retrieval.boolean_retrieval import BooleanRetriever
from src.retrieval.phrase_retrieval import PhraseRetriever
from src.retrieval.semantic_retrieval import SemanticRetriever
from src.recommendation.collaborative_filtering import CollaborativeFiltering
from src.recommendation.content_based import ContentBasedRecommender
from src.recommendation.hybrid_recommendation import HybridRecommender
from src.fusion.hybrid_system import HybridSystem

app = Flask(__name__, template_folder='templates', static_folder='static')
app.config['SECRET_KEY'] = 'your-secret-key'

# ==================== 全局数据结构 ====================
news_df = None
user_df = None
retrievers = {}
hybrid_sys = None
article_info = {}
user_list = []

# 互动数据存储
USER_FAVORITES = {}  # {user_id: [article_ids]}
USER_LIKES = {}  # {user_id: [article_ids]}
USER_FOLLOWS = {}  # {user_id: [author_names]}
USER_BLOCKED = {}  # {user_id: [article_ids]}
USER_HISTORY = {}  # {user_id: [article_ids]}
COMMENTS = {}  # {article_id: [{'user_id':xxx, 'content':xxx, 'time':xxx}]}
SHARES = {}  # {article_id: count}
REPORTS = {}  # {article_id: count}


def init_user_data():
    """初始化用户互动数据"""
    global USER_FAVORITES, USER_LIKES, USER_FOLLOWS, USER_BLOCKED, USER_HISTORY, COMMENTS, SHARES, REPORTS

    for i in range(1, 501):
        user_id = f"U{str(i).zfill(5)}"
        USER_FAVORITES[user_id] = []
        USER_LIKES[user_id] = []
        USER_FOLLOWS[user_id] = []
        USER_BLOCKED[user_id] = []
        USER_HISTORY[user_id] = []

    COMMENTS = {}
    SHARES = {}
    REPORTS = {}


def init_system():
    """初始化系统，加载模型"""
    global news_df, user_df, retrievers, hybrid_sys, article_info, user_list

    print(">>> 初始化Web系统...")

    # 加载数据
    news_df = load_news_data()
    user_df = load_user_data()
    news_df = news_df[news_df['title'].str.len() > 0]

    # 构建文章信息索引
    for _, row in news_df.iterrows():
        # 处理关键词
        keywords = row.get('keywords', [])
        if isinstance(keywords, str):
            keywords = [k.strip() for k in keywords.split(',') if k.strip()]

        # 处理二级分类
        subcategories = row.get('subcategories', [])
        if isinstance(subcategories, str):
            subcategories = [s.strip() for s in subcategories.split(',') if s.strip()]

        article_info[row['article_id']] = {
            'title': row['title'],
            'category': row.get('category', '未分类'),
            'subcategories': subcategories[:3],
            'keywords': keywords[:5],
            'author': row.get('author', '未知作者'),
            'publish_date': row.get('publish_date', ''),
            'views': int(row['views']) if not pd.isna(row.get('views', 0)) else random.randint(100, 10000),
            'likes': int(row['likes']) if not pd.isna(row.get('likes', 0)) else random.randint(0, 500),
            'shares': 0
        }

    # 获取用户列表
    user_list = user_df['user_id'].tolist()
    print(f"    用户列表: {len(user_list)}人")

    # 预处理
    tokenized = preprocess_corpus(news_df)
    article_ids = news_df["article_id"].tolist()

    # 构建向量空间
    tfidf_vec, tfidf_mat = build_tfidf_matrix(tokenized)
    count_vec, dtm = build_doc_term_matrix(tokenized)

    # 构建检索系统
    retrievers = {
        "tfidf": TFIDFRetriever(tfidf_vec, tfidf_mat, article_ids),
        "bm25": BM25Retriever(count_vec, dtm, article_ids),
        "boolean": BooleanRetriever(count_vec, dtm, article_ids),
        "phrase": PhraseRetriever(tokenized, article_ids),
        "semantic": SemanticRetriever(tokenized, article_ids),
    }

    # 构建推荐系统
    interaction_df, _ = build_user_item_matrix(news_df, user_df)
    cb = ContentBasedRecommender(news_df, user_df)

    if len(interaction_df) >= 100:
        user_counts = interaction_df.groupby('user_id').size()
        valid_users = user_counts[user_counts >= 15].index.tolist()
        filtered_df = interaction_df[interaction_df['user_id'].isin(valid_users[:50])]
        train_mat = filtered_df.pivot_table(index="user_id", columns="article_id", values="rating", fill_value=0)
        if not train_mat.empty and len(train_mat.index) > 0:
            cf = CollaborativeFiltering(train_mat)
            hybrid_rec = HybridRecommender(cf, cb)
            hybrid_sys = HybridSystem(retrievers["semantic"], hybrid_rec, news_df)

    # 初始化互动数据
    init_user_data()

    print(f">>> 系统初始化完成！用户数: {len(user_list)}, 文章数: {len(article_info)}")


@app.route('/')
def index():
    """首页"""
    return render_template('index.html', users=user_list[:20])


# ==================== 检索相关接口 ====================

@app.route('/api/search', methods=['POST'])
def search():
    """检索接口"""
    data = request.json
    query = data.get('query', '')
    method = data.get('method', 'semantic')
    top_k = data.get('top_k', 10)

    if not query:
        return jsonify({'error': '请输入查询内容'}), 400

    retriever = retrievers.get(method)
    if not retriever:
        return jsonify({'error': '不支持的检索方法'}), 400

    results = retriever.search(query, top_k)

    response = []
    for aid, score in results:
        info = article_info.get(aid, {})
        response.append({
            'id': aid,
            'title': info.get('title', ''),
            'category': info.get('category', ''),
            'subcategories': info.get('subcategories', []),
            'keywords': info.get('keywords', []),
            'author': info.get('author', ''),
            'publish_date': info.get('publish_date', ''),
            'score': round(score, 4),
            'views': info.get('views', 0),
            'likes': info.get('likes', 0),
            'shares': SHARES.get(aid, 0)
        })

    return jsonify({'results': response, 'query': query, 'method': method})


@app.route('/api/recommend', methods=['POST'])
def recommend():
    """推荐接口"""
    data = request.json
    user_id = data.get('user_id', '')
    top_k = data.get('top_k', 10)

    if not user_id:
        return jsonify({'error': '请选择用户'}), 400

    if hybrid_sys is None:
        return jsonify({'error': '推荐系统未初始化'}), 400

    try:
        recommendations = hybrid_sys.recommender.switch_hybrid(user_id, top_k)

        # 获取浏览历史和屏蔽列表
        history = USER_HISTORY.get(user_id, [])
        blocked = USER_BLOCKED.get(user_id, [])

        response = []
        for aid, score in recommendations:
            if aid in blocked:
                continue
            info = article_info.get(aid, {})
            response.append({
                'id': aid,
                'title': info.get('title', ''),
                'category': info.get('category', ''),
                'subcategories': info.get('subcategories', []),
                'keywords': info.get('keywords', []),
                'author': info.get('author', ''),
                'publish_date': info.get('publish_date', ''),
                'score': round(score, 4),
                'views': info.get('views', 0),
                'likes': info.get('likes', 0),
                'shares': SHARES.get(aid, 0),
                'is_new': aid not in history
            })

        response.sort(key=lambda x: (x['is_new'], x['score']), reverse=True)
        return jsonify({'results': response[:top_k], 'user_id': user_id})
    except Exception as e:
        return jsonify({'error': f'推荐失败: {str(e)}'}), 500


@app.route('/api/fusion', methods=['POST'])
def fusion():
    """融合检索与推荐接口"""
    data = request.json
    query = data.get('query', '')
    user_id = data.get('user_id', '')
    top_k = data.get('top_k', 10)

    if not query:
        return jsonify({'error': '请输入查询内容'}), 400
    if not user_id:
        return jsonify({'error': '请选择用户'}), 400

    if hybrid_sys is None:
        return jsonify({'error': '融合系统未初始化'}), 400

    try:
        results = hybrid_sys.search_and_recommend(query, user_id, top_k)

        response = []
        for aid, score, title in results:
            info = article_info.get(aid, {})
            response.append({
                'id': aid,
                'title': title or info.get('title', ''),
                'category': info.get('category', ''),
                'subcategories': info.get('subcategories', []),
                'keywords': info.get('keywords', []),
                'author': info.get('author', ''),
                'publish_date': info.get('publish_date', ''),
                'score': round(score, 4),
                'views': info.get('views', 0),
                'likes': info.get('likes', 0),
                'shares': SHARES.get(aid, 0)
            })

        return jsonify({'results': response, 'query': query, 'user_id': user_id})
    except Exception as e:
        return jsonify({'error': f'融合检索失败: {str(e)}'}), 500


# ==================== 用户相关接口 ====================

@app.route('/api/users', methods=['GET'])
def get_users():
    """获取用户列表（支持搜索和分页）"""
    search = request.args.get('search', '')
    page = request.args.get('page', 1, type=int)
    per_page = 20

    all_users = user_list

    if search:
        all_users = [u for u in all_users if search.upper() in u]

    start = (page - 1) * per_page
    users = all_users[start:start + per_page]

    return jsonify({
        'users': users,
        'total': len(all_users),
        'page': page,
        'has_more': start + per_page < len(all_users)
    })


@app.route('/api/user_info', methods=['GET'])
def user_info():
    """获取用户信息"""
    user_id = request.args.get('user_id', '')
    if not user_id or user_id not in user_df['user_id'].values:
        return jsonify({'error': '用户不存在'}), 400

    user_data = user_df[user_df['user_id'] == user_id].iloc[0]
    preferences = {}
    for col in user_df.columns:
        if col.startswith('pref_'):
            pref_name = col.replace('pref_', '')
            preferences[pref_name] = float(user_data[col]) if not pd.isna(user_data[col]) else 0.0

    return jsonify({
        'user_id': user_id,
        'username': user_data.get('username', user_id),
        'activity_level': int(user_data.get('activity_level', 1)) if not pd.isna(
            user_data.get('activity_level')) else 1,
        'preferences': preferences,
        'favorites_count': len(USER_FAVORITES.get(user_id, [])),
        'likes_count': len(USER_LIKES.get(user_id, [])),
        'follows_count': len(USER_FOLLOWS.get(user_id, [])),
        'history_count': len(USER_HISTORY.get(user_id, []))
    })


# ==================== 新闻详情及互动接口 ====================

@app.route('/api/news_detail', methods=['GET'])
def news_detail():
    """新闻详情接口"""
    article_id = request.args.get('id', '')
    user_id = request.args.get('user_id', '')

    info = article_info.get(article_id)
    if not info:
        return jsonify({'error': '未找到该文章'}), 404

    # 增加浏览数
    info['views'] = info.get('views', 0) + 1

    # 记录浏览历史
    if user_id in USER_HISTORY:
        if article_id not in USER_HISTORY[user_id]:
            USER_HISTORY[user_id].insert(0, article_id)
            USER_HISTORY[user_id] = USER_HISTORY[user_id][:50]

    # 获取内容（从news_df中查找）
    content = ""
    if news_df is not None:
        row = news_df[news_df['article_id'] == article_id]
        if not row.empty and 'content' in row.columns:
            content = row.iloc[0].get('content', '')
            if not content or len(content) < 20:
                content = f"【{info.get('category', '新闻')}】{info.get('title', '')}\n\n这是一篇关于{info.get('category', '')}领域的重要报道。\n\n关键词：{', '.join(info.get('keywords', [])[:3])}"

    return jsonify({
        'id': article_id,
        'title': info.get('title', ''),
        'category': info.get('category', ''),
        'subcategories': info.get('subcategories', []),
        'keywords': info.get('keywords', []),
        'author': info.get('author', ''),
        'publish_date': info.get('publish_date', ''),
        'content': content[:800] if content else '暂无详细内容',
        'views': info.get('views', 0),
        'likes': info.get('likes', 0),
        'shares': SHARES.get(article_id, 0),
        'is_favorited': article_id in USER_FAVORITES.get(user_id, []),
        'is_liked': article_id in USER_LIKES.get(user_id, []),
        'is_following_author': info.get('author') in USER_FOLLOWS.get(user_id, []),
        'comments': COMMENTS.get(article_id, [])[-10:]
    })


@app.route('/api/favorite', methods=['POST'])
def favorite():
    """收藏/取消收藏接口"""
    data = request.json
    user_id = data.get('user_id', '')
    article_id = data.get('article_id', '')
    action = data.get('action', 'add')

    if action == 'add':
        if article_id not in USER_FAVORITES.get(user_id, []):
            USER_FAVORITES.setdefault(user_id, []).append(article_id)
    else:
        if article_id in USER_FAVORITES.get(user_id, []):
            USER_FAVORITES[user_id].remove(article_id)

    return jsonify({'success': True, 'count': len(USER_FAVORITES.get(user_id, []))})


@app.route('/api/like', methods=['POST'])
def like():
    """点赞/取消点赞接口"""
    data = request.json
    user_id = data.get('user_id', '')
    article_id = data.get('article_id', '')
    action = data.get('action', 'add')

    if action == 'add':
        if article_id not in USER_LIKES.get(user_id, []):
            USER_LIKES.setdefault(user_id, []).append(article_id)
            # 增加文章点赞数
            if article_id in article_info:
                article_info[article_id]['likes'] = article_info[article_id].get('likes', 0) + 1
    else:
        if article_id in USER_LIKES.get(user_id, []):
            USER_LIKES[user_id].remove(article_id)
            if article_id in article_info:
                article_info[article_id]['likes'] = max(0, article_info[article_id].get('likes', 0) - 1)

    return jsonify({'success': True, 'count': len(USER_LIKES.get(user_id, []))})


@app.route('/api/follow', methods=['POST'])
def follow():
    """关注/取消关注作者接口"""
    data = request.json
    user_id = data.get('user_id', '')
    author = data.get('author', '')
    action = data.get('action', 'add')

    if action == 'add':
        if author not in USER_FOLLOWS.get(user_id, []):
            USER_FOLLOWS.setdefault(user_id, []).append(author)
    else:
        if author in USER_FOLLOWS.get(user_id, []):
            USER_FOLLOWS[user_id].remove(author)

    return jsonify({'success': True, 'count': len(USER_FOLLOWS.get(user_id, []))})


@app.route('/api/block', methods=['POST'])
def block():
    """屏蔽文章接口"""
    data = request.json
    user_id = data.get('user_id', '')
    article_id = data.get('article_id', '')
    action = data.get('action', 'add')

    if action == 'add':
        if article_id not in USER_BLOCKED.get(user_id, []):
            USER_BLOCKED.setdefault(user_id, []).append(article_id)
    else:
        if article_id in USER_BLOCKED.get(user_id, []):
            USER_BLOCKED[user_id].remove(article_id)

    return jsonify({'success': True, 'count': len(USER_BLOCKED.get(user_id, []))})


@app.route('/api/share', methods=['POST'])
def share():
    """分享接口"""
    data = request.json
    article_id = data.get('article_id', '')
    SHARES[article_id] = SHARES.get(article_id, 0) + 1
    if article_id in article_info:
        article_info[article_id]['shares'] = SHARES[article_id]
    return jsonify({'success': True, 'shares': SHARES[article_id]})


@app.route('/api/comment', methods=['POST'])
def add_comment():
    """评论接口"""
    data = request.json
    user_id = data.get('user_id', '')
    article_id = data.get('article_id', '')
    content = data.get('content', '').strip()

    if not content:
        return jsonify({'error': '评论内容不能为空'}), 400

    comment = {
        'user_id': user_id,
        'content': content[:200],
        'time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }

    COMMENTS.setdefault(article_id, []).append(comment)

    return jsonify({'success': True, 'comment': comment})


@app.route('/api/report', methods=['POST'])
def report():
    """举报接口"""
    data = request.json
    article_id = data.get('article_id', '')
    reason = data.get('reason', '')

    REPORTS[article_id] = REPORTS.get(article_id, 0) + 1

    return jsonify({'success': True, 'message': '举报已提交'})


@app.route('/api/related', methods=['GET'])
def related():
    """相关推荐接口（基于分类和关键词）"""
    article_id = request.args.get('id', '')
    limit = request.args.get('limit', 5, type=int)

    current = article_info.get(article_id)
    if not current:
        return jsonify({'results': []})

    results = []
    for aid, info in article_info.items():
        if aid == article_id:
            continue

        score = 0
        if info.get('category') == current.get('category'):
            score += 0.5
        for sub in info.get('subcategories', []):
            if sub in current.get('subcategories', []):
                score += 0.3
        for kw in info.get('keywords', []):
            if kw in current.get('keywords', []):
                score += 0.2

        if score > 0:
            results.append({
                'id': aid,
                'title': info.get('title', ''),
                'category': info.get('category', ''),
                'score': min(0.95, score)
            })

    results.sort(key=lambda x: x['score'], reverse=True)
    return jsonify({'results': results[:limit]})


@app.route('/api/hot', methods=['GET'])
def hot():
    """热门文章接口"""
    limit = request.args.get('limit', 10, type=int)

    sorted_articles = sorted(
        article_info.items(),
        key=lambda x: (
                x[1].get('views', 0) * 0.4 +
                x[1].get('likes', 0) * 0.4 +
                SHARES.get(x[0], 0) * 0.2
        ),
        reverse=True
    )

    results = [{
        'id': aid,
        'title': info.get('title', ''),
        'category': info.get('category', ''),
        'views': info.get('views', 0),
        'likes': info.get('likes', 0),
        'shares': SHARES.get(aid, 0)
    } for aid, info in sorted_articles[:limit]]

    return jsonify({'results': results})


@app.route('/api/stats', methods=['GET'])
def stats():
    """获取系统统计信息"""
    total_views = sum(info.get('views', 0) for info in article_info.values())
    total_likes = sum(info.get('likes', 0) for info in article_info.values())
    total_shares = sum(SHARES.values())

    categories = {}
    for info in article_info.values():
        cat = info.get('category', '未分类')
        categories[cat] = categories.get(cat, 0) + 1

    return jsonify({
        'total_news': len(article_info),
        'total_users': len(user_list),
        'total_views': total_views,
        'total_likes': total_likes,
        'total_shares': total_shares,
        'categories': categories,
        'retrieval_methods': list(retrievers.keys())
    })


if __name__ == '__main__':
    init_system()
    print("\n" + "=" * 50)
    print("🚀 Web服务器已启动！")
    print("📱 请在浏览器中打开: http://localhost:5000")
    print("=" * 50 + "\n")
    app.run(debug=True, host='0.0.0.0', port=5000)