from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import json
import os
from datetime import datetime
import random

app = Flask(__name__, template_folder='templates', static_folder='static')
CORS(app)

# ==================== 全局数据结构 ====================
NEWS_DATA = []  # 新闻列表
NEWS_CONTENT = {}  # 新闻内容映射
USER_FAVORITES = {}  # 用户收藏 {user_id: [article_ids]}
USER_LIKES = {}  # 用户点赞 {user_id: [article_ids]}
USER_FOLLOWS = {}  # 用户关注作者 {user_id: [author_names]}
USER_BLOCKED = {}  # 用户屏蔽 {user_id: [article_ids]}
USER_HISTORY = {}  # 用户浏览历史 {user_id: [article_ids]}
COMMENTS = {}  # 评论 {article_id: [{'user_id':xxx, 'content':xxx, 'time':xxx}]}
SHARES = {}  # 分享数 {article_id: count}
REPORTS = {}  # 举报数 {article_id: count}


# ==================== 数据加载 ====================
def load_news():
    """加载新闻数据"""
    global NEWS_DATA, NEWS_CONTENT
    NEWS_DATA = []
    NEWS_CONTENT = {}

    file_dir = os.path.dirname(__file__)
    news_path = os.path.join(file_dir, '新闻数据.txt')

    if not os.path.exists(news_path):
        print("错误：未找到新闻数据文件，使用模拟数据")
        generate_mock_data()
        return

    try:
        with open(news_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    if data.get('type') == 'article':
                        # 处理关键词
                        keywords = data.get('keywords', [])
                        if isinstance(keywords, str):
                            keywords = [k.strip() for k in keywords.split(',') if k.strip()]

                        # 处理二级分类
                        subcategories = data.get('subcategories', [])
                        if isinstance(subcategories, str):
                            subcategories = [s.strip() for s in subcategories.split(',') if s.strip()]

                        NEWS_DATA.append({
                            'id': data.get('article_id'),
                            'title': data.get('title', '无标题'),
                            'category': data.get('category', '未分类'),
                            'subcategories': subcategories[:3],
                            'keywords': keywords[:5],
                            'author': data.get('author', '未知作者'),
                            'publish_date': data.get('publish_date', ''),
                            'views': data.get('views', random.randint(100, 10000)),
                            'likes': data.get('likes', random.randint(0, 500)),
                            'shares': 0
                        })
                    elif data.get('type') == 'article_content':
                        content = data.get('content', '')
                        # 清理内容
                        import re
                        clean_content = re.sub(
                            r'[^\u4e00-\u9fa5\u3000-\u303f\uff00-\uffef\.\,\!\?\;\:\"\'\(\)\[\]\{\}\<\>\n]', '',
                            content)
                        if len(clean_content) < 50:
                            clean_content = f"【{data.get('title', '新闻')}】\n\n这是一篇关于此主题的详细报道。\n\n关键词：{', '.join(keywords[:3]) if 'keywords' in locals() else '相关'}。"
                        NEWS_CONTENT[data.get('article_id')] = clean_content[:800]
                except:
                    continue

        if len(NEWS_DATA) == 0:
            generate_mock_data()
        else:
            print(f"成功加载 {len(NEWS_DATA)} 条新闻")

    except Exception as e:
        print(f"加载失败: {e}")
        generate_mock_data()


def generate_mock_data():
    """生成模拟新闻数据"""
    global NEWS_DATA, NEWS_CONTENT

    categories_data = {
        '科技': {
            'subcategories': ['人工智能', '5G通信', '半导体', '互联网', '软件服务'],
            'keywords': ['机器学习', '算法', '数据', '创新', '技术突破'],
            'authors': ['张科技', '李创新', '王技术', '陈数据']
        },
        '健康': {
            'subcategories': ['养生保健', '疾病预防', '心理健康', '运动健身', '营养饮食'],
            'keywords': ['健康', '养生', '运动', '饮食', '保健'],
            'authors': ['李健康', '张医生', '王营养', '赵运动']
        },
        '汽车': {
            'subcategories': ['新能源汽车', '自动驾驶', '交通安全', '汽车评测', '行业资讯'],
            'keywords': ['新能源', '自动驾驶', '汽车', '电动', '充电'],
            'authors': ['王汽车', '李评测', '张驾驶', '陈电动']
        },
        '文化': {
            'subcategories': ['传统文化', '艺术展览', '文学出版', '博物馆', '非遗传承'],
            'keywords': ['文化', '艺术', '传统', '展览', '博物馆'],
            'authors': ['赵文化', '李艺术', '王传统', '张文学']
        },
        '国际': {
            'subcategories': ['国际关系', '全球经济', '地区冲突', '国际合作', '外交动态'],
            'keywords': ['国际', '全球', '外交', '合作', '经济'],
            'authors': ['陈国际', '王全球', '李外交', '张合作']
        },
        '体育': {
            'subcategories': ['足球', '篮球', '奥运', '电子竞技', '综合体育'],
            'keywords': ['比赛', '冠军', '运动', '赛事', '体育'],
            'authors': ['王体育', '李运动', '张比赛', '陈冠军']
        },
        '娱乐': {
            'subcategories': ['影视', '音乐', '综艺', '明星', '时尚'],
            'keywords': ['明星', '电影', '音乐', '综艺', '娱乐'],
            'authors': ['李娱乐', '王明星', '张影视', '赵音乐']
        },
        '财经': {
            'subcategories': ['股票', '基金', '房地产', '理财', '宏观经济'],
            'keywords': ['股票', '投资', '基金', '理财', '经济'],
            'authors': ['张财经', '李投资', '王股票', '陈理财']
        },
        '教育': {
            'subcategories': ['高等教育', '职业教育', 'K12教育', '在线教育', '留学'],
            'keywords': ['教育', '学习', '考试', '大学', '培训'],
            'authors': ['王教育', '李学习', '张考试', '赵大学']
        },
        '社会': {
            'subcategories': ['民生', '法治', '环境', '公益', '城市发展'],
            'keywords': ['社会', '民生', '公益', '法治', '环境'],
            'authors': ['张社会', '李民生', '王公益', '陈法治']
        }
    }

    NEWS_DATA = []
    NEWS_CONTENT = {}

    for i in range(1, 301):
        cat = list(categories_data.keys())[i % len(categories_data)]
        cat_data = categories_data[cat]
        subcat = cat_data['subcategories'][i % len(cat_data['subcategories'])]
        keywords = cat_data['keywords']
        author = cat_data['authors'][i % len(cat_data['authors'])]

        news_id = f"N{i:05d}"
        NEWS_DATA.append({
            'id': news_id,
            'title': f"{subcat}：{cat}领域重要新闻{i}",
            'category': cat,
            'subcategories': [subcat, cat_data['subcategories'][(i + 1) % len(cat_data['subcategories'])]],
            'keywords': [keywords[i % len(keywords)], keywords[(i + 1) % len(keywords)], '趋势', '发展'],
            'author': author,
            'publish_date': f"2025-{random.randint(1, 12):02d}-{random.randint(1, 28):02d} {random.randint(8, 22):02d}:{random.randint(0, 59):02d}:{random.randint(0, 59):02d}",
            'views': random.randint(100, 50000),
            'likes': random.randint(0, 2000),
            'shares': random.randint(0, 500)
        })

        NEWS_CONTENT[news_id] = f"""
【{subcat}】{cat}领域最新动态

近日，{author}在相关领域发表重要观点。本文深入分析了当前{cat}行业的发展趋势和未来方向。

主要内容包括：
1. 行业现状与挑战
2. 关键技术突破
3. 未来发展趋势预测
4. 专家观点与建议

相关关键词：{', '.join(keywords[:3])}

本文为原创内容，转载需授权。更多精彩内容，请持续关注。
"""

    print(f"生成 {len(NEWS_DATA)} 条模拟新闻数据")


# ==================== 用户数据初始化 ====================
def init_user_data():
    global USER_FAVORITES, USER_LIKES, USER_FOLLOWS, USER_BLOCKED, USER_HISTORY, COMMENTS, SHARES, REPORTS

    for i in range(1, 501):
        user_id = f"U{str(i).zfill(5)}"
        USER_FAVORITES[user_id] = []
        USER_LIKES[user_id] = []
        USER_FOLLOWS[user_id] = []
        USER_BLOCKED[user_id] = []
        USER_HISTORY[user_id] = []

    COMMENTS = {}
    SHARES = {news['id']: news.get('shares', 0) for news in NEWS_DATA}
    REPORTS = {}


load_news()
init_user_data()


# ==================== 路由接口 ====================

@app.route('/')
def index():
    return render_template('index.html')


# 检索接口
@app.route('/api/search', methods=['POST'])
def search():
    data = request.json
    query = data.get('query', '')
    top_k = data.get('top_k', 10)

    results = []
    for news in NEWS_DATA:
        score = 0
        title = news.get('title', '')
        category = news.get('category', '')
        keywords = news.get('keywords', [])
        subcategories = news.get('subcategories', [])

        if query in title:
            score += 0.8
        if query == category:
            score += 0.5
        if query in subcategories:
            score += 0.4
        for kw in keywords:
            if query in kw or kw in query:
                score += 0.3

        if score > 0:
            results.append({
                'id': news.get('id'),
                'title': title,
                'category': category,
                'subcategories': subcategories,
                'keywords': keywords,
                'author': news.get('author'),
                'publish_date': news.get('publish_date'),
                'score': min(0.95, score),
                'views': news.get('views', 0),
                'likes': news.get('likes', 0),
                'shares': SHARES.get(news.get('id'), 0)
            })

    results.sort(key=lambda x: x['score'], reverse=True)
    return jsonify({'results': results[:top_k], 'query': query})


# 推荐接口（基于用户偏好）
@app.route('/api/recommend', methods=['POST'])
def recommend():
    data = request.json
    user_id = data.get('user_id', '')
    top_k = data.get('top_k', 10)

    # 用户偏好映射
    user_prefs = {
        'U00001': {'categories': ['科技', '国际', '汽车'], 'keywords': ['AI', '技术', '创新']},
        'U00002': {'categories': ['娱乐', '体育', '文化'], 'keywords': ['明星', '电影', '比赛']},
        'U00003': {'categories': ['财经', '科技', '国际'], 'keywords': ['股票', '投资', '经济']},
        'U00004': {'categories': ['健康', '体育', '科技'], 'keywords': ['健康', '运动', '养生']},
        'U00005': {'categories': ['汽车', '科技', '财经'], 'keywords': ['新能源', '汽车', '充电']},
    }

    prefs = user_prefs.get(user_id, {'categories': ['科技', '健康', '汽车'], 'keywords': []})

    # 获取浏览历史和收藏，用于排除
    history = USER_HISTORY.get(user_id, [])
    favorites = USER_FAVORITES.get(user_id, [])
    blocked = USER_BLOCKED.get(user_id, [])

    results = []
    for news in NEWS_DATA:
        if news['id'] in blocked:
            continue

        score = 0.5
        if news.get('category') in prefs['categories']:
            score += 0.3
        for kw in news.get('keywords', []):
            if kw in prefs['keywords']:
                score += 0.2

        # 流行度加成
        score += min(news.get('likes', 0) / 5000, 0.15)

        results.append({
            'id': news.get('id'),
            'title': news.get('title'),
            'category': news.get('category'),
            'subcategories': news.get('subcategories'),
            'keywords': news.get('keywords'),
            'author': news.get('author'),
            'publish_date': news.get('publish_date'),
            'score': min(0.95, score),
            'views': news.get('views', 0),
            'likes': news.get('likes', 0),
            'shares': SHARES.get(news.get('id'), 0),
            'is_new': news['id'] not in history
        })

    results.sort(key=lambda x: (x['is_new'], x['score']), reverse=True)
    return jsonify({'results': results[:top_k], 'user_id': user_id})


# 融合检索接口
@app.route('/api/fusion', methods=['POST'])
def fusion():
    data = request.json
    query = data.get('query', '')
    user_id = data.get('user_id', '')
    top_k = data.get('top_k', 10)

    # 检索结果
    search_results = []
    for news in NEWS_DATA:
        score = 0
        if query in news.get('title', ''):
            score += 0.8
        if query == news.get('category', ''):
            score += 0.5
        if query in news.get('subcategories', []):
            score += 0.4
        if score > 0:
            search_results.append({
                'id': news.get('id'),
                'title': news.get('title'),
                'category': news.get('category'),
                'subcategories': news.get('subcategories'),
                'keywords': news.get('keywords'),
                'author': news.get('author'),
                'score': min(0.95, score),
                'views': news.get('views', 0),
                'likes': news.get('likes', 0)
            })
    search_results.sort(key=lambda x: x['score'], reverse=True)

    # 用户偏好
    user_prefs = {'U00001': ['科技', '国际', '汽车'], 'U00002': ['娱乐', '体育', '文化']}
    prefs = user_prefs.get(user_id, ['科技', '健康', '汽车'])

    rec_results = []
    for news in NEWS_DATA:
        if news.get('category') in prefs:
            rec_results.append({
                'id': news.get('id'),
                'title': news.get('title'),
                'category': news.get('category'),
                'subcategories': news.get('subcategories'),
                'keywords': news.get('keywords'),
                'author': news.get('author'),
                'score': 0.6,
                'views': news.get('views', 0),
                'likes': news.get('likes', 0)
            })
    rec_results.sort(key=lambda x: x['score'], reverse=True)

    # 融合
    fused = {}
    for r in search_results[:top_k]:
        fused[r['id']] = {**r, 'score': r['score'] * 0.6}
    for r in rec_results[:top_k]:
        if r['id'] in fused:
            fused[r['id']]['score'] += r['score'] * 0.4
            fused[r['id']]['score'] = min(0.98, fused[r['id']]['score'])
        else:
            fused[r['id']] = {**r, 'score': r['score'] * 0.4}

    results = sorted(fused.values(), key=lambda x: x['score'], reverse=True)[:top_k]
    return jsonify({'results': results, 'query': query, 'user_id': user_id})


# 用户列表接口
@app.route('/api/users', methods=['GET'])
def get_users():
    search = request.args.get('search', '')
    page = request.args.get('page', 1, type=int)
    per_page = 20

    all_users = [f"U{str(i).zfill(5)}" for i in range(1, 501)]

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


# 用户信息接口
@app.route('/api/user_info', methods=['GET'])
def get_user_info():
    user_id = request.args.get('user_id', '')

    user_names = {
        'U00001': '科技爱好者', 'U00002': '娱乐达人', 'U00003': '财经专家',
        'U00004': '健康生活家', 'U00005': '汽车迷', 'U00006': '国际视野者',
        'U00007': '文化爱好者', 'U00008': '体育迷', 'U00009': '教育工作者',
        'U00010': '社会观察者',
    }

    user_prefs = {
        'U00001': {'科技': 0.92, '国际': 0.85, '汽车': 0.78},
        'U00002': {'娱乐': 0.95, '体育': 0.88, '文化': 0.75},
        'U00003': {'财经': 0.96, '科技': 0.82, '国际': 0.78},
        'U00004': {'健康': 0.94, '体育': 0.85, '科技': 0.68},
        'U00005': {'汽车': 0.95, '科技': 0.80, '财经': 0.70},
    }

    return jsonify({
        'user_id': user_id,
        'username': user_names.get(user_id, user_id),
        'activity_level': 4,
        'preferences': user_prefs.get(user_id, {'科技': 0.5, '健康': 0.5}),
        'favorites_count': len(USER_FAVORITES.get(user_id, [])),
        'likes_count': len(USER_LIKES.get(user_id, [])),
        'follows_count': len(USER_FOLLOWS.get(user_id, [])),
        'history_count': len(USER_HISTORY.get(user_id, []))
    })


# 新闻详情接口
@app.route('/api/news_detail', methods=['GET'])
def news_detail():
    article_id = request.args.get('id', '')
    user_id = request.args.get('user_id', '')

    for news in NEWS_DATA:
        if news.get('id') == article_id:
            # 增加浏览数
            news['views'] = news.get('views', 0) + 1

            # 记录浏览历史
            if user_id in USER_HISTORY:
                if article_id not in USER_HISTORY[user_id]:
                    USER_HISTORY[user_id].insert(0, article_id)
                    USER_HISTORY[user_id] = USER_HISTORY[user_id][:50]

            content = NEWS_CONTENT.get(article_id, '暂无详细内容')

            return jsonify({
                'id': news.get('id'),
                'title': news.get('title'),
                'category': news.get('category'),
                'subcategories': news.get('subcategories'),
                'keywords': news.get('keywords'),
                'author': news.get('author'),
                'publish_date': news.get('publish_date'),
                'content': content,
                'views': news.get('views', 0),
                'likes': news.get('likes', 0),
                'shares': SHARES.get(article_id, 0),
                'is_favorited': article_id in USER_FAVORITES.get(user_id, []),
                'is_liked': article_id in USER_LIKES.get(user_id, []),
                'is_following_author': news.get('author') in USER_FOLLOWS.get(user_id, []),
                'comments': COMMENTS.get(article_id, [])[-10:]
            })

    return jsonify({'error': '未找到该文章'}), 404


# 收藏接口
@app.route('/api/favorite', methods=['POST'])
def favorite():
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


# 点赞接口
@app.route('/api/like', methods=['POST'])
def like():
    data = request.json
    user_id = data.get('user_id', '')
    article_id = data.get('article_id', '')
    action = data.get('action', 'add')

    if action == 'add':
        if article_id not in USER_LIKES.get(user_id, []):
            USER_LIKES.setdefault(user_id, []).append(article_id)
            # 增加文章点赞数
            for news in NEWS_DATA:
                if news['id'] == article_id:
                    news['likes'] = news.get('likes', 0) + 1
                    break
    else:
        if article_id in USER_LIKES.get(user_id, []):
            USER_LIKES[user_id].remove(article_id)
            for news in NEWS_DATA:
                if news['id'] == article_id:
                    news['likes'] = max(0, news.get('likes', 0) - 1)
                    break

    return jsonify({'success': True, 'count': len(USER_LIKES.get(user_id, []))})


# 关注作者接口
@app.route('/api/follow', methods=['POST'])
def follow():
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


# 屏蔽文章接口
@app.route('/api/block', methods=['POST'])
def block():
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


# 分享接口
@app.route('/api/share', methods=['POST'])
def share():
    data = request.json
    article_id = data.get('article_id', '')
    SHARES[article_id] = SHARES.get(article_id, 0) + 1
    return jsonify({'success': True, 'shares': SHARES[article_id]})


# 评论接口
@app.route('/api/comment', methods=['POST'])
def add_comment():
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


# 举报接口
@app.route('/api/report', methods=['POST'])
def report():
    data = request.json
    article_id = data.get('article_id', '')
    reason = data.get('reason', '')

    REPORTS[article_id] = REPORTS.get(article_id, 0) + 1

    return jsonify({'success': True, 'message': '举报已提交'})


# 获取推荐文章（基于分类）
@app.route('/api/related', methods=['GET'])
def related():
    article_id = request.args.get('id', '')
    limit = request.args.get('limit', 5, type=int)

    current = None
    for news in NEWS_DATA:
        if news['id'] == article_id:
            current = news
            break

    if not current:
        return jsonify({'results': []})

    results = []
    for news in NEWS_DATA:
        if news['id'] == article_id:
            continue

        score = 0
        if news['category'] == current['category']:
            score += 0.5
        for sub in news.get('subcategories', []):
            if sub in current.get('subcategories', []):
                score += 0.3
        for kw in news.get('keywords', []):
            if kw in current.get('keywords', []):
                score += 0.2

        if score > 0:
            results.append({
                'id': news['id'],
                'title': news['title'],
                'category': news['category'],
                'score': min(0.95, score)
            })

    results.sort(key=lambda x: x['score'], reverse=True)
    return jsonify({'results': results[:limit]})


# 统计接口
@app.route('/api/stats', methods=['GET'])
def stats():
    total_views = sum(news.get('views', 0) for news in NEWS_DATA)
    total_likes = sum(news.get('likes', 0) for news in NEWS_DATA)
    total_shares = sum(SHARES.values())

    return jsonify({
        'total_news': len(NEWS_DATA),
        'total_users': 500,
        'total_views': total_views,
        'total_likes': total_likes,
        'total_shares': total_shares,
        'categories': {}
    })


# 获取热门文章
@app.route('/api/hot', methods=['GET'])
def hot():
    limit = request.args.get('limit', 10, type=int)

    sorted_news = sorted(NEWS_DATA, key=lambda x: (
            x.get('views', 0) * 0.4 +
            x.get('likes', 0) * 0.4 +
            SHARES.get(x['id'], 0) * 0.2
    ), reverse=True)

    results = [{
        'id': n['id'],
        'title': n['title'],
        'category': n['category'],
        'views': n.get('views', 0),
        'likes': n.get('likes', 0),
        'shares': SHARES.get(n['id'], 0)
    } for n in sorted_news[:limit]]

    return jsonify({'results': results})


if __name__ == '__main__':
    print(f"初始化完成，共加载 {len(NEWS_DATA)} 条新闻，{500} 个用户")
    app.run(debug=True, host='0.0.0.0', port=5001)