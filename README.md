 混合检索与推荐系统 — 融合5种检索算法与多种推荐策略

  项目简介

基于 Python 开发的混合检索与推荐系统，融合了 TF-IDF、BM25、布尔检索、短语检索、语义检索 5 种检索算法，以及协同过滤、基于内容推荐、加权混合、级联混合、Switch 混合等多种推荐策略。系统提供完整的 Web 可视化界面，支持文章收藏、点赞、评论、关注作者等用户互动功能。

  核心功能

-  多种推荐策略：协同过滤（User-CF/Item-CF）、基于内容推荐、Weighted 混合、Cascade 混合、Switch 混合、Rank 混合
-  检索推荐融合：动态权重融合检索与推荐结果，实现个性化排序
-  完整评估体系：Precision、Recall、F1、MAP、NDCG、Hit Ratio
-  数据可视化：自动生成评估对比图表
-  Web 交互界面：基于 Flask 的可视化操作面板
-  用户互动：收藏、点赞、评论、关注作者、分享、举报

 技术栈

| 技术 | 用途 |
|------|------|
| Python 3.8+ | 编程语言 |   | Python 3.8  | 编程语言 || Python 3.8  | 编程语言 |   | Python 3.8  | 编程语言 |
| Flask | Web 后端框架 |
| scikit-learn | TF-IDF 向量化、相似度计算 |
| gensim | Word2Vec 语义检索 |
| jieba | 中文分词 |
| numpy / pandas | 数据处理 |
| matplotlib | 可视化图表 |
| HTML / CSS / JS | 前端界面 |

 运行说明

 环境要求
- Python 3.8+   - Python 3.8- Python 3.8 - Python 3.8- Python 3.8- Python 3.8- Python 3.8
- pip 包管理工具

 1. 克隆项目
```bash   ”“bash   “bash”;“bash
git clone https://github.com/mbabmmmm/Hybrid-Search-Recommendation-System.git
cd Hybrid-Search-Recommendation-System
```

 2. 安装依赖
```bash   ”“bash   “bash”;“bash
pip install -r requirements.txtPIP install -r requirements.txt
```

 3. 配置数据路径
在 `config.py` 中配置数据文件路径：
```python   ”“python   ```python   ”“python
NEWS_DATA_PATH = Path(r"你的路径/新闻数据.txt")NEWS_DATA_PATH = Path(r"你的路径/新闻数据.txt")NEWS_DATA_PATH = Path(r"你的路径/新闻数据.txt")NEWS_DATA_PATH = Path(r"你的路径/新闻数据.txt")
USER_DATA_PATH = Path(r"你的路径/用户偏好数据.json")USER_DATA_PATH = Path(r"你的路径/用户偏好数据.json")USER_DATA_PATH = Path(r"你的路径/用户偏好数据.json")USER_DATA_PATH = Path(r"你的路径/用户偏好数据.json")USER_DATA_PATH = Path(r"你的路径/用户偏好数据.json")USER_DATA_PATH = Path(r"你的路径/用户偏好数据.json")USER_DATA_PATH = Path(r"你的路径/用户偏好数据.json")USER_DATA_PATH = Path(r"你的路径/用户偏好数据.json")
```

 4. 运行 Web 界面
```bash   ”“bash   “bash”;“bash```bash   ”“bash   “bash”;“bash```bash   ”“bash   “bash”;“bash
python app.py
```
在浏览器中打开 `http://localhost:5000`

 5. 命令行模式
```bash   ”“bash   “bash”;“bash
python main.py
```

 实验结果

 检索系统评估

| 检索算法 | Precision@10 | Recall@10 | NDCG |
|---------|-------------|-----------|------|
| TF-IDF | 100% | 2.97% | 1.00 |
| BM25 | 100% | 2.97% | 1.00 |   | bm25 | 100% | 2.97% | 1.00 || BM25 | 100% | 2.97% | 1.00 | | BM25 | 100% | 2.97% | 1.00 |
| 语义检索 | 100% | 2.97% | 1.02 |
|   ```python   ”“python 布尔检索 | 52% | 1.58% | 0.54 ||NEWS_DATA_PATH = Path(r"你的路径/新闻数据.txt")NEWS_DATA_PATH = Path(r"你的路径/新闻数据.txt")NEWS_DATA_PATH = Path(r"你的路径/新闻数据.txt")NEWS_DATA_PATH = Path(r"你的路径/新闻数据.txt")   ```python   ”“python 布尔检索 | 52% | 1.58% | 0.54 |

 推荐系统评估

| 推荐策略 | Precision@10 | Hit Ratio |
|---------|-------------|-----------|
| Switch 混合 | 80.0% | 80.0% |
| Rank 混合 | 27.3% | 83.3% |
| Weighted 混合 | 29.5% | 62.5% |

 核心算法

| 模块 | 算法 | 说明 |
|------|------|------|
| 检索 | TF-IDF | 词频-逆文档频率向量检索 |
| 检索 | BM25 | 概率检索模型，词频饱和+长度归一化 |
| 检索 | 布尔检索 | 支持 AND / OR / NOT 逻辑操作 |
| 检索 | 短语检索 | 基于位置索引的精确短语匹配 |
| 检索 | 语义检索 | Word2Vec 词向量语义匹配 |
| 推荐 | 协同过滤 | User-User / Item-Item CF |
| 推荐 | 基于内容推荐 | 用户画像 + TF-IDF 特征匹配 |
| 推荐 | Switch 混合 | 根据用户活跃度自适应选择算法 |
| 推荐 | Rank 混合 | 非线性融合 + 交集奖励 |

 开发者

- GitHub: [mbabmmmm](https://github.com/mbabmmmm)
