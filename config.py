"""项目配置文件。"""
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
NEWS_DATA_PATH = Path(r"E:\count\新闻数据.txt")
USER_DATA_PATH = Path(r"E:\count\用户偏好数据.json")
OUTPUT_DIR = BASE_DIR / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

STOPWORDS = {
    "的", "了", "在", "是", "我", "有", "和", "就", "不", "人", "都", "一",
    "上", "也", "很", "到", "说", "要", "去", "你", "会", "着", "没有", "看",
    "自己", "这", "那", "他", "她", "它", "我们", "他们", "以及", "而", "与",
    "或", "被", "为", "以", "等", "中", "对", "从", "将", "可以", "已经", "还",
}

TOP_K_RETRIEVAL = 10
BM25_K1 = 1.5
BM25_B = 0.75
TOP_K_RECOMMEND = 10
CF_NEIGHBORS = 5
FUSION_ALPHA = 0.6   # 检索权重
FUSION_BETA = 0.4    # 推荐权重
RANDOM_SEED = 42