"""IE.py
Information Extraction pipeline for the IMDb movie review dataset.

读取 `IMDB Dataset.csv` 文件，执行基本文本清理，
提取有用的文本特征（命名实体、关键词/名词、情感极性），
并将增强数据保存到新的 CSV 文件中，用于后续情感分析工作。

依赖项（如果缺少，通过 pip 安装）:
    pandas, spacy, textblob, tqdm, beautifulsoup4
    - 安装 spaCy 后，下载模型:  python -m spacy download en_core_web_sm
示例:
    python IE.py --input IMDB Dataset.csv --output imdb_enriched.csv --sample 10000
"""

from __future__ import annotations  # 启用 Python 3.10+ 的类型注解语法

import argparse  # 用于解析命令行参数
import html  # 用于处理 HTML 实体
import re  # 用于正则表达式处理
from pathlib import Path  # 用于跨平台路径处理
from typing import List, Tuple  # 用于类型注解

import pandas as pd  # 用于数据处理和分析
from bs4 import BeautifulSoup  # 用于解析和清理 HTML
from textblob import TextBlob  # type: ignore  # 用于情感分析
from tqdm import tqdm  # type: ignore  # 用于显示进度条

# 延迟导入 spaCy 模型 - 在函数内部加载以避免长时间启动
try:
    import spacy  # type: ignore  # 用于自然语言处理
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "spaCy is required for IE.py. Install with `pip install spacy` and download model via "
        "`python -m spacy download en_core_web_sm`"  # noqa: E501
    ) from e


def clean_html(text: str) -> str:
    """
    移除 HTML 标签并解码 HTML 实体。
    
    参数:
        text: 包含可能的 HTML 标记的输入文本
        
    返回:
        清理后的纯文本
    """
    text = html.unescape(text)  # 解码 HTML 实体（如 &amp; -> &）
    text = BeautifulSoup(text, "html.parser").get_text(separator=" ")  # 移除 HTML 标签
    return text.strip()  # 移除首尾空白


def normalize_whitespace(text: str) -> str:
    """
    将多个空白字符合并为单个空格。
    
    参数:
        text: 输入文本
        
    返回:
        空白规范化后的文本
    """
    return re.sub(r"\s+", " ", text).strip()  # 用正则表达式替换多个空白为单个空格


def preprocess(text: str) -> str:
    """
    评论字符串的完整预处理流程。
    
    参数:
        text: 原始评论文本
        
    返回:
        清理和规范化后的文本
    """
    return normalize_whitespace(clean_html(text))  # 组合 HTML 清理和空白规范化


def load_spacy_model():
    """
    加载小型英文 spaCy 模型，禁用不需要的组件以提高速度。
    
    返回:
        加载的 spaCy 模型
    """
    return spacy.load("en_core_web_sm", disable=["textcat"])  # 禁用文本分类组件以加快速度


def extract_entities(doc) -> List[Tuple[str, str]]:
    """
    返回 spaCy Doc 中命名实体的 (文本, 标签) 列表。
    
    参数:
        doc: spaCy 处理后的文档对象
        
    返回:
        实体列表，每个实体为 (文本, 类型) 元组
    """
    return [(ent.text, ent.label_) for ent in doc.ents]  # 提取所有实体及其类型


def extract_nouns(doc) -> List[str]:
    """
    从 spaCy Doc 返回词形还原的名词（作为关键词代理）。
    
    参数:
        doc: spaCy 处理后的文档对象
        
    返回:
        名词列表（已词形还原和小写化）
    """
    # 提取所有名词，排除停用词，并进行词形还原和小写化
    return [token.lemma_.lower() for token in doc if token.pos_ == "NOUN" and not token.is_stop]


def sentiment_polarity(text: str) -> float:
    """
    使用 TextBlob 计算情感极性（范围 -1 到 1）。
    
    参数:
        text: 输入文本
        
    返回:
        情感极性分数：负值表示负面情感，正值表示正面情感
    """
    return TextBlob(text).sentiment.polarity  # 使用 TextBlob 计算情感极性


def process_dataframe(df: pd.DataFrame, sample: int | None = None, batch_size: int = 1000) -> pd.DataFrame:
    """
    预处理并提取每个评论行的信息提取特征。

    参数:
        df: 输入 DataFrame，包含 'review' 和 'sentiment' 列
        sample: 可选整数 - 随机抽样这么多行以加快实验速度
        batch_size: spaCy 处理的批处理大小
        
    返回:
        增强后的 DataFrame，包含额外的特征列
    """
    # 如果指定了样本大小且小于数据集大小，则随机抽样
    if sample is not None and sample < len(df):
        df = df.sample(sample, random_state=42).reset_index(drop=True)

    # 清理文本，并显示进度条
    tqdm.pandas(desc="Cleaning text")
    df["clean_text"] = df["review"].progress_apply(preprocess)

    # 加载 spaCy 模型
    nlp = load_spacy_model()

    # 批处理 spaCy
    print(f"Processing with spaCy in batches of {batch_size}...")
    all_entities = []
    all_keywords = []
    
    # 将数据分成批次处理，提高效率
    for i in tqdm(range(0, len(df), batch_size), desc="spaCy batch processing"):
        # 获取当前批次的文本
        batch = df["clean_text"].iloc[i:i+batch_size].tolist()
        # 使用 spaCy 的 pipe 方法批量处理文本
        docs = list(nlp.pipe(batch))
        
        # 从处理后的文档中提取实体和关键词
        batch_entities = [extract_entities(doc) for doc in docs]
        batch_keywords = [extract_nouns(doc) for doc in docs]
        
        # 将当前批次的结果添加到总结果中
        all_entities.extend(batch_entities)
        all_keywords.extend(batch_keywords)
    
    # 将提取的特征添加到 DataFrame
    df["entities"] = all_entities
    df["keywords"] = all_keywords

    # 计算情感极性
    tqdm.pandas(desc="Sentiment polarity")
    df["polarity"] = df["clean_text"].progress_apply(sentiment_polarity)

    return df


def cli():  # pragma: no cover
    # 设置命令行参数解析器
    parser = argparse.ArgumentParser(description="IMDb Information Extraction")
    # 获取脚本所在目录，用于设置默认文件路径
    script_dir = Path(__file__).parent
    default_input = script_dir / "IMDB Dataset.csv"
    default_output = script_dir / "imdb_enriched.csv"
    # 添加命令行参数
    parser.add_argument("--input", type=Path, default=default_input, help="IMDb csv 文件路径")
    parser.add_argument("--output", type=Path, default=default_output, help="保存增强 csv 的路径")
    parser.add_argument("--sample", type=int, default=None, help="可选择抽样 N 行以加快速度")
    parser.add_argument("--batch-size", type=int, default=1000, help="spaCy 处理的批处理大小")  
    # 解析命令行参数
    args = parser.parse_args()
    # 检查输入文件是否存在
    if not args.input.exists():
        raise FileNotFoundError(f"Input file {args.input} not found")
    # 读取输入数据集
    print(f"Reading {args.input} …")
    df_in = pd.read_csv(args.input)
    # 检查数据集是否包含所需列
    required_cols = {"review", "sentiment"}
    if not required_cols.issubset(df_in.columns):
        raise ValueError(f"Expected columns {required_cols} in dataset, got {set(df_in.columns)}")
    # 处理数据集
    df_out = process_dataframe(df_in, sample=args.sample, batch_size=args.batch_size)
    # 保存处理后的数据集
    print(f"Saving enriched dataset → {args.output}")
    df_out.to_csv(args.output, index=False)

# 接运行时执行 CLI 函数
if __name__ == "__main__":
    cli()