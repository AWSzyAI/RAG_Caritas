import json
import os
import sys
import requests
session = requests.Session()
import hashlib
import re
import time
import argparse
import datetime
import logging
import threading
import concurrent.futures  

import jieba
import numpy as np
import pandas as pd  # 用于 CSV 处理
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm  # 进度条库
from dotenv import load_dotenv
load_dotenv()
from zhipuai import ZhipuAI

from kimi_api import  send_messages
KIMI_API_KEY = os.getenv("KIMI_API_KEY")



# ================================
# EMBEDDING_MODEL = "bge-m3"     #
EMBEDDING_MODEL = "embedding-3"  #
CHUNK_SIZE = 50                  #
# CHUNK_SIZE = 200               #
BATCH_SIZE = 10
SENTENCE_BATCH_SIZE = 100
BATCH_REQUEST_TO_ZHIPU = True #测试一次性请求一篇文章中的所有切片
MAX_WORKER = 20
# ================================


# ======================================================================
ARTICLES_FILE = "./data/articles.json"
SYSTEM_FILE = "./data/system.json"
log_dir = "logs"
# ======================================================================
if EMBEDDING_MODEL == "bge-m3":
    # bge-m3
    API_URL = os.getenv("API_URL_siliconflow")
    API_KEY = os.getenv("API_KEY_siliconflow")
    MODEL_NAME = "BAAI/bge-m3"
    # zhipu embedding-3
elif EMBEDDING_MODEL == "embedding-3":
    MODEL_NAME = "embedding-3"
    API_URL = os.getenv("API_URL_ZHIPU")
    API_KEY = os.getenv("ZHIPU_API_key")
    zhipu_client = ZhipuAI(api_key=API_KEY)

# ========================================================================
os.makedirs(log_dir, exist_ok=True)
log_filename = os.path.join(log_dir, f"log_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
cache_dir = os.path.join(".", "cache", f"{EMBEDDING_MODEL}-{CHUNK_SIZE}")
OUTPUT_DIR = os.path.join(".", "output")
os.makedirs(cache_dir, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
METADATA_FILE = f"{cache_dir}/metadata_cache.json"
EMBEDDING_FILE = f"{cache_dir}/embedding_vectors.json"
SYSTEM_METADATA_FILE = f"{cache_dir}/system_metadata_cache.json"
SYSTEM_EMBEDDING_FILE = f"{cache_dir}/system_embedding_vectors.json"
INDEX_FILE = f"{cache_dir}/index.csv"
FAIL_FILE = f"{cache_dir}/fail.csv"

UNPROCESSED = "❌"
PROCESSED = "✅"
# =========================================================================
# 设置日志记录
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    filename=log_filename,  # 把日志写入文件
    filemode='a'  # 可选：'w' 覆盖写入，'a' 追加写入
)
# =========================================================================


class SpinLock:
    def __init__(self):
        self.lock = threading.Lock()
    def acquire(self):
        while not self.lock.acquire(False):
            time.sleep(0.001)
    def release(self):
        self.lock.release()
    def __enter__(self):
        self.acquire()
        return self
    def __exit__(self, exc_type, exc_value, traceback):
        self.release()

global_lock = SpinLock()


def extract_json(response):
    match = re.search(r'\{.*\}', response, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError as e:
            logging.warning(f"JSON 解析失败: {e}, 跳过该条数据: {response}")
            return None
    else:
        logging.warning(f"未找到 JSON 数据: {response}")
        return None

def clean_value(value):
    """简单清洗返回值（如去除两端空格）"""
    return value.strip() if isinstance(value, str) else value

def query_partitions(sentence, k=3, threshold=0.4):
    """
    根据输入的句子，使用 find_matches 查询获得前 k 个匹配的文本切片，
    拼接片段内容和对应链接形成素材字符串。
    """
    matches = find_matches([sentence], top_n=k, threshold=threshold)
    parts = ""
    if matches and matches[0]["matches"]:
        for match in matches[0]["matches"]:
            # match 格式：(相似度, 文本片段, 文章标题, zhihu_link)
            parts += f"素材：{match[1]} (链接：{match[3]})\n"
    return parts

def clean_text(text):
    logging.info("清洗文本")
    text = re.sub(r'https?://\S+', '', text)
    text = re.sub(r'!\[.*?\]\(.*?\.svg\)', '', text, flags=re.IGNORECASE)
    text = re.sub(r'<svg.*?>.*?</svg>', '', text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def safe_write_json(filepath, data):
    temp_filepath = filepath + ".tmp"
    with open(temp_filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(temp_filepath, filepath)

def safe_write_csv(filepath, df):
    try:
        df.to_csv(filepath, index=False, encoding="utf-8")
    except Exception as e:
        logging.error("写入 index.csv 失败：%s", e)

def load_index():
    if os.path.exists(INDEX_FILE):
        logging.info("加载索引: %s", INDEX_FILE)
        df = pd.read_csv(INDEX_FILE)
        return df[["title", "url", "processed"]]
    else:
        logging.info("索引文件不存在，根据 articles.json 创建新的索引表")
        return pd.DataFrame(columns=["title", "url", "processed"])

def save_index(index_df):
    index_df["url"] = index_df["url"].apply(lambda x: x.rstrip() + " ")
    index_df = index_df[["title", "url", "processed"]]
    index_df["sort_key"] = index_df["processed"].apply(lambda x: 0 if x == PROCESSED else 1)
    index_df = index_df.sort_values(by=["sort_key"]).drop(columns=["sort_key"])
    safe_write_csv(INDEX_FILE, index_df)
    return index_df

def update_index_from_metadata(index_df, metadata_cache):
    logging.info("根据 metadata_cache 更新 index 状态")
    processed_urls = { entry.get("url", "").strip() for entry in metadata_cache.values() if entry.get("url", "").strip() }
    index_df["url_stripped"] = index_df["url"].apply(lambda x: x.strip())
    index_df["processed"] = index_df["url_stripped"].apply(lambda x: PROCESSED if x in processed_urls else UNPROCESSED)
    index_df.drop(columns=["url_stripped"], inplace=True)
    index_df = save_index(index_df)
    logging.info("index 更新完毕，共更新 %d 个条目", len(processed_urls))
    return index_df

def load_metadata_cache():
    logging.info("加载元数据缓存: %s", METADATA_FILE)
    if os.path.exists(METADATA_FILE):
        try:
            with open(METADATA_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            logging.error("解析 %s 失败: %s", METADATA_FILE, e)
            backup_file = METADATA_FILE + ".bak"
            os.rename(METADATA_FILE, backup_file)
            logging.info("已将损坏的文件备份到: %s", backup_file)
            return {}
    return {}

def load_embedding_cache():
    logging.info("加载嵌入向量缓存: %s", EMBEDDING_FILE)
    if os.path.exists(EMBEDDING_FILE):
        try:
            with open(EMBEDDING_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            logging.error("解析 %s 失败: %s", EMBEDDING_FILE, e)
            backup_file = EMBEDDING_FILE + ".bak"
            os.rename(EMBEDDING_FILE, backup_file)
            logging.info("已将损坏的文件备份到: %s", backup_file)
            return {}
    return {}

def save_metadata_cache():
    safe_write_json(METADATA_FILE, metadata_cache)

def save_embedding_cache():
    safe_write_json(EMBEDDING_FILE, embedding_vectors)

def save_cache():
    save_metadata_cache()
    save_embedding_cache()

def load_system_metadata_cache():
    logging.info("加载 system 级别的元数据缓存: %s", SYSTEM_METADATA_FILE)
    if os.path.exists(SYSTEM_METADATA_FILE):
        try:
            with open(SYSTEM_METADATA_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            logging.error("解析 %s 失败: %s", SYSTEM_METADATA_FILE, e)
            backup_file = SYSTEM_METADATA_FILE + ".bak"
            os.rename(SYSTEM_METADATA_FILE, backup_file)
            logging.info("已将损坏的文件备份到: %s", backup_file)
            return {}
    return {}

def load_system_embedding_cache():
    logging.info("加载 system 级别的嵌入向量缓存: %s", SYSTEM_EMBEDDING_FILE)
    if os.path.exists(SYSTEM_EMBEDDING_FILE):
        try:
            with open(SYSTEM_EMBEDDING_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            return {k: np.array(v) for k, v in data.items()}
        except json.JSONDecodeError as e:
            logging.error("解析 %s 失败: %s", SYSTEM_EMBEDDING_FILE, e)
            backup_file = SYSTEM_EMBEDDING_FILE + ".bak"
            os.rename(SYSTEM_EMBEDDING_FILE, backup_file)
            logging.info("已将损坏的文件备份到: %s", backup_file)
            return {}
    return {}

def save_system_metadata_cache():
    safe_write_json(SYSTEM_METADATA_FILE, system_metadata_cache)

def save_system_embedding_cache():
    system_embedding_vectors_serializable = {k: v.tolist() if isinstance(v, np.ndarray) else v 
                                             for k, v in system_embedding_vectors.items()}
    safe_write_json(SYSTEM_EMBEDDING_FILE, system_embedding_vectors_serializable)
    logging.info(f"system_embedding_vectors.json 已成功保存，共 {len(system_embedding_vectors)} 条数据")

metadata_cache = load_metadata_cache()
embedding_vectors = load_embedding_cache()
system_metadata_cache = load_system_metadata_cache()
system_embedding_vectors = load_system_embedding_cache()

def build_index_from_articles(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    total_articles = len(data.get("articles", []))
    effective_articles = []
    for article in data["articles"]:
        if "想法集" in article.get("tags", []):
            continue
        title = article.get("title", "")
        if not title:
            continue
        zhihu_link = article.get("zhihuLink", "").strip()
        if not zhihu_link or zhihu_link.startswith("ID_"):
            continue
        effective_articles.append({
            "title": title,
            "url": zhihu_link,
            "processed": UNPROCESSED
        })
    effective_count = len(effective_articles)
    df = pd.DataFrame(effective_articles)
    logging.info("索引表中总文章数量：%d", effective_count)
    return total_articles, effective_count, df

def hash_sentence(text):
    return hashlib.md5(text.encode()).hexdigest()

def split_text_into_chunks(text, chunk_size=CHUNK_SIZE, overlap=10):
    tokens = list(jieba.lcut(text))
    chunks = []
    if not tokens:
        logging.warning("文本分词结果为空")
        return chunks
    step = chunk_size - overlap
    i = 0
    while i < len(tokens):
        chunk = tokens[i:i + chunk_size]
        chunks.append("".join(chunk))
        i += step
    return chunks

def clean_and_prepare_text(content):
    content = clean_text(content)
    text = " ".join([line for line in content.split('\n') if not line.startswith("![]") and not line.startswith("```")])
    return text

def get_batch_embeddings(chunks, article_titles, urls, provider="zhipu"):
    global metadata_cache, embedding_vectors

    logging.info("获取 batch 嵌入，共 %d 个切片", len(chunks))
    batch_hashes = [hash_sentence(c) for c in chunks]
    batch_chunks, batch_titles, batch_urls = [], [], []
    filtered_hashes = []

    for i, h in enumerate(batch_hashes):
        if h not in embedding_vectors:
            filtered_hashes.append(h)
            batch_chunks.append(chunks[i])
            batch_titles.append(article_titles[i])
            batch_urls.append(urls[i])

    if not batch_chunks:
        logging.info("所有切片均已计算，无需请求 API")
        return

    max_retries = 3
    retry_delay = 5

    for attempt in range(max_retries):
        try:
            logging.info("请求嵌入 API（%s），尝试次数 %d", provider, attempt + 1)

            if provider == "zhipu":
                # 调用 Zhipu SDK
                response = zhipu_client.embeddings.create(
                    model=EMBEDDING_MODEL,
                    input=batch_chunks
                )
                if "data" in response and response["data"]:
                    sorted_data = sorted(response["data"], key=lambda x: x["index"])
                    embeddings = [item["embedding"] for item in sorted_data]
                else:
                    logging.warning("Zhipu 返回为空")
                    return

            elif provider == "openai":
                # 使用 requests 调用 OpenAI（示例）
                headers = {
                    "Authorization": f"Bearer {API_KEY}",
                    "Content-Type": "application/json"
                }
                data = {
                    "input": batch_chunks,
                    "model": EMBEDDING_MODEL
                }
                response = requests.post(API_URL, headers=headers, json=data, timeout=30)
                response.raise_for_status()
                response_json = response.json()
                if "data" in response_json and response_json["data"]:
                    sorted_data = sorted(response_json["data"], key=lambda x: x["index"])
                    embeddings = [item["embedding"] for item in sorted_data]
                else:
                    logging.warning("OpenAI 返回为空")
                    return

            else:
                logging.error("不支持的 provider: %s", provider)
                return

            for i, h in enumerate(filtered_hashes):
                embedding_vectors[h] = embeddings[i] if i < len(embeddings) else []
                metadata_cache[h] = {
                    "sentence": batch_chunks[i],
                    "article": batch_titles[i],
                    "url": batch_urls[i].strip()
                }
            save_cache()
            logging.info("Batch 嵌入成功")
            return

        except Exception as e:
            logging.warning("请求失败: %s, 正在重试 (%d/%d)...", e, attempt + 1, max_retries)
            time.sleep(retry_delay)

    logging.error("API 请求失败，跳过该 batch")

def get_embedding(text, article_title, url, is_query=False):
    global metadata_cache, embedding_vectors, system_metadata_cache, system_embedding_vectors
    text_hash = hash_sentence(text)
    if is_query:
        embedding_cache = system_embedding_vectors
        metadata_cache_ref = system_metadata_cache
        save_metadata_cache_func = save_system_metadata_cache
        save_embedding_cache_func = save_system_embedding_cache
    else:
        embedding_cache = embedding_vectors
        metadata_cache_ref = metadata_cache
        save_metadata_cache_func = save_metadata_cache
        save_embedding_cache_func = save_embedding_cache
    with global_lock:
        if text_hash in embedding_cache:
            logging.info(f"使用缓存中的嵌入: {text[:30]}")
            return embedding_cache[text_hash]
    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
    data = {"input": text, "model": MODEL_NAME}
    max_retries = 3
    retry_delay = 5
    for attempt in range(max_retries):
        try:
            logging.info(f"请求 OpenAI API 获取嵌入 (尝试 {attempt+1}/{max_retries}): {text[:30]}")
            response = requests.post(API_URL, headers=headers, json=data, timeout=30)
            response.raise_for_status()
            response_json = response.json()
            if "data" in response_json and response_json["data"]:
                embedding = response_json["data"][0]["embedding"]
                with global_lock:
                    embedding_cache[text_hash] = embedding
                    metadata_cache_ref[text_hash] = {
                        "sentence": text,
                        "article": article_title if not is_query else "预设 query",
                        "url": url.strip() if not is_query else "#"
                    }
                    save_metadata_cache_func()
                    save_embedding_cache_func()
                logging.info(f"新嵌入已存入 {'system' if is_query else '文章'} 缓存: {text[:30]}")
                return embedding
            else:
                logging.error(f"API 返回错误: {response_json}, 跳过该文本")
                return None
        except requests.exceptions.RequestException as e:
            logging.warning(f"网络错误: {e}, 正在重试 ({attempt+1}/{max_retries})...")
            time.sleep(retry_delay)
    logging.error(f"达到最大重试次数，跳过该文本: {text[:30]}")
    return None

def get_batch_embeddings(texts, article_title, url, is_query=False, batch_size=64):
    global metadata_cache, embedding_vectors, system_metadata_cache, system_embedding_vectors

    if is_query:
        embedding_cache = system_embedding_vectors
        metadata_cache_ref = system_metadata_cache
        save_metadata_cache_func = save_system_metadata_cache
        save_embedding_cache_func = save_system_embedding_cache
    else:
        embedding_cache = embedding_vectors
        metadata_cache_ref = metadata_cache
        save_metadata_cache_func = save_metadata_cache
        save_embedding_cache_func = save_embedding_cache

    # 提取未缓存的文本及其 hash
    new_texts, new_text_hashes = [], []
    for text in texts:
        h = hash_sentence(text)
        with global_lock:
            if h not in embedding_cache:
                new_texts.append(text)
                new_text_hashes.append(h)

    if not new_texts:
        logging.info("所有嵌入均已存在于缓存中")
        return [embedding_cache[hash_sentence(t)] for t in texts]

    # 分批处理，每批最多 64 个
    for i in range(0, len(new_texts), batch_size):
        batch_texts = new_texts[i:i + batch_size]
        batch_hashes = new_text_hashes[i:i + batch_size]

        headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
        data = {"input": batch_texts, "model": MODEL_NAME}
        max_retries = 3
        retry_delay = 5

        for attempt in range(max_retries):
            try:
                logging.info(f"请求 OpenAI API 获取嵌入 (尝试 {attempt+1}/{max_retries})，本批 %d 条", len(batch_texts))
                response = requests.post(API_URL, headers=headers, json=data, timeout=60)
                response.raise_for_status()
                response_json = response.json()

                if "data" in response_json:
                    with global_lock:
                        for item in response_json["data"]:
                            h = batch_hashes[item["index"]]
                            emb = item["embedding"]
                            embedding_cache[h] = emb
                            metadata_cache_ref[h] = {
                                "sentence": batch_texts[item["index"]],
                                "article": article_title if not is_query else "预设 query",
                                "url": url.strip() if not is_query else "#"
                            }
                    break  # 本批成功，跳出 retry 循环
                else:
                    logging.error(f"API 返回格式错误: {response_json}")
                    break
            except requests.exceptions.RequestException as e:
                logging.warning(f"网络错误: {e}, 正在重试 ({attempt+1}/{max_retries})...")
                time.sleep(retry_delay)
        else:
            logging.error("达到最大重试次数，本批失败")

    # 保存全部缓存
    with global_lock:
        save_metadata_cache_func()
        save_embedding_cache_func()

    # 返回完整嵌入列表（含已有的 + 本次新增的）
    return [embedding_cache.get(hash_sentence(t)) for t in texts]


def process_sentences(chunks, article_title, url):
    logging.info("处理文章切片（并发模式），文章标题: %s, 切片数量: %d", article_title, len(chunks))
    filtered_chunks = []
    for chunk in chunks:
        h = hash_sentence(chunk)
        with global_lock:
            if h in embedding_vectors:
                continue
        filtered_chunks.append(chunk)
    if not filtered_chunks:
        logging.info("所有切片均已计算，无需再次处理")
        return

    if BATCH_REQUEST_TO_ZHIPU:
        _ = get_batch_embeddings(filtered_chunks, article_title, url, is_query=False)
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKER) as executor:
            futures = [executor.submit(get_embedding, chunk, article_title, url, is_query=False) for chunk in filtered_chunks]
            for future in concurrent.futures.as_completed(futures):
                _ = future.result()
    

def load_articles(file_path, sample_ratio=0.1, sample_count=None):
    logging.info("加载文章文件: %s", file_path)
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    total_articles = len(data.get("articles", []))
    logging.info("当前总文章数量: %d", total_articles)
    valid_urls_set = set()
    for article in data["articles"]:
        if "想法集" in article.get("tags", []):
            continue
        title = article.get("title", "")
        if not title:
            continue
        zhihu_link = article.get("zhihuLink", "").strip()
        if not zhihu_link or zhihu_link.startswith("ID_"):
            continue
        valid_urls_set.add(zhihu_link)
    logging.info("当前有效文章数量: %d", len(valid_urls_set))
    index_df = load_index()
    if index_df.empty:
        _, effective_count, index_df = build_index_from_articles(file_path)
        logging.info("索引表中共有 %d 篇文章", effective_count)
        index_df = save_index(index_df)
    else:
        index_df = index_df[index_df["url"].apply(lambda x: x.strip()).isin(valid_urls_set)]
    index_df = update_index_from_metadata(index_df, metadata_cache)
    logging.info("当前已建立索引的文章数量: %d", len(index_df))
    new_entries = []
    for article in data["articles"]:
        if "想法集" in article.get("tags", []):
            continue
        title = article.get("title", "")
        if not title:
            continue
        zhihu_link = article.get("zhihuLink", "").strip()
        if not zhihu_link or zhihu_link.startswith("ID_"):
            continue
        if zhihu_link not in index_df["url"].apply(lambda x: x.strip()).values:
            new_entries.append({"title": title, "url": zhihu_link, "processed": UNPROCESSED})
    if new_entries:
        logging.info("新增 %d 篇文章到索引", len(new_entries))
        index_df = pd.concat([index_df, pd.DataFrame(new_entries)], ignore_index=True)
        index_df = save_index(index_df)
    unprocessed_articles = index_df[index_df["processed"] == UNPROCESSED]
    if unprocessed_articles.empty:
        logging.info("所有文章已处理，无需采样。")
        return [], index_df
    if sample_count is not None:
        sample_size = sample_count
    else:
        sample_size = max(1, int(len(unprocessed_articles) * sample_ratio))
    logging.info("随机采样 %d 篇未处理的文章...", sample_size)
    sampled_articles = unprocessed_articles.sample(n=sample_size)
    articles = []
    for _, row in tqdm(sampled_articles.iterrows(), total=sample_size, desc="加载采样文章", file=sys.stdout):
        article = next(filter(lambda a: a.get("zhihuLink", "").strip() == row["url"].strip(), data["articles"]), None)
        if article is None:
            logging.warning("未匹配到文章，URL: %s", row["url"])
            index_df.loc[index_df["url"].apply(lambda x: x.strip()) == row["url"].strip(), "processed"] = "未找到"
            continue
        content = article.get("content", "")
        content = clean_text(content)
        text = " ".join([line for line in content.split('\n') if not line.startswith("![]") and not line.startswith("```")])
        chunk_list = split_text_into_chunks(text)
        process_sentences(chunk_list, article["title"], article.get("zhihuLink", "").strip())
        articles.append({
            "id": article["id"],
            "title": article["title"],
            "url": article.get("zhihuLink", "").strip(),
            "chunks": chunk_list,
            "content": content
        })
        index_df.loc[index_df["url"].apply(lambda x: x.strip()) == row["url"].strip(), "processed"] = PROCESSED
    index_df = save_index(index_df)
    logging.info("采样文章加载并处理完毕")
    

def process_articles(valid_articles, index_df):
    unprocessed_articles = index_df[index_df["processed"] == UNPROCESSED]
    if unprocessed_articles.empty:
        logging.info("所有文章已处理，无需进一步采样。")
        return index_df
    logging.info("按 batch=%d 开始处理文章...", BATCH_SIZE)
    batches = []
    for i in range(0, len(unprocessed_articles), BATCH_SIZE):
        batches.append(unprocessed_articles.iloc[i:i + BATCH_SIZE])
    
    def process_batch(batch_articles):
        chunks_list, titles, urls = [], [], []
        processed_urls_in_batch = []
        for _, row in batch_articles.iterrows():
            article = next(filter(lambda a: (a.get("zhihuLink", "").strip() if a.get("zhihuLink", "").strip() 
                                             else f"ID_{a.get('id', a.get('title'))}") == row["url"].strip(), valid_articles), None)
            if not article:
                logging.warning("在有效文章中未匹配到，URL: %s", row["url"])
                index_df.loc[index_df["url"].apply(lambda x: x.strip()) == row["url"].strip(), "processed"] = "未找到"
                continue
            content = article.get("content", "")
            content = clean_text(content)
            text = " ".join([line for line in content.split('\n') if not line.startswith("![]") and not line.startswith("```")])
            chunk_list = split_text_into_chunks(text)
            if chunk_list:
                chunks_list.extend(chunk_list)
                titles.extend([article["title"]] * len(chunk_list))
                urls.extend([(article.get("zhihuLink", "").strip() if article.get("zhihuLink", "").strip() 
                              else f"ID_{article.get('id', article['title'])}")] * len(chunk_list))
                processed_urls_in_batch.append(row["url"].strip())
        logging.info("当前 batch 共 %d 个切片，开始调用 API", len(chunks_list))
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = []
            for j in range(0, len(chunks_list), SENTENCE_BATCH_SIZE):
                batch_chunk = chunks_list[j:j + SENTENCE_BATCH_SIZE]
                batch_titles = titles[j:j + SENTENCE_BATCH_SIZE]
                batch_urls = urls[j:j + SENTENCE_BATCH_SIZE]
                futures.append(executor.submit(get_batch_embeddings, batch_chunk, batch_titles, batch_urls))
            for future in concurrent.futures.as_completed(futures):
                _ = future.result()
        index_df.loc[index_df["url"].apply(lambda x: x.strip()).isin(processed_urls_in_batch), "processed"] = PROCESSED
        save_index(index_df)
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        pool.map(process_batch, batches)
    
    logging.info("所有文章批量处理完毕")
    return index_df

def find_matches(leaf_nodes, top_n=10, threshold=0.6):
    global system_embedding_vectors, system_metadata_cache
    logging.info("开始进行 query 匹配，共 %d 个 query", len(leaf_nodes))
    matches = []
    for query in tqdm(leaf_nodes, desc="计算 Query 嵌入", file=sys.stdout):
        query_hash = hash_sentence(query)
        if query_hash in system_embedding_vectors:
            logging.info(f"使用 system 缓存中的 query 嵌入: {query}")
            query_embedding = np.array(system_embedding_vectors[query_hash])
        else:
            query_embedding = np.array(get_embedding(query, "未知标题", "#", is_query=True))
            if query_embedding is not None:
                system_embedding_vectors[query_hash] = query_embedding
                system_metadata_cache[query_hash] = {
                    "sentence": query,
                    "article": "预设 query",
                    "url": "#"
                }
                save_system_metadata_cache()
                save_system_embedding_cache()
                logging.info(f"新的 query 嵌入已存入 system 缓存: {query}")
        similarities = []
        for text_hash, vector in tqdm(embedding_vectors.items(), desc="进行相似度计算", leave=False, file=sys.stdout):
            sim = cosine_similarity([query_embedding], [np.array(vector)])[0][0]
            meta = metadata_cache.get(text_hash, {})
            similarities.append((sim, meta.get("sentence", ""), meta.get("article", ""), meta.get("url", "")))
        sorted_results = sorted(similarities, key=lambda x: x[0], reverse=True)
        filtered_results = [item for item in sorted_results if item[0] > threshold][:top_n]
        if filtered_results:
            matches.append({"query": query, "matches": filtered_results})
    logging.info("query 匹配完成")
    return matches

def save_results(matches, processed_count, total_slices, top_n, threshold):
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(f"{OUTPUT_DIR}/{timestamp}", exist_ok=True)
    result_filename = f"{OUTPUT_DIR}/{timestamp}/{processed_count}篇{total_slices}切片中排名前{top_n}相似度大于{threshold}的检索结果.md"
    logging.info("保存检索结果到: %s", result_filename)
    with open(result_filename, "w", encoding="utf-8") as f:
        f.write("# 匹配结果\n\n")
        for match in matches:
            f.write(f"## Query: {match['query']}\n")
            f.write("🔍 检索到以下匹配片段：\n\n")
            for sim, sentence, title, url in match["matches"]:
                f.write(f"- **{title}** (相似度: {sim:.2f})\n")
                f.write(f"  - URL: [{url}]({url})\n")
                f.write(f"  - 匹配片段: {sentence}\n\n")
    logging.info("检索结果保存完毕")

def extract_leaf_nodes(tree):
    leaves = []
    if isinstance(tree, dict):
        for key, value in tree.items():
            leaves.extend(extract_leaf_nodes(value))
    elif isinstance(tree, list):
        for item in tree:
            leaves.extend(extract_leaf_nodes(item))
    else:
        leaves.append(tree)
    return leaves

def new_business_run(df):
    """
    对输入 DataFrame 中的每一行进行处理，生成内心旁白，
    并返回生成成功的结果列表及失败的句子列表（原始数据行）。
    """
    results = []
    fails = []
    for idx, row in df.iterrows():
        sentence = row["自我肯定语"]
        # 获取多个素材片段作为生成依据
        parts_material = query_partitions(sentence, k=3, threshold=0.6)
        prompt = f"""
自我肯定语：{sentence}

请使用这些素材<parts>{parts_material}</parts>，为输入的自我肯定语生成一段内心旁白。
注意适当换行以减少读者的阅读难度。分三到四段生成内心旁白。不要写诗。
约500字。
必须以第一人称叙述。

请严格按照以下 JSON 格式返回数据：
{{
"inner_monologue": "这里是生成的内心旁白内容"
}}
"""
        try:
            messages = [{"role": "user", "content": prompt}]
            response = send_messages(messages, api_key=KIMI_API_KEY)
            parsed_response = extract_json(response)
            if parsed_response and "inner_monologue" in parsed_response:
                check_prompt = f"""
针对上一次生成的内心旁白：
{parsed_response["inner_monologue"]}

请检查并优化以下内容：
- 修正标点/空格问题
- 改善语句通顺度
- 统一人称（第一人称），必须以第一人称叙述。
- 删除外语内容
- 防止场景过于具体
- 确保500字长度
- 删除奇怪比喻
- 修正语病/错别字

直接返回优化后的JSON：
{{
    "inner_monologue": "这里是修改后生成的内心旁白内容"
}}
"""
                messages.append({"role": "user", "content": check_prompt})
                check_response = send_messages(messages, api_key=KIMI_API_KEY)
                check_parsed_response = extract_json(check_response)
                paragraphs = check_parsed_response["inner_monologue"].split("\n")
                valid = True
                for para in paragraphs:
                    if len(para) > 300:
                        logging.warning(f"单段落超过300字，跳过：{para}")
                        valid = False
                        break
                if not valid:
                    fails.append(row)
                    continue
                if check_parsed_response and "inner_monologue" in check_parsed_response:
                    inner_monologue = clean_value(check_parsed_response["inner_monologue"])
                    # 使用生成的内心旁白进行检索，获取前 3 个参考片段和对应的 zhihu_link
                    retrieval_results = find_matches([inner_monologue], top_n=3, threshold=0.6)
                    reference_sources = []
                    if retrieval_results and retrieval_results[0]["matches"]:
                        for match in retrieval_results[0]["matches"]:
                            reference_sources.append({
                                "snippet": match[1],
                                "zhihu_link": match[3]
                            })
                    result = {
                        "自我肯定语": sentence,
                        "自我旁白": inner_monologue,
                        "MODEL_NAME": MODEL_NAME,
                        "参考源": json.dumps(reference_sources, ensure_ascii=False)
                    }
                    results.append(result)
                else:
                    logging.warning(f"生成失败，跳过: {sentence}")
                    fails.append(row)
            else:
                logging.warning(f"生成失败，跳过: {sentence}")
                fails.append(row)
        except Exception as e:
            logging.error(f"处理 {sentence} 时发生错误: {e}")
            fails.append(row)
    return results, fails

def make_inner_monologue_by_kimi(mode=None):
    """
    从 CSV 中抽样句子生成内心旁白，并保存生成结果；
    对生成失败的句子保存到 fail.csv 中，且支持重试。
    """
    if not mode:
        while True:
            print("请选择新业务处理模式：")
            print("1. 从头开始")
            print("2. 从 fail.csv 开始")
            mode = input("请输入 1 或 2：").strip()
            if mode == "1":
                try:
                    df = pd.read_csv("./data/0315句子更新 - 汇总表.csv").sample(5)
                except Exception as e:
                    logging.error("读取 CSV 文件失败: %s", e)
                    return
                break
            elif mode == "2":
                if os.path.exists(FAIL_FILE):
                    df = pd.read_csv(FAIL_FILE)
                    if df.empty:
                        print("./cache/fail.csv 中没有失败的记录，程序退出。")
                        return
                    break
                else:
                    print("./cache/fail.csv 文件不存在，请选择从头开始。")
            else:
                print("输入有误，请重新选择。")
    else:
        if mode == "1":
            try:
                df = pd.read_csv("./data/0315句子更新 - 汇总表.csv").sample(5)
            except Exception as e:
                logging.error("读取 CSV 文件失败: %s", e)
                return
        elif mode == "2":
            if os.path.exists(FAIL_FILE):
                df = pd.read_csv(FAIL_FILE)
                if df.empty:
                    print("./cache/fail.csv 中没有失败的记录，程序退出。")
                    return
            else:
                print("./cache/fail.csv 文件不存在，请选择从头开始。")
                return
        else:
            print("输入有误，请重新选择。")
            return


    results, fails = new_business_run(df)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    result_filename = f"{OUTPUT_DIR}/{timestamp}-{cache_dir}.csv"
    df_results = pd.DataFrame(results)
    df_results.to_csv(result_filename, index=False, encoding='utf-8-sig')
    logging.info("新业务生成结果已保存到 CSV 文件: %s", result_filename)
    print("新业务生成结果已保存到 CSV 文件:", result_filename)

    # 保存失败记录
    df_fail = pd.DataFrame(fails)
    df_fail.to_csv(FAIL_FILE, index=False, encoding='utf-8-sig')
    if not df_fail.empty:
        print(f"共有 {len(df_fail)} 个句子生成失败，已保存到 fail.csv。")
        retry = input("是否立即重试这些失败的记录？(Y/N): ").strip().lower()
        if retry == "y":
            # 递归重试
            make_inner_monologue_by_kimi(mode="2")
    else:
        print("全部生成成功，无失败记录。")



# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="文章检索与嵌入处理")
#     parser.add_argument("-q", "--question", type=str, help="输入单个检索 query")
#     parser.add_argument("--debug", action="store_true", help="是否显示 DEBUG 信息")
#     args = parser.parse_args()
#     if args.debug:
#         logging.getLogger().setLevel(logging.DEBUG)
#         logging.debug("DEBUG 模式已开启")

#     current_index = load_index()
#     if current_index.empty:
#         total_articles, effective_count, current_index = build_index_from_articles(ARTICLES_FILE)
#         logging.info("当前总文章数量: %d", total_articles)
#         logging.info("当前有效文章数量: %d", effective_count)
#         logging.info("索引表中共有 %d 篇文章", effective_count)
#         current_index = save_index(current_index)
#     else:
#         logging.info("索引表中共有 %d 篇文章", len(current_index))
#     current_index = update_index_from_metadata(current_index, metadata_cache)
#     total_index_count = len(current_index)
#     processed_count = len(current_index[current_index["processed"] == PROCESSED])
#     logging.info("当前已建立索引的文章数量: %d", total_index_count)
#     logging.info("当前已建立切片的文章数量: %d", processed_count)
#     logging.info("当前未建立切片的文章数量: %d", total_index_count - processed_count)
#     logging.info("当前总切片数量：%d", len(embedding_vectors))
    
#     if args.question:
#         query = args.question
#         logging.info("收到检索 query: %s", query)
#         try:
#             top_n = int(input("请输入保留的匹配数量（默认10）: ").strip() or 10)
#         except:
#             top_n = 10
#         try:
#             threshold = float(input("请输入相似度阈值（默认0.6）: ").strip() or 0.6)
#         except:
#             threshold = 0.6
#         matches = find_matches([query], top_n=top_n, threshold=threshold)
#         save_results(matches, processed_count, len(embedding_vectors), top_n, threshold)
#     else:
#         print(f"{cache_dir}")
#         print("请选择后续操作:")
#         print("1. 添加新的切片内容（处理新文章）")
#         print("2. 直接进行查询")
#         print("3. 生成自我旁白（新业务）")
#         choice = input("请输入 1, 2 或 3：").strip()
#         if choice == "1":
#             logging.info("用户选择添加新的切片内容")
#             current_index = load_index()
#             unprocessed_articles = current_index[current_index["processed"] == UNPROCESSED]
#             unprocessed_count = len(unprocessed_articles)
#             print(f"当前还有 {unprocessed_count} 篇文章未处理。")
#             while True:
#                 try:
#                     sample_count = int(input("请输入要添加处理的文章数量（不超过未处理文章数量）：").strip())
#                     if 1 <= sample_count <= unprocessed_count:
#                         break
#                     else:
#                         print(f"请输入一个介于 1 到 {unprocessed_count} 的整数。")
#                 except ValueError:
#                     print("输入无效，请输入一个整数。")
#             valid_articles, index_df = load_articles(ARTICLES_FILE, sample_count=sample_count)
#             current_index = load_index()  # 重新加载更新后的索引
#             total_index_count = len(current_index)
#             processed_count = len(current_index[current_index["processed"] == PROCESSED])
#             logging.info("当前已建立索引的文章数量: %d", total_index_count)
#             logging.info("当前已建立切片的文章数量: %d", processed_count)
#             try:
#                 top_n = int(input("请输入保留的匹配数量（默认10）: ").strip() or 10)
#             except:
#                 top_n = 10
#             try:
#                 threshold = float(input("请输入相似度阈值（默认0.6）: ").strip() or 0.6)
#             except:
#                 threshold = 0.6
#             sub_choice = input("请选择查询模式:\n1. 自定义 query\n2. 使用 system.json 中预设的 query\n请输入 1 或 2：").strip()
#             if sub_choice == "1":
#                 query = input("请输入检索 query: ").strip()
#                 if query:
#                     matches = find_matches([query], top_n=top_n, threshold=threshold)
#                     save_results(matches, processed_count, len(embedding_vectors), top_n, threshold)
#                 else:
#                     logging.info("未进行查询")
#             elif sub_choice == "2":
#                 logging.info("使用 system.json 中预设的 query")
#                 try:
#                     with open(SYSTEM_FILE, "r", encoding="utf-8") as f:
#                         json_tree = json.load(f)
#                     preset_queries = extract_leaf_nodes(json_tree.get("家族", {}))
#                     if preset_queries:
#                         matches = find_matches(preset_queries, top_n=top_n, threshold=threshold)
#                         save_results(matches, processed_count, len(embedding_vectors), top_n, threshold)
#                     else:
#                         logging.info("system.json 中没有预设的 query")
#                 except Exception as e:
#                     logging.error("读取 system.json 失败: %s", e)
#             else:
#                 logging.info("未选择任何查询模式，程序退出")
#         elif choice == "2":
#             logging.info("用户选择直接进行查询")
#             sub_choice = input("请选择查询模式:\n1. 自定义 query\n2. 使用 system.json 中预设的 query\n请输入 1 或 2：").strip()
#             try:
#                 top_n = int(input("请输入保留的匹配数量（默认10）: ").strip() or 10)
#             except:
#                 top_n = 10
#             try:
#                 threshold = float(input("请输入相似度阈值（默认0.6）: ").strip() or 0.6)
#             except:
#                 threshold = 0.6
#             if sub_choice == "1":
#                 query = input("请输入检索 query: ").strip()
#                 if query:
#                     matches = find_matches([query], top_n=top_n, threshold=threshold)
#                     save_results(matches, processed_count, len(embedding_vectors), top_n, threshold)
#                 else:
#                     logging.info("未进行查询")
#             elif sub_choice == "2":
#                 logging.info("使用 system.json 中预设的 query")
#                 try:
#                     with open(SYSTEM_FILE, "r", encoding="utf-8") as f:
#                         json_tree = json.load(f)
#                     preset_queries = extract_leaf_nodes(json_tree.get("家族", {}))
#                     if preset_queries:
#                         matches = find_matches(preset_queries, top_n=top_n, threshold=threshold)
#                         save_results(matches, processed_count, len(embedding_vectors), top_n, threshold)
#                     else:
#                         logging.info("system.json 中没有预设的 query")
#                 except Exception as e:
#                     logging.error("读取 system.json 失败: %s", e)
#             else:
#                 logging.info("未选择任何查询模式，程序退出")
#         elif choice == "3":
#             logging.info("用户选择新业务：生成自我旁白")
#             make_inner_monologue_by_kimi()
#         else:
#             logging.info("未选择任何操作，程序退出")

import argparse
import logging
import json


def setup_logging(debug: bool):
    if debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logging.debug("DEBUG 模式已开启")


def load_or_build_index():
    current_index = load_index()
    if current_index.empty:
        total_articles, effective_count, current_index = build_index_from_articles(ARTICLES_FILE)
        logging.info("当前总文章数量: %d", total_articles)
        logging.info("当前有效文章数量: %d", effective_count)
        logging.info("索引表中共有 %d 篇文章", effective_count)
        current_index = save_index(current_index)
    else:
        logging.info("索引表中共有 %d 篇文章", len(current_index))
    return current_index


def update_and_log_index(current_index):
    current_index = update_index_from_metadata(current_index, metadata_cache)
    total_index_count = len(current_index)
    processed_count = len(current_index[current_index["processed"] == PROCESSED])
    logging.info("当前已建立索引的文章数量: %d", total_index_count)
    logging.info("当前已建立切片的文章数量: %d", processed_count)
    logging.info("当前未建立切片的文章数量: %d", total_index_count - processed_count)
    logging.info("当前总切片数量：%d", len(embedding_vectors))
    return processed_count


def handle_query(query: str, top_n: int, threshold: float, processed_count: int):
    logging.info("收到检索 query: %s", query)
    matches = find_matches([query], top_n=top_n, threshold=threshold)
    save_results(matches, processed_count, len(embedding_vectors), top_n, threshold)


def add_new_slices():
    logging.info("用户选择添加新的切片内容")
    current_index = load_index()
    unprocessed_articles = current_index[current_index["processed"] == UNPROCESSED]
    unprocessed_count = len(unprocessed_articles)
    print(f"当前还有 {unprocessed_count} 篇文章未处理。")

    sample_count = get_sample_count(unprocessed_count)
    load_articles(ARTICLES_FILE, sample_count=sample_count)
    current_index = load_index()
    
    top_n, threshold = get_top_n_threshold()

    sub_choice = input("请选择查询模式:\n1. 自定义 query\n2. 使用 system.json 中预设的 query\n请输入 1 或 2：").strip()
    processed_count = len(current_index[current_index["processed"] == PROCESSED])
    if sub_choice == "1":
        query = input("请输入检索 query: ").strip()
        if query:
            handle_query(query, top_n, threshold, processed_count)
        else:
            logging.info("未进行查询")
    elif sub_choice == "2":
        use_preset_queries(top_n, threshold, processed_count)
    else:
        logging.info("未选择任何查询模式，程序退出")


def use_preset_queries(top_n, threshold, processed_count):
    logging.info("使用 system.json 中预设的 query")
    try:
        with open(SYSTEM_FILE, "r", encoding="utf-8") as f:
            json_tree = json.load(f)
        preset_queries = extract_leaf_nodes(json_tree.get("家族", {}))
        if preset_queries:
            matches = find_matches(preset_queries, top_n=top_n, threshold=threshold)
            save_results(matches, processed_count, len(embedding_vectors), top_n, threshold)
        else:
            logging.info("system.json 中没有预设的 query")
    except Exception as e:
        logging.error("读取 system.json 失败: %s", e)


def get_sample_count(unprocessed_count):
    while True:
        try:
            sample_count = int(input(f"请输入要添加处理的文章数量（不超过未处理文章数量 {unprocessed_count}）：").strip())
            if 1 <= sample_count <= unprocessed_count:
                return sample_count
            else:
                print(f"请输入一个介于 1 到 {unprocessed_count} 的整数。")
        except ValueError:
            print("输入无效，请输入一个整数。")


def get_top_n_threshold():
    try:
        top_n = int(input("请输入保留的匹配数量（默认10）: ").strip() or 10)
    except:
        top_n = 10
    try:
        threshold = float(input("请输入相似度阈值（默认0.6）: ").strip() or 0.6)
    except:
        threshold = 0.6
    return top_n, threshold


def main():
    parser = argparse.ArgumentParser(description="文章检索与嵌入处理")
    parser.add_argument("-q", "--question", type=str, help="输入单个检索 query")
    parser.add_argument("--debug", action="store_true", help="是否显示 DEBUG 信息")
    args = parser.parse_args()

    setup_logging(args.debug)

    current_index = load_or_build_index()
    processed_count = update_and_log_index(current_index)

    if args.question:
        query = args.question
        top_n, threshold = get_top_n_threshold()
        handle_query(query, top_n, threshold, processed_count)
    else:
        print(f"{cache_dir}")
        print("请选择后续操作:")
        print("1. 添加新的切片内容（处理新文章）")
        print("2. 直接进行查询")
        print("3. 生成自我旁白（新业务）")
        choice = input("请输入 1, 2 或 3：").strip()
        if choice == "1":
            add_new_slices()
        elif choice == "2":
            logging.info("用户选择直接进行查询")
            sub_choice = input("请选择查询模式:\n1. 自定义 query\n2. 使用 system.json 中预设的 query\n请输入 1 或 2：").strip()
            top_n, threshold = get_top_n_threshold()
            if sub_choice == "1":
                query = input("请输入检索 query: ").strip()
                if query:
                    handle_query(query, top_n, threshold, processed_count)
                else:
                    logging.info("未进行查询")
            elif sub_choice == "2":
                use_preset_queries(top_n, threshold, processed_count)
            else:
                logging.info("未选择任何查询模式，程序退出")
        elif choice == "3":
            logging.info("用户选择新业务：生成自我旁白")
            make_inner_monologue_by_kimi()
        else:
            logging.info("未选择任何操作，程序退出")


if __name__ == "__main__":
    main()
