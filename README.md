
# RAG_Caritas
针对4007篇Caritas文章进行切片和向量化嵌入。通过余弦相似度进行相似度召回。利用召回的片段进行阅读或生产。

1. 配置`.env`
```
KIMI_API_KEY = 
ZHIPU_API_key = 
API_URL_ZHIPU=
API_URL_siliconflow = 
API_KEY_siliconflow = 

```

2. 下载数据集：[百度网盘链接](https://pan.baidu.com/s/1awbAjIMVueLZez9BRmVGqw)
内部使用，密码暂不公开
放置到`./data/`
```
(base) szy@aw:~/2025/github/RAG_Caritas$ tree
szy@aw ~/2/g/RAG_Caritas (main)> tree                                                                                                           (Caritas) 
.
├── Makefile
├── README.md
├── cache
│   ├── bge-m3-50
│   │   ├── embedding_vectors.json
│   │   ├── index.csv
│   │   ├── metadata_cache.json
│   │   ├── system_embedding_vectors.json
│   │   └── system_metadata_cache.json
│   └── embedding-3-50
│       ├── embedding_vectors.json
│       ├── fail.csv
│       ├── index.csv
│       ├── metadata_cache.json
│       ├── system_embedding_vectors.json
│       └── system_metadata_cache.json
├── data
│   ├── 0315句子更新 - 汇总表.csv
│   ├── articles.json
│   ├── example.md
│   └── system.json
├── kimi_api.py
├── logs
├── main.py
├── output
├── requirements.txt
└── test
    └── clean_metadata.py
```

3. 配置环境
```bash
pip install -r requirements.txt
```

4. 运行程序
```bash
python main.py
```


---
下一步计划：
1. 不采用任何外部材料，根据prompt生成reponse
2. 使用reponse，作为Query召回相关材料
3. 过滤掉不相关的材料
4. 使用材料改写reponse


