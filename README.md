
# RAG_Caritas
针对4007篇Caritas文章进行切片和向量化嵌入。通过余弦相似度进行相似度召回。利用召回的片段进行阅读或生产。

1. 配置`.env`
```
KIMI_API_KEY = 
KIMI_URL
ZHIPU_API_KEY = 
ZHIPU_URL
SiliconFlow_API_KEY = 
SiliconFlow_URL = 
```

2. 下载数据集：[百度网盘链接]()
放置到`./data/`
```
(base) szy@aw:~/2025/github/RAG_Caritas$ tree
.
├── Makefile
├── README.md
├── cache
├── data
│   ├── 0315句子更新 - 汇总表.csv
│   ├── articles.json
│   ├── example.md
│   └── system.json
├── kimi_api.py
├── logs
├── main.py
├── requirements.txt
└── test
```

3. 配置环境
```bash
pip install -r requirements.txt
```

4. 运行程序
```bash
python main.py
```





