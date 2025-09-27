# K-FLARE
K-FLARE
## 数据集准备
请先在config.py中配置数据集路径，然后按照以下步骤进行准备：
### 支持的数据集

1. **Twitter15/16**: 推特谣言检测数据集
2. **Pheme**: 多事件谣言检测数据集

### 数据集结构

```
data/
├── twitter15/
│   ├── label.txt          # 标签文件
│   ├── source_tweets.txt  # 源推文内容
│   └── tree/             # 传播树文件
│       ├── <tweet_id>.txt
│       └── ...
├── twitter16/             # 同上结构
└── pheme/                 # 使用pheme_data_processor.py处理
```
### 数据预处理

对于Pheme数据集，需要单独运行预处理脚本：

```bash
python pheme_data_processor.py
```