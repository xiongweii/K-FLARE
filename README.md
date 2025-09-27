# K-FLARE

K-FLARE: Knowledge-enhanced Fake news and misinformation detection through Language model Attention and Rumor Evidence

## Dataset Preparation

Please configure the dataset path in `config.py` first, then follow these steps for preparation:

### Supported Datasets

1. **Twitter15/16**: Twitter rumor detection datasets
2. **Pheme**: Multi-event rumor detection dataset

### Dataset Structure

```
data/
├── twitter15/
│   ├── label.txt          # Label file
│   ├── source_tweets.txt  # Source tweet content
│   └── tree/             # Propagation tree files
│       ├── <tweet_id>.txt
│       └── ...
├── twitter16/             # Same structure as above
└── pheme/                 # Process using pheme_data_processor.py
```

### Data Preprocessing

For the Pheme dataset, run the preprocessing script separately:

```bash
python pheme_data_processor.py
```

## Additional Documentation

More detailed documentation including model architecture, training procedures, evaluation metrics, and usage examples will be coming soon.