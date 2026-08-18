# Canonical op alias maps per domain. Extend freely.

TECHNICAL = {
    # RSI
    "rsi": "technical.rsi",
    "RSI": "technical.rsi",
    "technical.RSI": "technical.rsi",
    "technical.rsi": "technical.rsi",
    # MACD
    "macd": "technical.macd",
    "MACD": "technical.macd",
    "technical.MACD": "technical.macd",
    "technical.macd": "technical.macd",
    # ATR
    "atr": "technical.atr",
    "ATR": "technical.atr",
    "technical.ATR": "technical.atr",
    "technical.atr": "technical.atr",
    # VWAP
    "vwap": "technical.vwap",
    "VWAP": "technical.vwap",
    "technical.VWAP": "technical.vwap",
    "technical.vwap": "technical.vwap",
    # Bollinger
    "bollinger": "technical.bollinger_bands",
    "bollinger_bands": "technical.bollinger_bands",
    "BollingerBands": "technical.bollinger_bands",
    "technical.BollingerBands": "technical.bollinger_bands",
    "technical.bollinger_bands": "technical.bollinger_bands",
}

SCALING = {
    "zscore": "normalize.zscore",
    "standard": "normalize.zscore",
    "standardize": "normalize.zscore",
    "normalize.zscore": "normalize.zscore",
    "minmax": "normalize.minmax",
    "min_max": "normalize.minmax",
    "normalize.minmax": "normalize.minmax",
    "robust": "normalize.robust",
    "normalize.robust": "normalize.robust",
    "clip": "transform.clip",
    "transform.clip": "transform.clip",
    "winsorize": "transform.winsorize",
    "transform.winsorize": "transform.winsorize",
}

SENTIMENT = {
    # Sentiment
    "finbert_sentiment": "nlp.sentiment.finbert",
    "finbert_sentiment_gpu": "nlp.sentiment.finbert",
    "nlp.sentiment.finbert": "nlp.sentiment.finbert",
    "sentiment.hf": "nlp.sentiment.hf",
    "nlp.sentiment.hf": "nlp.sentiment.hf",
    # ESG
    "esg_normalizer": "esg.normalize",
    "esg_normalizer_gpu": "esg.normalize",
    "esg.normalize": "esg.normalize",
    # Cleaning & columns
    "text.clean": "text.clean_ascii",
    "text.clean_ascii": "text.clean_ascii",
    "column.create": "column.create",
    "columns.create": "column.create",
    # Transforms
    "clip": "transform.clip",
    "transform.clip": "transform.clip",
    "normalize.minmax": "normalize.minmax",
    "normalize.zscore": "normalize.zscore",
}

TEMPORAL = {
    "lag": "time.lag",
    "lags": "time.lag",
    "time.lag": "time.lag",
    "session_flag": "time.session_flag",
    "session_flags": "time.session_flag",
    "time.session_flag": "time.session_flag",
    "calendar": "time.calendar",
    "time.calendar": "time.calendar",
    "bucket": "time.bucket",
    "time.bucket": "time.bucket",
}

SEQUENCE = {
    "sequence.window": "sequence.window",
    "sequence.make": "sequence.window",
    "sequence.build": "sequence.window",
    "seq.window": "sequence.window",
    "seq.make": "sequence.window",
    "seq.as_strided": "sequence.window",
    "sequence.as_strided": "sequence.window",
    "target.shift": "target.shift",
    "seq.target.shift": "target.shift",
}

EXPLAIN = {
    # Kernel SHAP (CPU/GPU-agnostic; backend decides device)
    "shap_kernel": "explain.shap.kernel",
    "kernel_shap": "explain.shap.kernel",
    "explain.shap.kernel": "explain.shap.kernel",
    # TreeSHAP (GPU/cuML-backed or CPU fallback handled in backends)
    "cuml_tree_shap": "explain.shap.treeshap",
    "tree_shap": "explain.shap.treeshap",
    "explain.shap.treeshap": "explain.shap.treeshap",
    "explain.shap.tree": "explain.shap.treeshap",
}

EMBEDDING = {
    # Sentence-Transformers / SBERT family
    "hf_embedding": "nlp.embed.sentence_transformers",
    "sentence_transformers": "nlp.embed.sentence_transformers",
    "sbert": "nlp.embed.sentence_transformers",
    "nlp.embed.sentence_transformers": "nlp.embed.sentence_transformers",
    # Optional generic embedding op, if you support multiple providers
    "nlp.embed": "nlp.embed",
}

TOPIC = {
    # BERTopic (backend decides UMAP/HDBSCAN impls & device)
    "bertopic": "nlp.topic.bertopic",
    "topic.bertopic": "nlp.topic.bertopic",
    "nlp.topic.bertopic": "nlp.topic.bertopic",
    # Optional post-processing op for probability masking
    "topic.filter_prob": "topic.filter_prob",
    "filter_prob": "topic.filter_prob",
}
