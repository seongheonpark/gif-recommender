"""학습용 데이터셋 클래스와 train/valid 분할."""

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

from . import config
from .models import BERTSentenceTransform


def load_splits(
    path: Optional[Path] = None,
    test_size: float = 0.2,
    seed: int = config.SEED,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """전처리된 data.csv(기본: data/processed/data.csv)를 읽어 (sentence, emotion[int]) 형태로 층화 분할한다."""
    path = Path(path or config.PROCESSED_DATA_PATH)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} 가 없습니다. 먼저 `python -m scripts.preprocess` 를 실행하세요."
        )

    df = pd.read_csv(path).rename(columns={"Sentence": "sentence", "Emotion": "emotion"})
    if df["emotion"].dtype == object:
        df["emotion"] = df["emotion"].map(config.TARGET_CLASSES).astype(int)

    train_df, valid_df = train_test_split(
        df, test_size=test_size, random_state=seed, stratify=df["emotion"]
    )
    for name, split in (("train", train_df), ("valid", valid_df)):
        assert split["emotion"].between(0, config.NUM_LABELS - 1).all(), f"{name} 라벨 범위 오류"
    return train_df, valid_df


class EmotionDataset(Dataset):
    """HuggingFace 토크나이저(KoELECTRA 등)용 데이터셋."""

    def __init__(self, sentences, labels, tokenizer, max_len=config.KOELECTRA_MAX_LEN):
        self.sentences = sentences.reset_index(drop=True)
        self.labels = labels.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        encoded = self.tokenizer(
            str(self.sentences.iloc[idx]),
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt",
        )
        item = {k: v.squeeze(0) for k, v in encoded.items()}
        item["labels"] = torch.tensor(int(self.labels.iloc[idx]), dtype=torch.long)
        return item


class BERTDataset(Dataset):
    """KoBERT용 데이터셋: (input_ids, valid_length, segment_ids, label)."""

    def __init__(self, sentences, labels, tokenizer, max_len=config.KOBERT_MAX_LEN):
        transform = BERTSentenceTransform(tokenizer, max_seq_length=max_len, pad=True, pair=False)
        self.sentences = [transform([str(s)]) for s in sentences]
        self.labels = [np.int32(label) for label in labels]

    def __getitem__(self, i):
        return self.sentences[i] + (self.labels[i],)

    def __len__(self):
        return len(self.labels)
