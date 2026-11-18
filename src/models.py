"""KoBERT / KoELECTRA 감정 분류 모델 정의와 로딩 유틸."""

from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from torch import nn
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BertConfig,
    BertModel,
)

from . import config

BACKENDS = ("KoBERT", "KoELECTRA")


############################################
# KoBERT 입력 변환 / 분류기
############################################

class BERTSentenceTransform:
    """문장을 KoBERT 입력 형식 (input_ids, valid_length, segment_ids)으로 변환한다."""

    def __init__(self, tokenizer, max_seq_length, pad=True, pair=False):
        self._tokenizer = tokenizer
        self._max_seq_length = max_seq_length
        self._pad = pad
        self._pair = pair

        self.cls_token = self._tokenizer.cls_token or "[CLS]"
        self.sep_token = self._tokenizer.sep_token or "[SEP]"
        self.pad_token_id = self._tokenizer.pad_token_id

    def _truncate_seq_pair(self, tokens_a, tokens_b, max_length):
        while True:
            total_length = len(tokens_a) + len(tokens_b)
            if total_length <= max_length:
                break
            if len(tokens_a) > len(tokens_b):
                tokens_a.pop()
            else:
                tokens_b.pop()

    def __call__(self, line):
        text_a = line[0]
        tokens_a = self._tokenizer.tokenize(text_a)
        tokens_b = None

        if self._pair:
            assert len(line) == 2
            text_b = line[1]
            tokens_b = self._tokenizer.tokenize(text_b)

        if tokens_b:
            self._truncate_seq_pair(tokens_a, tokens_b, self._max_seq_length - 3)
        else:
            if len(tokens_a) > self._max_seq_length - 2:
                tokens_a = tokens_a[:self._max_seq_length - 2]

        tokens = [self.cls_token, *tokens_a, self.sep_token]
        segment_ids = [0] * len(tokens)

        if tokens_b:
            tokens.extend(tokens_b)
            tokens.append(self.sep_token)
            segment_ids.extend([1] * (len(tokens) - len(segment_ids)))

        input_ids = self._tokenizer.convert_tokens_to_ids(tokens)
        valid_length = len(input_ids)

        if self._pad:
            padding_length = self._max_seq_length - valid_length
            input_ids.extend([self.pad_token_id] * padding_length)
            segment_ids.extend([0] * padding_length)

        return (
            np.array(input_ids, dtype="int32"),
            np.array(valid_length, dtype="int32"),
            np.array(segment_ids, dtype="int32"),
        )


class BERTClassifier(nn.Module):
    """KoBERT pooler 출력 위에 선형 분류기를 얹은 감정 분류 모델."""

    def __init__(self, bert, hidden_size=768, num_classes=config.NUM_LABELS, dr_rate=None):
        super().__init__()
        self.bert = bert
        self.dr_rate = dr_rate

        self.classifier = nn.Linear(hidden_size, num_classes)
        if dr_rate:
            self.dropout = nn.Dropout(p=dr_rate)

    def gen_attention_mask(self, token_ids, valid_length):
        attention_mask = torch.zeros_like(token_ids)
        for i, v in enumerate(valid_length):
            attention_mask[i][:v] = 1
        return attention_mask.float()

    def forward(self, token_ids, valid_length, segment_ids):
        attention_mask = self.gen_attention_mask(token_ids, valid_length)
        _, pooler = self.bert(
            input_ids=token_ids,
            token_type_ids=segment_ids.long(),
            attention_mask=attention_mask.to(token_ids.device),
            return_dict=False,
        )
        out = self.dropout(pooler) if self.dr_rate else pooler
        return self.classifier(out)


############################################
# 모델 / 토크나이저 생성
############################################
# pretrained=False 이면 config만 받아 구조를 만든다.
# 파인튜닝된 state_dict를 바로 덮어쓸 때(추론) 사전학습 가중치를 따로 받을 필요가 없다.

def build_kobert(pretrained: bool = True, dr_rate: float = config.KOBERT_DROPOUT) -> BERTClassifier:
    if pretrained:
        bert = BertModel.from_pretrained(config.KOBERT_PRETRAINED)
    else:
        bert = BertModel(BertConfig.from_pretrained(config.KOBERT_PRETRAINED))
    return BERTClassifier(bert, num_classes=config.NUM_LABELS, dr_rate=dr_rate)


def load_kobert_tokenizer():
    from kobert_tokenizer import KoBERTTokenizer

    return KoBERTTokenizer.from_pretrained(config.KOBERT_PRETRAINED)


def build_koelectra(pretrained: bool = True):
    if pretrained:
        return AutoModelForSequenceClassification.from_pretrained(
            config.KOELECTRA_PRETRAINED, num_labels=config.NUM_LABELS
        )
    model_config = AutoConfig.from_pretrained(
        config.KOELECTRA_PRETRAINED, num_labels=config.NUM_LABELS
    )
    return AutoModelForSequenceClassification.from_config(model_config)


def load_koelectra_tokenizer(use_fast: bool = True):
    return AutoTokenizer.from_pretrained(config.KOELECTRA_PRETRAINED, use_fast=use_fast)


############################################
# 학습된 가중치 로드
############################################

def weights_filename(backend: str) -> str:
    if backend == "KoBERT":
        return config.KOBERT_WEIGHTS
    if backend == "KoELECTRA":
        return config.KOELECTRA_WEIGHTS
    raise ValueError(f"알 수 없는 backend: {backend!r} (가능한 값: {BACKENDS})")


def resolve_weights(filename: str) -> Path:
    """models/ 에 가중치가 있으면 그 경로를, 없으면 Hugging Face Hub에서 받아 경로를 반환한다."""
    local_path = config.MODELS_DIR / filename
    if local_path.exists():
        return local_path

    from huggingface_hub import hf_hub_download

    downloaded = hf_hub_download(
        repo_id=config.HF_MODEL_REPO,
        filename=filename,
        local_dir=config.MODELS_DIR,
    )
    return Path(downloaded)


def load_finetuned_model(backend: str, device: torch.device) -> Tuple[nn.Module, object]:
    """학습된 감정 분류 모델과 토크나이저를 불러온다 (추론용, eval 모드)."""
    weights_path = resolve_weights(weights_filename(backend))

    if backend == "KoBERT":
        model = build_kobert(pretrained=False)
        tokenizer = load_kobert_tokenizer()
    else:
        model = build_koelectra(pretrained=False)
        tokenizer = load_koelectra_tokenizer()

    state_dict = torch.load(weights_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model, tokenizer
