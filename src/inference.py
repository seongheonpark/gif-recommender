"""문장 → 감정 예측 + 키워드 추출."""

from typing import Dict, List, Tuple

import torch

from . import config
from .models import BERTSentenceTransform


def extract_keywords(
    sentence: str,
    okt,
    stopwords=config.STOPWORDS,
    pos_tags=config.KEYWORD_POS,
) -> Tuple[List[str], list]:
    """Okt 형태소 분석으로 명사/형용사/동사 키워드를 순서를 유지한 채 중복 없이 뽑는다."""
    morphs = okt.pos(sentence, stem=True)
    keywords = [
        word for word, pos in morphs
        if pos in pos_tags and len(word) > 1 and word not in stopwords
    ]
    return list(dict.fromkeys(keywords)), morphs


def predict_emotion_kobert(sentence, model, tokenizer, max_len=config.KOBERT_MAX_LEN):
    device = next(model.parameters()).device
    model.eval()

    transform = BERTSentenceTransform(tokenizer, max_seq_length=max_len, pad=True, pair=False)
    input_ids, valid_length, segment_ids = transform([sentence])

    input_ids = torch.from_numpy(input_ids).unsqueeze(0).to(device)
    segment_ids = torch.from_numpy(segment_ids).unsqueeze(0).to(device)
    valid_length = torch.tensor([int(valid_length)], dtype=torch.int64).to(device)

    with torch.no_grad():
        logits = model(input_ids, valid_length, segment_ids)
        probs = torch.softmax(logits, dim=-1)[0]

    pred_id = int(torch.argmax(probs))
    return pred_id, config.IDX2LABEL[pred_id], float(probs[pred_id])


def predict_emotion_koelectra(sentence, model, tokenizer, max_len=config.KOELECTRA_MAX_LEN):
    device = next(model.parameters()).device
    model.eval()

    encoded = tokenizer(
        sentence,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=max_len,
    )
    encoded = {k: v.to(device) for k, v in encoded.items()}

    with torch.no_grad():
        logits = model(**encoded).logits
        probs = torch.softmax(logits, dim=-1)[0]

    pred_id = int(torch.argmax(probs))
    return pred_id, config.IDX2LABEL[pred_id], float(probs[pred_id])


def predict_emotion(sentence, backend, model, tokenizer, max_len):
    if backend == "KoBERT":
        return predict_emotion_kobert(sentence, model, tokenizer, max_len=max_len)
    if backend == "KoELECTRA":
        return predict_emotion_koelectra(sentence, model, tokenizer, max_len=max_len)
    raise ValueError(f"알 수 없는 backend: {backend!r}")


def analyze_sentence(sentence, backend, model, tokenizer, okt, max_len) -> Dict:
    """감정 예측과 키워드 추출 결과를 하나의 dict로 반환한다."""
    pred_id, pred_label, pred_prob = predict_emotion(sentence, backend, model, tokenizer, max_len)
    keywords, morphs = extract_keywords(sentence, okt)

    return {
        "sentence": sentence,
        "emotion_id": pred_id,
        "emotion_label": pred_label,
        "emotion_prob": pred_prob,
        "keywords": keywords,
        "morphs_raw": morphs,
    }
