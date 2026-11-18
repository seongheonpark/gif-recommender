"""프로젝트 전역 설정: 경로, 라벨, 모델 이름, 하이퍼파라미터 기본값."""

import os
from pathlib import Path

############################################
# 경로
############################################
ROOT_DIR = Path(__file__).resolve().parent.parent

try:  # 프로젝트 루트의 .env 를 환경변수로 읽는다 (GIPHY_API_KEY, HF_MODEL_REPO 등)
    from dotenv import load_dotenv

    load_dotenv(ROOT_DIR / ".env")
except ImportError:
    pass

MODELS_DIR = ROOT_DIR / "models"
ASSETS_DIR = ROOT_DIR / "assets"
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_PATH = DATA_DIR / "processed" / "data.csv"
OUTPUTS_DIR = ROOT_DIR / "outputs"

############################################
# 감정 라벨
############################################
TARGET_CLASSES = {
    "공포": 0,
    "놀람": 1,
    "분노": 2,
    "슬픔": 3,
    "중립": 4,
    "행복": 5,
    "혐오": 6,
}
IDX2LABEL = {v: k for k, v in TARGET_CLASSES.items()}
LABEL_NAMES = [IDX2LABEL[i] for i in range(len(IDX2LABEL))]
NUM_LABELS = len(TARGET_CLASSES)

EMOTION_TO_EN = {
    "공포": "scared",
    "놀람": "surprised",
    "분노": "angry",
    "슬픔": "sad",
    "중립": "neutral",
    "행복": "happy",
    "혐오": "disgust",
}

############################################
# 모델
############################################
KOBERT_PRETRAINED = "skt/kobert-base-v1"
KOELECTRA_PRETRAINED = "monologg/koelectra-base-v3-discriminator"

KOBERT_WEIGHTS = "kobert_emotion_model_state_dict.pt"
KOELECTRA_WEIGHTS = "koelectra_emotion_model_state_dict.pt"

KOBERT_MAX_LEN = 100
KOELECTRA_MAX_LEN = 64
KOBERT_DROPOUT = 0.5

# 학습된 가중치(.pt)를 올려둔 Hugging Face Hub 모델 저장소
# 다른 저장소를 쓰려면 환경변수 HF_MODEL_REPO 로 덮어쓰면 된다.
HF_MODEL_REPO = os.getenv("HF_MODEL_REPO") or "qkrjds201/korean-emotion-gif-recommender"

############################################
# 키워드 추출
############################################
KEYWORD_POS = {"Noun", "Adjective", "Verb"}
STOPWORDS = {"하다", "되다", "있다", "그냥", "정말", "진짜", "근데"}

############################################
# GIPHY
############################################
GIPHY_SEARCH_URL = "https://api.giphy.com/v1/gifs/search"

SEED = 42
