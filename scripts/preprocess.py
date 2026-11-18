"""원본 데이터셋 3종을 정리·결합해 data/processed/data.csv 를 만든다.

사용법 (프로젝트 루트에서):
    python -m scripts.preprocess

필요한 원본 파일 (data/raw/ 에 위치, 다운로드 방법은 data/README.md 참고):
    - 감성대화말뭉치(최종데이터)_Training.xlsx
    - 감성대화말뭉치(최종데이터)_Validation.xlsx
    - 한국어_연속적_대화_데이터셋.xlsx
    - 혐오 데이터.csv
"""

import argparse
from pathlib import Path

import pandas as pd

from src import config

EMOTION_CORPUS_FILES = [
    "감성대화말뭉치(최종데이터)_Training.xlsx",
    "감성대화말뭉치(최종데이터)_Validation.xlsx",
]
CONTINUOUS_DIALOG_FILE = "한국어_연속적_대화_데이터셋.xlsx"
HATE_SPEECH_FILE = "혐오 데이터.csv"

# 감성 대화 말뭉치: 감정_소분류 → 7개 감정 대분류로 재매핑
SUBCATEGORY_TO_EMOTION = {
    **dict.fromkeys(
        ["만족스러운", "편안한", "신뢰하는", "안도", "기쁨", "감사하는", "신이 난", "자신하는", "느긋"],
        "행복",
    ),
    **dict.fromkeys(
        ["고립된", "혼란스러운", "두려운", "불안", "초조한", "당혹스러운"],
        "공포",
    ),
    **dict.fromkeys(
        ["좌절한", "눈물이 나는", "우울한", "슬픔", "상처", "실망한", "후회되는",
         "비통한", "낙담한", "버려진", "희생된"],
        "슬픔",
    ),
}

# 감성 대화 말뭉치: 7개 감정 어디에도 명확히 속하지 않아 제외하는 소분류
DROPPED_SUBCATEGORIES = [
    "걱정스러운", "스트레스 받는", "노여워하는", "툴툴대는", "성가신", "한심한",
    "충격 받은", "안달하는", "배신당한", "질투하는", "괴로워하는", "외로운",
    "조심스러운", "회의적인", "취약한", "마비된", "염세적인", "열등감",
    "방어적인", "흥분", "억울한", "죄책감의", "가난한, 불우한", "부끄러운",
    "당황", "남의 시선을 의식하는",
]

# 연속적 대화 데이터셋: 축약/오타 라벨 정리
LABEL_TYPO_FIXES = {
    "ㅍ": "공포",
    "분": "분노",
    "분ㄴ": "분노",
    "ㅈ중립": "중립",
    "중림": "중립",
    "ㄴ중립": "중립",
    "줄": "중립",
}


def load_emotion_corpus(raw_dir: Path) -> pd.DataFrame:
    """감성 대화 말뭉치 (AI Hub)."""
    corpus = pd.concat(
        [pd.read_excel(raw_dir / f, engine="openpyxl") for f in EMOTION_CORPUS_FILES],
        axis=0,
        ignore_index=True,
    )

    sub = corpus["감정_소분류"]
    mapped = sub.map(SUBCATEGORY_TO_EMOTION)
    corpus.loc[mapped.notna(), "감정_대분류"] = mapped[mapped.notna()]
    corpus = corpus[~sub.isin(DROPPED_SUBCATEGORIES)]

    corpus = corpus.rename(columns={"사람문장1": "Sentence", "감정_대분류": "Emotion"})
    return corpus[["Sentence", "Emotion"]].reset_index(drop=True)


def load_continuous_dialog(raw_dir: Path) -> pd.DataFrame:
    """한국어 감정 정보가 포함된 연속적 대화 데이터셋 (AI Hub)."""
    data = pd.read_excel(raw_dir / CONTINUOUS_DIALOG_FILE, engine="openpyxl")
    data = data[["Unnamed: 1", "Unnamed: 2"]].drop(index=0)  # 첫 행은 헤더
    data = data.rename(columns={"Unnamed: 1": "Sentence", "Unnamed: 2": "Emotion"})
    data["Emotion"] = data["Emotion"].replace(LABEL_TYPO_FIXES)
    return data.dropna(how="any")


def load_hate_speech(raw_dir: Path) -> pd.DataFrame:
    """한국어 혐오 표현 데이터셋 (smilegate-ai/korean_unsmile_dataset)."""
    data = pd.read_csv(raw_dir / HATE_SPEECH_FILE, encoding="cp949")
    data = data.loc[data["혐오"] == 1, ["문장"]].rename(columns={"문장": "Sentence"})
    data["Emotion"] = "혐오"
    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw-dir", type=Path, default=config.RAW_DATA_DIR)
    parser.add_argument("--out", type=Path, default=config.PROCESSED_DATA_PATH)
    args = parser.parse_args()

    data = pd.concat(
        [
            load_emotion_corpus(args.raw_dir),
            load_continuous_dialog(args.raw_dir),
            load_hate_speech(args.raw_dir),
        ],
        ignore_index=True,
    )

    unknown = set(data["Emotion"].unique()) - set(config.TARGET_CLASSES)
    if unknown:
        raise ValueError(f"정의되지 않은 감정 라벨이 있습니다: {sorted(map(str, unknown))}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(args.out, index=False)

    print(f"저장 완료: {args.out} ({len(data):,}개 문장)")
    print(data["Emotion"].value_counts().to_string())


if __name__ == "__main__":
    main()
