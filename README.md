# Korean Emotion GIF Recommender

BERT 기반 한국어 감성 분석·키워드 추출을 이용한 GIF 추천 시스템

## 프로젝트 소개
**KoBERT/KoELECTRA 기반 한국어 감정 분류 모델**과 **키워드 추출 로직**을 결합하여, 메신저 사용자의 입력 문장에 어울리는 **GIF를 자동으로 추천해주는 시스템**입니다.

<p align="center">
  <img src="docs/images/preview-1.png">
</p>
<p align="center">
  <img src="docs/images/preview-2.png">
</p>

### 차별점

- **대화 기반 자동 추천**: 메시지를 입력하면 감정 분석 + 키워드 추출로 GIF를 자동 추천
- **세분화된 감정 기반 추천**: 공포·놀람·분노·슬픔·중립·행복·혐오 7가지 감정을 구분
- **플랫폼·콘텐츠 제약 적음**: GIPHY의 방대한 무료 GIF 리소스 활용
- **한국어 대화 특화 모델**: KoBERT/KoELECTRA로 한국어 문장의 감정을 예측

### 시스템 아키텍처

```mermaid
flowchart TB
    A[사용자 입력 문장] --> B["텍스트 전처리"]
    B --> C1["감성 분석 모듈<br/>(KoBERT/KoELECTRA)"]
    B --> C2["키워드 추출 모듈<br/>(Konlpy Okt)"]

    C1 --> D1["감정 레이블<br/>(공포/놀람/분노/슬픔/중립/행복/혐오)"]
    C2 --> D2["핵심 키워드 리스트"]

    D1 --> E["GIF 검색 쿼리 생성<br/>(감정 + 키워드 조합)"]
    D2 --> E

    E --> F["GIF API 호출<br/>(GIPHY)"]
    F --> G["후처리 및 랭킹"]
    G --> H[사용자에게 GIF 추천 결과 반환]
```

## 목차
- [0. 주제 선정 배경](#0-주제-선정-배경)
- [1. 데이터 & 전처리](#1-데이터--전처리)
- [2. 모델 학습](#2-모델-학습)
- [3. 모델 평가](#3-모델-평가)
- [4. 감정 + 키워드 기반 GIF 추천 파이프라인](#4-감정--키워드-기반-gif-추천-파이프라인)
- [5. Streamlit 기반 웹 데모 앱(app.py) 구현](#5-streamlit-기반-웹-데모-앱apppy-구현)
- [6. 프로젝트 구조](#6-프로젝트-구조)
- [7. 실행 방법 (Quick Start)](#7-실행-방법-quick-start)
- [8. 직접 학습하기](#8-직접-학습하기)

## 0. 주제 선정 배경
카카오톡, 인스타그램과 같은 플랫폼 비즈니스는 사용자들이 서비스 안에 더 오래 머물고, 더 자주 상호작용하도록 만드는 것을 핵심 목표로 삼는다. 이러한 환경에서 사용자는 텍스트를 이용한 언어적 소통뿐만 아니라, 이모티콘·스티커·GIF와 같은 비언어적 표현 수단도 활발하게 사용한다. 최근 카카오톡에 도입된 이모티콘 추천 기능, 인스타그램 다이렉트 메시지(DM)의 GIPHY 연동 기능은 이러한 흐름을 잘 보여주는 대표적 사례다.

카카오톡의 이모티콘 추천 기능은 사용자가 입력한 문장을 분석해 이모티콘을 추천해 준다는 점에서 편의성과 대화 몰입도를 높이는 장점이 있다. 그러나 추천 대상이 카카오톡 내 유료 이모티콘에 제한된다는 한계가 있다. 이로 인해, 실제 사용자의 감정이나 상황에 더 잘 맞는 다양한 표현을 충분히 활용하기 어렵다.

반면 인스타그램 DM에 연동된 GIPHY는 방대한 양의 GIF를 제공하지만, 현재는 사용자가 직접 키워드를 검색하는 방식이고 대화 내용이나 감정 상태를 분석해 GIF를 추천해 주는 기능은 부재하다. 즉, 카카오톡은 추천 기능은 있으나 표현 수단과 범위가 제한적이고, GIPHY는 표현 수단은 풍부하지만 추천 인텔리전스가 부족한 상황인 것이다.

따라서 사용자의 대화를 이해하고 그 속에 담긴 감정과 키워드를 파악한 뒤, 이를 기반으로 적절한 GIF를 자동으로 제안해 주는 시스템이 생긴다면 카카오톡과 인스타그램이 가진 장점을 결합하면서도 기존 한계를 보완할 수 있을 것이라고 생각했다. 이에 따라, 본 프로젝트에서는 BERT/ELECTRA 기반 감정 분석과 키워드 추출 로직을 바탕으로 메신저 대화에 어울리는 GIF를 추천하는 시스템을 설계하고 구현하는 것을 목표로 한다.

## 1. 데이터 & 전처리
### 사용한 데이터
- 한국어 감정 정보가 포함된 연속적 대화 데이터셋
(https://aihub.or.kr/aihubdata/data/view.do?dataSetSn=271)
- 감성 대화 말뭉치
(https://aihub.or.kr/aihubdata/data/view.do?dataSetSn=86)
- 한국어 혐오 표현 데이터셋
(https://github.com/smilegate-ai/korean_unsmile_dataset)

'한국어 감정 정보가 포함된 연속적 대화 데이터셋'만으로 감정 분류 모델을 학습하려고 했으나, 아래와 같이 특정 감정에 데이터가 쏠려 있는 클래스 불균형이 있어 추가 데이터(감성 대화 말뭉치, 혐오 표현 데이터셋)를 결합하기로 결정했다.


| 감정   | 개수   |
|--------|-------:|
| 행복   | 1,030  |
| 중립   | 43,786 |
| 슬픔   | 1,972  |
| 공포   | 98     |
| 혐오   | 220    |
| 분노   | 3,628  |
| 놀람   | 4,866  |

### 1) 연속적 대화 데이터셋 전처리

- 축약/오타 레이블 정리

```python
# scripts/preprocess.py
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
```

- (Sentence, Emotion) 형태 정리

```python
# scripts/preprocess.py
def load_continuous_dialog(raw_dir: Path) -> pd.DataFrame:
    """한국어 감정 정보가 포함된 연속적 대화 데이터셋 (AI Hub)."""
    data = pd.read_excel(raw_dir / CONTINUOUS_DIALOG_FILE, engine="openpyxl")
    data = data[["Unnamed: 1", "Unnamed: 2"]].drop(index=0)  # 첫 행은 헤더
    data = data.rename(columns={"Unnamed: 1": "Sentence", "Unnamed: 2": "Emotion"})
    data["Emotion"] = data["Emotion"].replace(LABEL_TYPO_FIXES)
    return data.dropna(how="any")
```

### 2) 감성 대화 말뭉치 전처리

- Training / Validation 두 파일을 합쳐서 사용

```python
# scripts/preprocess.py
EMOTION_CORPUS_FILES = [
    "감성대화말뭉치(최종데이터)_Training.xlsx",
    "감성대화말뭉치(최종데이터)_Validation.xlsx",
]
```

- 감정_소분류 → 감정 재매핑
  - **행복**

    `만족스러운`, `편안한`, `신뢰하는`, `안도`, `기쁨`, `감사하는`, `신이 난`, `자신하는`, `느긋`
  - **공포**

    `고립된`, `혼란스러운`, `두려운`, `불안`, `초조한`, `당혹스러운`
  - **슬픔**

    `좌절한`, `눈물이 나는`, `우울한`, `슬픔`, `상처`, `실망한`, `후회되는`, `비통한`, `낙담한`, `버려진`, `희생된`
  - **제외**

    7개 감정 어디에도 명확히 속하지 않는 소분류(`걱정스러운`, `스트레스 받는`, `노여워하는`, `툴툴대는` 등)는 데이터에서 제거
  - 위 매핑에 해당하지 않는 소분류는 원래의 `감정_대분류` 값을 그대로 사용

```python
# scripts/preprocess.py
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
```

- (Sentence, Emotion) 형태 정리 — `사람문장1`을 문장으로, 재매핑된 `감정_대분류`를 감정으로 쓰고 나머지 컬럼(연령, 성별, 신체질환, 상황키워드, 시스템문장 등)은 버린다.

```python
# scripts/preprocess.py
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
```

### 3) 혐오 표현 데이터셋 전처리

- 혐오 문장만 필터링 (`혐오` 컬럼이 1인 문장)
- (Sentence, Emotion) 형태 정리 — 감정 라벨은 모두 `'혐오'`

```python
# scripts/preprocess.py
def load_hate_speech(raw_dir: Path) -> pd.DataFrame:
    """한국어 혐오 표현 데이터셋 (smilegate-ai/korean_unsmile_dataset)."""
    data = pd.read_csv(raw_dir / HATE_SPEECH_FILE, encoding="cp949")
    data = data.loc[data["혐오"] == 1, ["문장"]].rename(columns={"문장": "Sentence"})
    data["Emotion"] = "혐오"
    return data
```

### 4) 데이터 결합

위의 데이터들을 결합하여 최종적으로 클래스 불균형을 완화한 통합 데이터프레임 `data`를 완성했다.

| 감정   | 개수   |
|--------|-------:|
| 행복   | 7,725  |
| 중립   | 40,813 |
| 슬픔   | 15,144 |
| 공포   | 6,226  |
| 혐오   | 11,457 |
| 분노   | 9,790  |
| 놀람   | 3,979  |

결합한 데이터에 7개 감정 외의 라벨이 남아 있으면 오류를 내도록 검사한 뒤, `data/processed/data.csv`(`Sentence`, `Emotion` 컬럼)로 저장한다.

```python
# scripts/preprocess.py
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
```

## 2. 모델 학습

### 1) KoBERT 기반 감정 분류 모델
전처리된 통합 데이터셋(data.csv)을 바탕으로, 먼저 KoBERT 기반 한국어 감정 분류 모델을 전형적인 파인튜닝 구조를 통해 학습했다. SKTBrain에서 공개한 KoBERT는 한국어 위키 문장 약 500만 개와 한국어 뉴스 문장 약 2,000만 개를 학습한 모델로, 일반 BERT에 비해 한국어 문장 이해 성능이 우수하며, 단순 긍·부정 이진 분류가 아니라 여러 감정으로의 다중 분류에도 적합하다고 평가된다. 본 프로젝트에서는 HuggingFace Hub에 공개된 skt/kobert-base-v1 체크포인트를 사용하며, 의존성 문제를 해결하고 최신 환경에서도 안정적으로 동작하도록 하기 위해 기존 MXNet/GluonNLP 기반 구현을 HuggingFace 기반 토크나이저로 대체하고 vocab 구조를 직접 받아오는 BertModel + PyTorch 구조로 재구성했다.
- KoBERT
(https://github.com/SKTBrain/KoBERT)

```sh
pip install 'git+https://github.com/SKTBrain/KoBERT.git#egg=kobert_tokenizer&subdirectory=kobert_hf'
```

- 학습을 진행하기 위해 감정 레이블을 0~6의 정수로 매핑하고 train_test_split을 이용해 학습/검증 데이터를 8:2 비율로 나누었다. 이때 각 감정 클래스의 분포가 학습/검증 셋에서 비슷하게 유지되도록 stratify 옵션을 사용했다.
```python
# src/config.py
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
```

```python
# src/datasets.py
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
```

- KoBERT는 단순 문자열이 아니라 input_ids, segment_ids, valid_length 형태의 텐서를 입력으로 받기 때문에, 이를 자동으로 만들어 주는 BERTSentenceTransform 클래스를 구현하였다. BERTSentenceTransform 클래스는 하나의 문장을 토크나이저로 토큰화한 뒤, [CLS]와 [SEP] 토큰을 붙이고, 최대 길이(max_seq_length)를 넘는 부분을 잘라낸 다음, 부족한 부분은 PAD 토큰으로 채워서 고정 길이 시퀀스로 만드는 역할을 한다. 즉, 한 문장을 KoBERT가 바로 받을 수 있는 토큰 ID 시퀀스로 변환해주는 역할을 한다.
```python
# src/models.py
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
```

- BERTDataset 클래스는 학습용 데이터 구조를 PyTorch DataLoader에 바로 입력 가능한 형태로 만들기 위해 문장 목록과 라벨 목록을 받아, 각 문장을 BERTSentenceTransform으로 변환하고 PyTorch Dataset 형태로 감싸는 역할을 한다.
```python
# src/datasets.py
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
```

- 실제 데이터로부터 KoBERT 토크나이저와 사전학습된 BertModel을 불러온 뒤, 전처리된 데이터프레임을 기반으로 배치 단위 학습/검증용 DataLoader를 구성하는 코드는 다음과 같다.
```python
# src/models.py
def load_kobert_tokenizer():
    from kobert_tokenizer import KoBERTTokenizer

    return KoBERTTokenizer.from_pretrained(config.KOBERT_PRETRAINED)
```

```python
# scripts/train_kobert.py
train_df, valid_df = load_splits()

tokenizer = load_kobert_tokenizer()
model = build_kobert(pretrained=True).to(device)

train_set = BERTDataset(train_df["sentence"], train_df["emotion"], tokenizer, max_len=args.max_len)
valid_set = BERTDataset(valid_df["sentence"], valid_df["emotion"], tokenizer, max_len=args.max_len)
train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=0)
valid_loader = DataLoader(valid_set, batch_size=args.batch_size, shuffle=False, num_workers=0)
print(f"Train batch 수: {len(train_loader)}, Valid batch 수: {len(valid_loader)}")
```

- KoBERT 기반 감정 분류 모델은 사전학습된 BertModel 위에 단일 선형 분류 레이어를 올려 7개 감정 클래스를 예측하는 구조로 설계했다. 사전학습된 KoBERT에서 CLS 위치의 pooled_output을 가져와 Dropout으로 과적합을 완화하고 정규화한 뒤 최종 7차원으로 투영하는 과정을 정의하며, softmax를 적용해 각 감정 클래스에 대한 확률을 얻도록 하는 역할을 한다.
```python
# src/models.py
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
```

```python
# src/models.py
# pretrained=False 이면 config만 받아 구조를 만든다.
# 파인튜닝된 state_dict를 바로 덮어쓸 때(추론) 사전학습 가중치를 따로 받을 필요가 없다.

def build_kobert(pretrained: bool = True, dr_rate: float = config.KOBERT_DROPOUT) -> BERTClassifier:
    if pretrained:
        bert = BertModel.from_pretrained(config.KOBERT_PRETRAINED)
    else:
        bert = BertModel(BertConfig.from_pretrained(config.KOBERT_PRETRAINED))
    return BERTClassifier(bert, num_classes=config.NUM_LABELS, dr_rate=dr_rate)
```

- 학습에 사용한 주요 하이퍼파라미터는 다음과 같다.
```python
# src/config.py
KOBERT_PRETRAINED = "skt/kobert-base-v1"
KOBERT_MAX_LEN = 100
KOBERT_DROPOUT = 0.5
SEED = 42
```

```python
# scripts/train_kobert.py
parser.add_argument("--epochs", type=int, default=1)
parser.add_argument("--batch-size", type=int, default=64)
parser.add_argument("--lr", type=float, default=5e-5)
parser.add_argument("--max-len", type=int, default=config.KOBERT_MAX_LEN)
parser.add_argument("--warmup-ratio", type=float, default=0.1)
parser.add_argument("--max-grad-norm", type=float, default=1.0)
parser.add_argument("--log-interval", type=int, default=200)
```

- 옵티마이저는 BERT 계열 파인튜닝에서 표준적으로 사용하는 AdamW를 사용하고, AdamW + weight decay + cosine LR scheduler + warmup 조합을 통해 학습 초기에는 학습률을 점진적으로 증가시키고, 이후에는 점차 감소시키면서 안정적인 파인튜닝을 유도하도록 설계하였다.
```python
# scripts/train_kobert.py
no_decay = ["bias", "LayerNorm.weight"]
optimizer = AdamW(
    [
        {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         "weight_decay": 0.01},
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
         "weight_decay": 0.0},
    ],
    lr=args.lr,
)
loss_fn = nn.CrossEntropyLoss()

total_steps = len(train_loader) * args.epochs
scheduler = get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps=int(total_steps * args.warmup_ratio),
    num_training_steps=total_steps,
)
```

- 학습 루프는 에폭 단위로 반복되며, 각 에폭에서 학습 단계와 검증 단계를 순차적으로 수행한다. 학습 중에는 log_interval 배치마다 loss와 누적 train accuracy를 출력하고, 각 에폭이 끝난 뒤에는 eval() 모드에서 검증 데이터를 한 번 더 통과시켜 validation accuracy와 macro F1-score를 계산해 출력한다.
```python
# scripts/train_kobert.py
for epoch in range(1, args.epochs + 1):
    model.train()
    correct, seen = 0, 0
    print(f"\n===== KoBERT Epoch {epoch} / {args.epochs} =====")

    for batch_id, (token_ids, valid_length, segment_ids, label) in enumerate(tqdm(train_loader)):
        token_ids = token_ids.long().to(device)
        segment_ids = segment_ids.long().to(device)
        label = label.long().to(device)

        optimizer.zero_grad()
        logits = model(token_ids, valid_length, segment_ids)
        loss = loss_fn(logits, label)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        optimizer.step()
        scheduler.step()

        correct += (logits.argmax(dim=1) == label).sum().item()
        seen += label.size(0)
        if batch_id % args.log_interval == 0:
            print(f"Epoch {epoch} Batch {batch_id + 1} Loss {loss.item():.4f} Train Acc {correct / seen:.4f}")

    val_acc, val_f1 = compute_metrics(*predict_all(model, valid_loader, device))
    print(f"Epoch {epoch} Train Acc: {correct / seen:.4f}, Valid Acc: {val_acc:.4f}, Valid macro F1: {val_f1:.4f}")
```

- 학습이 끝나면, 추후 Streamlit 앱에서 재사용할 수 있도록 가중치를 `models/` 폴더에 저장한다. 이렇게 저장된 .pt 파일은 app.py에서 로드하여, 실제 사용자 입력에 대한 실시간 감정 분석 및 GIF 추천에 활용된다.
```python
# scripts/train_kobert.py
config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
weights_path = config.MODELS_DIR / config.KOBERT_WEIGHTS
torch.save(model.state_dict(), weights_path)
print("가중치 저장 완료:", weights_path)
```

### 2) KoELECTRA 기반 감정 분류 모델
본 프로젝트에서는 KoBERT와 더불어, 한국어에 특화된 사전학습 언어모델인 KoELECTRA 또한 함께 실험했다. KoELECTRA는 ELECTRA 구조를 기반으로 한 한국어 모델로 BERT 대비 학습 효율이 높고, 다양한 한국어 다운스트림 태스크에서 강한 성능을 보이는 것으로 알려져 있다. 따라서 동일한 감정 분류 데이터셋에 대해 KoBERT와 KoELECTRA를 각각 파인튜닝하고, 두 모델의 성능을 비교하는 실험을 진행했다.

- KoELECTRA (https://github.com/monologg/KoELECTRA)

- KoELECTRA와 같은 HuggingFace 기반 모델이 학습할 수 있도록, 문장과 감정 레이블을 모델 입력 포맷으로 변환하는 EmotionDataset 클래스를 정의했다. HuggingFace 모델은 단순 문자열을 바로 입력받지 못하고, 반드시 input_ids, attention_mask, token_type_ids 형태의 텐서를 입력으로 받아야 한다. 따라서 이 데이터셋 클래스는 토크나이저를 이용해 각 문장을 해당 구조로 자동 변환하고, 레이블도 PyTorch 텐서로 함께 반환한다.
```python
# src/datasets.py
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
```

- predict_all() 함수와 compute_metrics() 함수는 학습된 모델의 성능을 평가하기 위한 유틸리티로, 주어진 dataloader를 순회하며 예측값과 실제값을 모은 뒤 accuracy와 macro F1-score를 계산한다.
```python
# scripts/train_koelectra.py
def predict_all(model, dataloader, device):
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            logits = model(**batch).logits
            preds.extend(torch.argmax(logits, dim=-1).cpu().tolist())
            trues.extend(batch["labels"].cpu().tolist())
    return np.array(trues), np.array(preds)
```

```python
# src/evaluation.py
def compute_metrics(y_true, y_pred):
    return accuracy_score(y_true, y_pred), f1_score(y_true, y_pred, average="macro")
```

- 학습 스크립트의 main()은 HuggingFace 모델을 실제로 학습시키는 부분으로, 토크나이저·모델 로드 → 데이터셋/데이터로더 구성 → 학습 루프 실행 → validation 평가 과정으로 이루어진다. 이 구성은 이후 KoELECTRA 학습·평가 단계의 기반 역할을 한다.
```python
# src/models.py
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
```

```python
# scripts/train_koelectra.py
set_seed(config.SEED)
device = get_device()
print("사용 device:", device)

train_df, valid_df = load_splits()

tokenizer = load_koelectra_tokenizer(use_fast=False)
model = build_koelectra(pretrained=True)
model.resize_token_embeddings(len(tokenizer))
model.to(device)

train_set = EmotionDataset(train_df["sentence"], train_df["emotion"], tokenizer, max_len=args.max_len)
valid_set = EmotionDataset(valid_df["sentence"], valid_df["emotion"], tokenizer, max_len=args.max_len)
train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
valid_loader = DataLoader(valid_set, batch_size=args.batch_size, shuffle=False)

optimizer = AdamW(model.parameters(), lr=args.lr)

for epoch in range(1, args.epochs + 1):
    model.train()
    total_loss = 0.0
    loop = tqdm(train_loader, desc=f"Epoch {epoch}")
    for batch in loop:
        batch = {k: v.to(device) for k, v in batch.items()}
        loss = model(**batch).loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        loop.set_postfix(loss=loss.item())

    val_acc, val_f1 = compute_metrics(*predict_all(model, valid_loader, device))
    print(f"[Epoch {epoch}] loss: {total_loss / len(train_loader):.4f}, "
          f"val_acc: {val_acc:.4f}, val_macro_f1: {val_f1:.4f}")
```

- 사전학습된 KoELECTRA-base-v3-discriminator 모델을 사용해 감정 분류 작업에 맞게 파인튜닝을 수행한다. 학습 데이터(train_df)와 검증 데이터(valid_df)로 모델을 한 epoch 동안 학습시키고, 에폭마다 Validation Accuracy와 Macro F1-score를 출력한다. 학습이 완료되면 이후 실사용을 위한 예측 함수를 사용하여 문장 감정 분석을 수행할 수 있다. 사용한 하이퍼파라미터는 다음과 같다.
```python
# src/config.py
KOELECTRA_PRETRAINED = "monologg/koelectra-base-v3-discriminator"
KOELECTRA_MAX_LEN = 64
```

```python
# scripts/train_koelectra.py
parser.add_argument("--epochs", type=int, default=1)
parser.add_argument("--batch-size", type=int, default=32)
parser.add_argument("--lr", type=float, default=2e-5)
parser.add_argument("--max-len", type=int, default=config.KOELECTRA_MAX_LEN)
```

- 예측 함수인 predict_emotion_koelectra()는 입력 문장을 토크나이저로 인코딩한 뒤, KoELECTRA 모델에 전달하여 감정별 로짓(logit)을 계산한다. KoELECTRA에서도 모델 출력값은 softmax를 통해 확률로 변환되며, 가장 높은 확률을 가진 감정 클래스가 최종 예측 결과로 선택된다.
```python
# src/inference.py
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
```

- 문장을 입력하면 모델에 전달되어 감정을 예측하고, 예측된 감정 레이블과 그 확률을 함께 출력한다. 출력된 결과를 통해 모델이 특정 문장을 어떻게 해석하는지 직관적으로 확인할 수 있다.
```python
from src.inference import predict_emotion_koelectra
from src.models import load_finetuned_model
from src.utils import get_device

model_koel, tokenizer_koel = load_finetuned_model("KoELECTRA", get_device())

test_sent = "오늘 저녁은 치킨이닭"
pred_id, pred_label, pred_prob = predict_emotion_koelectra(test_sent, model_koel, tokenizer_koel)

print("입력 문장:", test_sent)
print("예측 감정 코드:", pred_id)
print("예측 감정 이름:", pred_label)
print("예측 감정 확률:", pred_prob)
```

- 학습이 끝나면, 추후 Streamlit 앱에서 재사용할 수 있도록 가중치를 `models/` 폴더에 저장한다. 이렇게 저장된 .pt 파일은 app.py에서 로드하여, 실제 사용자 입력에 대한 실시간 감정 분석 및 GIF 추천에 활용된다.
```python
# scripts/train_koelectra.py
config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
weights_path = config.MODELS_DIR / config.KOELECTRA_WEIGHTS
torch.save(model.state_dict(), weights_path)
print("가중치 저장 완료:", weights_path)
```

## 3. 모델 평가

### 1) KoBERT 기반 감정 분류 모델
이 단계에서는 학습이 완료된 KoBERT 감정 분류 모델의 성능을 정량적으로 평가하고, 각 감정 클래스별로 어떤 유형의 오분류가 발생하는지 분석한다. 이를 통해 KoBERT가 어떤 감정에 강점이 있고, 어떤 감정에서 상대적으로 취약한지 확인하여 이후 데이터 보완이나 모델 개선 방향을 설정할 수 있다.

- 먼저 KoBERT 전용 입력 포맷을 사용하는 BERTDataset과 학습 시에 사용했던 valid_loader를 그대로 활용하고, 별도의 평가 함수 predict_all()을 정의하여 실제 라벨과 예측 라벨을 수집한다. 이 함수는 DataLoader를 순회하면서 y_true, y_pred를 모아 반환하고, scikit-learn의 accuracy_score와 f1_score를 이용하는 compute_metrics()로 전체 성능을 산출한다. 반환된 y_true, y_pred는 이후 혼동행렬과 분류 리포트 생성에도 사용된다.
```python
# scripts/train_kobert.py
def predict_all(model, dataloader, device):
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for token_ids, valid_length, segment_ids, label in dataloader:
            logits = model(token_ids.long().to(device), valid_length, segment_ids.long().to(device))
            preds.extend(torch.argmax(logits, dim=-1).cpu().tolist())
            trues.extend(label.long().tolist())
    return np.array(trues), np.array(preds)


# 학습이 끝난 뒤 최종 평가
y_true, y_pred = predict_all(model, valid_loader, device)
save_evaluation_report("KoBERT", y_true, y_pred, extra={"args": vars(args)})
```

- 이후 classification_report()를 이용해 7개 감정 클래스(공포, 놀람, 분노, 슬픔, 중립, 행복, 혐오)에 대해 precision, recall, F1-score를 출력함으로써, KoBERT가 특정 감정에 더 강하거나 약한지 세부적으로 분석할 수 있다. 또한 confusion_matrix()를 사용해 실제 라벨과 예측 라벨의 조합을 행렬 형태로 출력하고, 이를 Matplotlib을 활용해 혼동행렬 이미지로 시각화한다. 마지막으로, classification_report 결과를 딕셔너리 형태로 받아 각 감정 클래스별 F1-score만 추출한 뒤, 막대 그래프로 시각화한다. 이 과정은 `save_evaluation_report()` 함수에 모여 있으며, 그림은 화면에 띄우는 대신 `outputs/` 폴더에 PNG로 저장하고, 전체 지표는 `outputs/<모델명>-metrics.json`으로 함께 저장한다. 그래프의 한글 라벨이 깨지지 않도록 설치된 한글 폰트를 찾아 지정한다.
```python
# src/evaluation.py
KOREAN_FONTS = ["NanumGothic", "AppleGothic", "Malgun Gothic", "Noto Sans CJK KR", "Noto Sans KR"]
# Colab 등에서 `apt-get install fonts-nanum` 직후에는 matplotlib 캐시에 없으므로 파일로 직접 등록
KOREAN_FONT_FILES = ["/usr/share/fonts/truetype/nanum/NanumGothic.ttf"]


def setup_korean_font() -> None:
    """설치된 한글 폰트를 찾아 matplotlib 기본 폰트로 지정한다 (없으면 기본 폰트 유지)."""
    for path in KOREAN_FONT_FILES:
        if Path(path).exists():
            font_manager.fontManager.addfont(path)
    installed = {f.name for f in font_manager.fontManager.ttflist}
    for name in KOREAN_FONTS:
        if name in installed:
            plt.rc("font", family=name)
            break
    plt.rcParams["axes.unicode_minus"] = False


def save_evaluation_report(name: str, y_true, y_pred, out_dir: Optional[Path] = None, extra=None) -> dict:
    """분류 리포트·혼동행렬을 출력하고, 그림과 metrics JSON을 out_dir(기본: outputs/)에 저장한다."""
    out_dir = Path(out_dir or config.OUTPUTS_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    setup_korean_font()
    labels = config.LABEL_NAMES
    label_ids = list(range(len(labels)))

    acc, macro_f1 = compute_metrics(y_true, y_pred)
    report = classification_report(
        y_true, y_pred, labels=label_ids, target_names=labels, output_dict=True, zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=label_ids)

    print(f"\n=== {name} 감정별 분류 리포트 ===")
    print(classification_report(y_true, y_pred, labels=label_ids, target_names=labels, zero_division=0))
    print(f"=== {name} 혼동행렬 (행: 실제, 열: 예측) ===")
    print(cm)

    slug = name.lower()

    plt.figure(figsize=(8, 6))
    plt.imshow(cm)
    plt.title(f"{name} Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.xticks(ticks=np.arange(len(labels)), labels=labels, rotation=45)
    plt.yticks(ticks=np.arange(len(labels)), labels=labels)
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(out_dir / f"{slug}-confusion-matrix.png", dpi=150)
    plt.close()

    f1_scores = [report[label]["f1-score"] for label in labels]
    plt.figure(figsize=(10, 5))
    plt.bar(labels, f1_scores)
    plt.title(f"{name} Per-Class F1-score")
    plt.xlabel("Emotion")
    plt.ylabel("F1-score")
    plt.ylim(0, 1.05)
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(out_dir / f"{slug}-f1-score.png", dpi=150)
    plt.close()

    metrics = {
        "model": name,
        "valid_accuracy": acc,
        "valid_macro_f1": macro_f1,
        "per_class_f1": dict(zip(labels, f1_scores)),
        **(extra or {}),
    }
    with open(out_dir / f"{slug}-metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print(f"\n{name} 평가 결과 저장: {out_dir}/{slug}-*.png, {slug}-metrics.json")
    return metrics
```

KoBERT 기반 감정 분류 모델은 검증 데이터에서 약 80.38%의 정확도와 0.72 수준의 macro F1-score를 기록하며, 전반적으로 KoELECTRA와 유사한 수준의 안정적인 성능을 보여주었다. 감정별 성능을 살펴보면, ‘중립’, ‘행복’, ‘혐오’ 감정에서 특히 우수한 결과를 확인할 수 있으며, 그 중에서도 ‘혐오’는 F1-score가 0.97로 거의 완벽에 가까운 분류 성능을 보였다. 반면, ‘놀람’과 ‘공포’, ‘분노’ 등 일부 감정은 상대적으로 성능이 떨어지는 편이다. 특히 ‘놀람’의 F1-score는 0.44, ‘공포’는 0.62, ‘분노’는 0.60으로, 유사한 정서적 톤을 가진 다른 부정 감정(슬픔, 분노, 공포 등)과 혼동되는 양상이 드러난다.

### 2) KoELECTRA 기반 감정 분류 모델
이 단계에서는 학습이 완료된 KoELECTRA 감정 분류 모델의 성능을 정량적으로 평가하고, 각 감정 클래스별로 어떤 패턴의 오분류가 발생하는지 분석한다.

- 먼저 학습에 사용한 EmotionDataset(train_set, valid_set)을 기반으로 평가용 DataLoader를 구성한다. 이를 통해 학습 데이터(train)와 검증 데이터(valid) 각각에 대해 성능을 측정하고, 학습 데이터의 accuracy와 macro F1-score는 `outputs/koelectra-metrics.json`에 검증 성능과 함께 기록해 한눈에 비교할 수 있도록 한다.
```python
# scripts/train_koelectra.py
# 최종 평가
train_eval_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=False)
train_acc, train_f1 = compute_metrics(*predict_all(model, train_eval_loader, device))
y_true, y_pred = predict_all(model, valid_loader, device)
save_evaluation_report(
    "KoELECTRA",
    y_true,
    y_pred,
    extra={"train_accuracy": train_acc, "train_macro_f1": train_f1, "args": vars(args)},
)
```

- 추가로, classification_report()를 사용해 7개 감정 클래스(공포, 놀람, 분노, 슬픔, 중립, 행복, 혐오)에 대해 precision, recall, F1-score를 출력함으로써, 어떤 감정에서 상대적으로 성능이 떨어지는지 세부적으로 확인할 수 있다. confusion_matrix()는 실제 라벨과 예측 결과의 조합을 행렬 형태로 제공하며, 이를 Matplotlib을 이용해 시각화함으로써 어떤 감정이 다른 감정으로 자주 오분류되는지를 직관적으로 파악할 수 있다. 마지막으로, classification_report의 결과를 각 감정 클래스별 F1-score의 막대 그래프로 시각화한다. 이 그래프는 모델이 특정 감정에 대해 상대적으로 강한지 혹은 약한지를 한눈에 보여주며, 이후 데이터 보완이나 모델 개선 방향을 설계하는 데 중요한 기준으로 활용될 수 있다. 이 과정은 KoBERT와 같은 `save_evaluation_report()` 함수(위 코드 참고)를 사용한다.

KoELECTRA 감정 분류 모델은 전체 검증 데이터 기준 약 81%의 정확도와 0.72의 macro F1-score를 보여, 7개 감정 클래스 전반에 걸쳐 비교적 안정적이고 균형 잡힌 성능을 확인할 수 있었다. 특히 ‘중립’, ‘행복’, ‘혐오’와 같이 표현적 특성이 뚜렷하거나 데이터가 충분한 클래스에서는 높은 정밀도와 재현율을 보이며 강한 분류 성능을 나타냈다. 반면, ‘놀람’과 같은 일부 감정 클래스는 재현율이 낮아 다른 감정(특히 중립)으로 오분류되는 경향이 있었다.

### 3) KoBERT, KoELECTRA 두 모델 성능 비교

학습된 두 모델의 성능을 종합해 봤을 때, KoELECTRA는 검증 정확도 81.16%, macro F1-score 0.715를 기록했으며, KoBERT는 정확도 80.38%, macro F1-score 0.723를 보였다. 정확도는 KoELECTRA가 약간 앞서지만, overall class 균형을 반영하는 macro F1-score에서는 KoBERT가 조금 더 우수한 성능을 보였다.

| 모델 | Accuracy | Macro F1 |
|---|---:|---:|
| KoBERT (`skt/kobert-base-v1`) | 80.38% | 0.723 |
| KoELECTRA (`monologg/koelectra-base-v3-discriminator`) | 81.16% | 0.715 |

<p align="center">
  <img src="docs/images/kobert-confusion-matrix.png" width="350">
  <img src="docs/images/koelectra-confusion-matrix.png" width="350">
</p>

<p align="center">
  <img src="docs/images/kobert-f1-score.png" width="350">
  <img src="docs/images/koelectra-f1-score.png" width="350">
</p>

두 학습 스크립트가 저장한 `outputs/*-metrics.json`을 읽어 성능을 표로 비교하는 스크립트도 있다.
```python
# scripts/compare_models.py
def main():
    rows = []
    for name in ("KoELECTRA", "KoBERT"):
        path = config.OUTPUTS_DIR / f"{name.lower()}-metrics.json"
        if not path.exists():
            print(f"[건너뜀] {path} 가 없습니다. 먼저 해당 모델을 학습하세요.")
            continue
        with open(path, encoding="utf-8") as f:
            m = json.load(f)
        rows.append((name, m["valid_accuracy"], m["valid_macro_f1"]))

    if not rows:
        return

    print(f"\n{'model':<10} {'accuracy':>9} {'macro_f1':>9}")
    for name, acc, f1 in rows:
        print(f"{name:<10} {acc:>9.4f} {f1:>9.4f}")
```


## 4. 감정 + 키워드 기반 GIF 추천 파이프라인

모델 학습과 평가가 끝난 뒤에는, 학습된 감정 분류 모델을 실제 서비스 형태로 활용하기 위해 **(1) 감정 예측 → (2) 키워드 추출 → (3) GIF 추천**으로 이어지는 파이프라인을 구현했다. 이때 감정 분류 모델은 KoBERT 또는 KoELECTRA를 사용할 수 있고, 키워드 추출은 공통적으로 KoNLPy의 Okt 형태소 분석기를 사용했다.

### 1) KoBERT 기반 감정 + 키워드 추출

먼저, 학습된 KoBERT 모델과 BERTSentenceTransform를 이용해 입력 문장에 대해 감정을 예측하는 predict_emotion_kobert() 함수를 정의하고, 형태소 기반 키워드 추출 함수 extract_keywords()와 두 결과를 하나로 묶는 analyze_sentence() 함수를 정의했다. 키워드 추출과 결과 묶기는 KoBERT와 KoELECTRA가 함께 사용한다.

- **1단계:** KoBERT로 감정을 예측
- **2단계:** Okt 형태소 분석기로 형태소 추출 (명사, 형용사, 동사만 후보 키워드로 사용, 한 글자 단어와 불용어 제거)
- **3단계:** 순서를 유지한 채로 중복 키워드를 제거하고, 최종적으로 {
    "sentence": 원본 문장,
    "emotion_label": 예측 감정,
    "emotion_prob": 감정 확률,
    "keywords": ['날씨', '춥다', '우울하다', ...], ...} 형태의 딕셔너리 반환

```python
# src/config.py
KEYWORD_POS = {"Noun", "Adjective", "Verb"}
STOPWORDS = {"하다", "되다", "있다", "그냥", "정말", "진짜", "근데"}
```

```python
# src/inference.py
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
```

```python
from konlpy.tag import Okt

from src.inference import analyze_sentence
from src.models import load_finetuned_model
from src.utils import get_device

kobert_model, tokenizer_kobert = load_finetuned_model("KoBERT", get_device())
okt = Okt()

sentence = "요즘 날씨도 춥고 기분도 좀 우울해."

res_kobert = analyze_sentence(
    sentence,
    "KoBERT",
    kobert_model,      # 학습된 KoBERT 분류기
    tokenizer_kobert,  # KoBERT 토크나이저
    okt,
    max_len=100,
)
```

### 2) KoELECTRA 기반 감정 + 키워드 추출
KoELECTRA 기반 파이프라인도 구조는 거의 유사하되, 입력 포맷이 HuggingFace 기본 포맷(input_ids, attention_mask, token_type_ids)이라는 점이 다르다. 감정 예측에는 [2. 모델 학습](#2-koelectra-기반-감정-분류-모델)에서 본 predict_emotion_koelectra()를 사용하고, analyze_sentence()는 backend 값에 따라 알맞은 예측 함수를 고른다. 키워드 추출과 반환 형식은 KoBERT와 같다.

```python
# src/inference.py
def predict_emotion(sentence, backend, model, tokenizer, max_len):
    if backend == "KoBERT":
        return predict_emotion_kobert(sentence, model, tokenizer, max_len=max_len)
    if backend == "KoELECTRA":
        return predict_emotion_koelectra(sentence, model, tokenizer, max_len=max_len)
    raise ValueError(f"알 수 없는 backend: {backend!r}")
```

```python
model_koel, tokenizer_koel = load_finetuned_model("KoELECTRA", get_device())

sentence = "날이 너무 추워져서 감기걸리기 쉬워."

res_koel = analyze_sentence(
    sentence,
    "KoELECTRA",
    model_koel,       # KoELECTRA 학습된 모델
    tokenizer_koel,   # KoELECTRA 토크나이저
    okt,
    max_len=64,
)
```

### 3) GIPHY API를 이용한 GIF 추천
마지막으로, 위에서 얻은 emotion_label, keywords 를 이용하여 GIPHY에서 실제 GIF를 검색하고 추천하는 로직을 구현했다. 최종적으로 GIPHY에 검색에 사용할 키워드 리스트는:
- **Okt 기반 키워드** (예: 날씨, 감기, 춥다 등)
- **한국어 감정 레이블** (예: 슬픔)
- **영어 감정 레이블** (예: sad)

위 세 요소를 합쳐서 구성된다. 즉, `["날씨", "감기", "춥다", "슬픔", "sad"]` 와 같은 형태가 되게 된다.

```python
# src/config.py
EMOTION_TO_EN = {
    "공포": "scared",
    "놀람": "surprised",
    "분노": "angry",
    "슬픔": "sad",
    "중립": "neutral",
    "행복": "happy",
    "혐오": "disgust",
}

GIPHY_SEARCH_URL = "https://api.giphy.com/v1/gifs/search"
```

```python
# src/giphy.py
def build_search_keywords(result: Dict) -> List[str]:
    """Okt 키워드 + 한국어 감정 라벨 + 영어 감정 라벨을 순서를 유지한 채 중복 없이 합친다."""
    emotion_ko = result["emotion_label"]
    emotion_en = config.EMOTION_TO_EN.get(emotion_ko, "emotion")
    keywords = [*result["keywords"], emotion_ko, emotion_en]
    return list(dict.fromkeys(keywords))
```

- 이후 각 키워드에 대해 GIPHY API를 호출해 최대 25개의 GIF를 검색하고(`rating=pg-13`), 그 중 최대 3개(`limit_per_kw`)를 랜덤으로 선택하여 전체 추천 리스트를 만든다. 전체 중복을 제거한 뒤 개수가 `total_limit`(기본 12개)를 넘으면 그 중 일부를 무작위로 골라 남긴다(`total_limit=None`이면 전부 반환). API 키가 없거나, 응답 오류·네트워크 오류가 난 키워드는 건너뛴다.

```python
# src/giphy.py
# 앞에서부터 순서대로 시도할 GIPHY 이미지 크기 (일부 결과는 downsized가 빈 값으로 온다)
RENDITION_PREFERENCE = ("downsized", "downsized_medium", "fixed_height", "original")


def gif_url(item: Dict) -> Optional[str]:
    """GIPHY 검색 결과 하나에서 쓸 수 있는 GIF URL을 고른다. 없으면 None."""
    images = item.get("images") or {}
    for name in RENDITION_PREFERENCE:
        url = (images.get(name) or {}).get("url")
        if url:
            return url
    return None


def recommend_gifs(
    result: Dict,
    api_key: str,
    limit_per_kw: int = 3,
    total_limit: Optional[int] = 12,
    rating: str = "pg-13",
    timeout: float = 5,
) -> Tuple[List[str], str]:
    """키워드마다 GIF를 최대 limit_per_kw개씩 무작위로 골라 (URL 목록, 안내 메시지)를 반환한다."""
    if not api_key:
        return [], "GIPHY API Key가 설정되지 않았습니다."

    all_urls = []
    for kw in build_search_keywords(result):
        params = {"q": kw, "api_key": api_key, "limit": "25", "rating": rating}
        try:
            resp = requests.get(config.GIPHY_SEARCH_URL, params=params, timeout=timeout)
            if resp.status_code != 200:
                logger.warning("GIPHY 응답 오류 (kw=%s, status=%s)", kw, resp.status_code)
                continue
            results = resp.json().get("data", [])
        except (requests.RequestException, ValueError) as e:
            logger.warning("GIPHY 검색 중 에러 (kw=%s): %s", kw, e)
            continue

        urls = [url for url in map(gif_url, results) if url]
        if not urls:
            continue

        all_urls.extend(random.sample(urls, min(limit_per_kw, len(urls))))

    if not all_urls:
        return [], "GIF 검색 결과가 없습니다."

    all_urls = list(dict.fromkeys(all_urls))
    if total_limit is not None and len(all_urls) > total_limit:
        all_urls = random.sample(all_urls, total_limit)

    picked = [re.sub(r"media\d", "i", url) for url in all_urls]
    return picked, f"총 {len(picked)}개 GIF 추천"
```

```python
import os

from src.giphy import build_search_keywords, recommend_gifs

print("최종 GIPHY 검색 키워드:", build_search_keywords(res_kobert))

gif_urls, msg = recommend_gifs(res_kobert, os.getenv("GIPHY_API_KEY"), limit_per_kw=3, total_limit=12)
print(msg)
```
정리하면, GIF 추천 로직은 다음과 같은 전략을 따른다.
- **모델이 이해한 감정 + 형태소 분석으로 추출한 키워드**를 모두 활용
- 감정(한국어/영어) + 키워드 조합으로 여러 검색 쿼리를 만들어 GIPHY를 조회
- 각 검색 결과에서 몇 개씩 랜덤 샘플링하여,
    - 특정 키워드에 치우치지 않으면서
    - 적당한 다양성을 가진 GIF 묶음을 구성
- 최종적으로 중복 제거 후 사용자에게 GIF 리스트를 보여준다.

이 로직은 이후 app.py (Streamlit 앱)에서 버튼 클릭 → 감정 + 키워드 추출 → GIPHY 검색 → 이미지 렌더링으로 통합되어, 사용자가 실제로 사용할 수 있는 서비스의 형태로 구현된다.

## 5. Streamlit 기반 웹 데모 앱(app.py) 구현

앞까지의 과정에서 우리는
- 통합 데이터셋 구축 및 전처리
- KoBERT / KoELECTRA 감정 분류 모델 학습 및 평가
- 감정 + 키워드 기반 GIPHY GIF 추천 로직 설계

를 완료했다. 이제 이러한 기능을 실제 사용자가 체험할 수 있도록, Streamlit 기반 웹 데모 앱을 구현했다. 이 앱은 로컬/서버에서 실행 가능한 간단한 웹 인터페이스로, 사용자가 한국어 문장을 입력하면 **문장 → 감정 예측(KoBERT/KoELECTRA) → 키워드 추출(OKT) → GIPHY 검색 & 추천**까지의 전체 파이프라인을 한 번에 경험할 수 있도록 한다.

app.py는 크게 다음과 같이 네 부분으로 구성된다.
- **모델 유틸 및 감정 + 키워드 추출 함수** → `src/models.py`, `src/inference.py`
    - KoBERT/KoELECTRA
    - predict_emotion_kobert / predict_emotion_koelectra, extract_keywords, analyze_sentence
- **GIPHY API 기반 GIF 추천 함수** → `src/giphy.py`
    - 감정 레이블 + 키워드를 이용해 GIPHY에서 GIF 검색 & 샘플링 (recommend_gifs)
- **모델/토크나이저/형태소 분석기 로딩 (캐시)** → `app.py`
    - KoBERT, KoELECTRA 모델 가중치(`.pt`) 파일 로드 — `models/`에 없으면 Hugging Face Hub에서 자동 다운로드
    - HuggingFace에서 모델 구조(config) + 토크나이저 불러오기
    - Okt 초기화
- **Streamlit UI (main 함수)** → `app.py`
    - 사이드바(KoBERT/KoELECTRA 모델 선택, 최대 길이, GIPHY API 키 — 비워두면 환경변수 `GIPHY_API_KEY` 사용)
    - 메인 채팅 UI
    - GIF 추천 결과 그리드
    - 사이드바에 감정 분석 결과 및 키워드 요약

학습된 가중치를 불러오는 함수는 다음과 같다. 추론 시에는 파인튜닝된 state_dict로 모든 가중치를 덮어쓰므로, 사전학습 가중치를 따로 받지 않고 config만으로 모델 구조를 만든다(`pretrained=False`).
```python
# src/models.py
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
```

```python
# src/config.py
KOBERT_WEIGHTS = "kobert_emotion_model_state_dict.pt"
KOELECTRA_WEIGHTS = "koelectra_emotion_model_state_dict.pt"

# 학습된 가중치(.pt)를 올려둔 Hugging Face Hub 모델 저장소
# 다른 저장소를 쓰려면 환경변수 HF_MODEL_REPO 로 덮어쓰면 된다.
HF_MODEL_REPO = os.getenv("HF_MODEL_REPO") or "qkrjds201/korean-emotion-gif-recommender"
```

app.py 전체 코드는 다음과 같다.
```python
# app.py
"""감정 분석 + GIF 추천 Streamlit 데모 앱.

실행 (프로젝트 루트에서):
    streamlit run app.py
"""

import html
import os

import streamlit as st

from src import config
from src.giphy import recommend_gifs
from src.inference import analyze_sentence
from src.models import BACKENDS, load_finetuned_model
from src.utils import get_device, set_seed

set_seed(config.SEED)

DEFAULT_MAX_LEN = {"KoBERT": config.KOBERT_MAX_LEN, "KoELECTRA": config.KOELECTRA_MAX_LEN}
KEYBOARD_IMAGE = config.ASSETS_DIR / "keyboard.png"


############################################
# 모델 / 형태소 분석기 로드 (캐시)
############################################

@st.cache_resource(show_spinner=False)
def load_model(backend):
    return load_finetuned_model(backend, get_device())


@st.cache_resource(show_spinner=False)
def load_okt():
    from konlpy.tag import Okt

    return Okt()


############################################
# UI
############################################

def inject_chat_css():
    st.markdown("""
    <style>
    .chat-wrapper {
        max-width: 700px;
        margin: 0.5rem 0 1.0rem 0;
    }

    .chat-row {
        display: flex;
        margin-bottom: 0.8rem;
    }

    .chat-bubble {
        padding: 18px 24px;
        border-radius: 26px;
        max-width: 100%;
        font-size: 1.2rem;
        line-height: 1.6;
        word-wrap: break-word;
        word-break: keep-all;
    }

    .chat-bubble.bot {
        background: #f1f0f0;
        color: #111;
        border-bottom-left-radius: 8px;
    }

    .chat-bubble.user {
        margin-left: auto;
        background: linear-gradient(
            to bottom,
            #a855f7,
            #7c3aed
        );
        color: #fff;
        border-bottom-right-radius: 6px;
    }

    .chat-meta {
        font-size: 0.8rem;
        color: #999;
        margin-top: 0.1rem;
        margin-bottom: 0.9rem;
    }

    .keyboard-img {
        opacity: 0.9;
        margin-top: 0.6rem;
    }
    </style>
    """, unsafe_allow_html=True)


def main():
    st.set_page_config(page_title="감정 분석 + GIF 추천 서비스", layout="wide")
    inject_chat_css()

    # ===== 세션 상태 초기화 =====
    st.session_state.setdefault("sent_once", False)
    st.session_state.setdefault("last_result", None)

    # ===== 사이드바: 설정 =====
    backend = st.sidebar.radio("사용할 모델", list(BACKENDS), index=0)

    max_len = st.sidebar.slider(
        "최대 토큰 길이",
        min_value=32,
        max_value=256,
        value=DEFAULT_MAX_LEN[backend],
        step=4,
    )

    # API 키는 환경변수(또는 .env)에서 읽는다. 입력창은 다른 키로 바꿔 쓰고 싶을 때만 사용.
    env_api_key = os.getenv("GIPHY_API_KEY", "")
    user_api_key = st.sidebar.text_input(
        "GIPHY API Key",
        value="",
        type="password",
        placeholder="환경변수 GIPHY_API_KEY 사용 중" if env_api_key else "API Key를 입력하세요",
        help="비워두면 환경변수 GIPHY_API_KEY 값을 사용합니다.",
    )
    giphy_api_key = user_api_key.strip() or env_api_key

    # ===== 모델 로딩 =====
    with st.spinner("모델 로딩 중... (최초 1회, 가중치가 없으면 Hugging Face Hub에서 다운로드)"):
        try:
            model, tokenizer = load_model(backend)
            okt = load_okt()
        except Exception as e:
            st.error(
                f"{backend} 모델을 불러오지 못했습니다: {e}\n\n"
                f"`models/` 폴더에 가중치 파일이 있는지, 또는 Hugging Face Hub 저장소 "
                f"`{config.HF_MODEL_REPO}` 에 접근할 수 있는지 확인하세요. "
                f"(KoNLPy를 쓰려면 Java(JDK)도 설치되어 있어야 합니다.)"
            )
            st.stop()

    st.session_state["backend"] = backend

    # ===== 메인 영역 =====
    st.markdown("## 📩 감정 분석 + GIF 추천 서비스")

    chat_container = st.container()

    # ===== 입력창 & 버튼 =====
    default_text = "요즘 날씨도 춥고 기분도 좀 우울해."
    col_input, col_btn = st.columns([5, 1])

    with col_input:
        sentence = st.text_input("메시지 입력", value=default_text, label_visibility="collapsed")
    with col_btn:
        analyze_clicked = st.button("GIF 추천받기")

    # 1단계: 버튼 클릭 처리 & 감정 분석
    if analyze_clicked:
        if not sentence.strip():
            st.warning("문장을 입력해 주세요.")
            return

        st.session_state["sent_once"] = True

        with st.spinner("감정 분석 중..."):
            res = analyze_sentence(sentence, backend, model, tokenizer, okt, max_len)

        st.session_state["last_result"] = res

    # 2단계: 위쪽 채팅 영역 렌더
    with chat_container:
        # 1) 상대방 말풍선
        st.markdown("""
        <div class="chat-wrapper">
          <div class="chat-row">
            <div class="chat-bubble bot">
              오늘 어떤 일이 있었어?
            </div>
          </div>
        """, unsafe_allow_html=True)

        if st.session_state["last_result"] is not None:
            user_text = html.escape(st.session_state["last_result"]["sentence"])
        else:
            user_text = '<span style="opacity:0.4;">여기에 당신의 메시지가 들어갑니다</span>'

        st.markdown(f"""
          <div class="chat-row">
            <div class="chat-bubble user">
              {user_text}
            </div>
          </div>
        </div> <!-- chat-wrapper end -->
        """, unsafe_allow_html=True)

    # ===== 키보드 그림 =====
    if not st.session_state["sent_once"] and KEYBOARD_IMAGE.exists():
        st.image(str(KEYBOARD_IMAGE), width=1000)

    # ===== GIF 추천 =====
    if st.session_state["last_result"] is not None:
        result = st.session_state["last_result"]

        st.markdown("---")
        st.markdown("### 🎬 GIF 추천")

        with st.spinner("GIPHY에서 GIF 검색 중..."):
            gif_urls, msg = recommend_gifs(result, giphy_api_key, limit_per_kw=3, total_limit=12)

        st.write(msg)

        if gif_urls:
            cols = st.columns(3)
            for i, url in enumerate(gif_urls):
                with cols[i % 3]:
                    st.image(url, width=250)

    # ===== 사이드바: 감정 분석 결과 영역 =====
    st.sidebar.markdown("---")
    st.sidebar.subheader("🔍 감정 분석 결과")

    if st.session_state["last_result"] is not None:
        res = st.session_state["last_result"]
        st.sidebar.markdown("**문장**")
        st.sidebar.write(res["sentence"])

        st.sidebar.markdown("**예측 감정**")
        st.sidebar.markdown(
            f"🧾 **{res['emotion_label']}**  "
            f"({res['emotion_prob'] * 100:.2f}% 확신)"
        )

        st.sidebar.markdown("**키워드**")
        if res["keywords"]:
            st.sidebar.write(", ".join(res["keywords"]))
        else:
            st.sidebar.write("_키워드가 추출되지 않았습니다._")
    else:
        st.sidebar.caption("아직 분석된 문장이 없습니다.")


if __name__ == "__main__":
    main()
```

## 6. 프로젝트 구조

```
.
├── app.py                    # Streamlit 데모 앱 (UI)
├── src/
│   ├── config.py             # 경로, 감정 라벨, 모델 이름, 불용어 등 설정
│   ├── models.py             # KoBERT/KoELECTRA 모델 정의, 가중치 로드(HF Hub 자동 다운로드)
│   ├── inference.py          # 감정 예측 + Okt 키워드 추출
│   ├── giphy.py              # GIPHY 검색 & GIF 추천
│   ├── datasets.py           # 학습용 Dataset, train/valid 분할
│   ├── evaluation.py         # 평가 지표, 혼동행렬/F1 그림 저장
│   └── utils.py              # 시드, 디바이스
├── scripts/
│   ├── preprocess.py         # 원본 데이터 → data/processed/data.csv
│   ├── train_kobert.py       # KoBERT 학습 + 평가
│   ├── train_koelectra.py    # KoELECTRA 학습 + 평가
│   ├── compare_models.py     # 두 모델 성능 비교
│   ├── download_models.py    # HF Hub에서 가중치 받기
│   └── upload_models.py      # HF Hub로 가중치 올리기
├── data/                     # 데이터 안내 (원본은 git 제외) → data/README.md
├── models/                   # 학습된 가중치 위치 (git 제외) → models/README.md
├── outputs/                  # 학습 시 생성되는 평가 그림·지표 (git 제외)
├── assets/                   # 앱에서 쓰는 이미지
├── docs/images/              # README 이미지
├── requirements.txt          # 앱 실행용
└── requirements-train.txt    # 전처리·학습용
```

## 7. 실행 방법 (Quick Start)

### 0) 준비물
- Python 3.9 이상
- **Java (JDK 8 이상)**: 키워드 추출에 쓰는 KoNLPy(Okt)가 Java를 필요로 합니다. ([KoNLPy 설치 가이드](https://konlpy.org/ko/latest/install/))
- GIPHY API Key: [GIPHY Developers](https://developers.giphy.com/dashboard/)에서 발급

### 1) 저장소 클론 & 패키지 설치
```bash
git clone https://github.com/seongheonpark/gif-recommender.git
cd gif-recommender

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2) API 키 설정
```bash
cp .env.example .env
# .env 파일을 열어 GIPHY_API_KEY= 뒤에 발급받은 키를 입력
```
앱 사이드바에서 키를 직접 입력해도 됩니다.

### 3) 앱 실행
```bash
streamlit run app.py
```
처음 실행하면 학습된 가중치(각 350~430MB)를 Hugging Face Hub에서 `models/`로 자동으로 받습니다.
미리 받아두려면 `python -m scripts.download_models` 를 실행하세요.

## 8. 직접 학습하기

```bash
pip install -r requirements-train.txt

# 1. 원본 데이터를 data/raw/ 에 준비 (data/README.md 참고) 후 전처리
python -m scripts.preprocess

# 2. 모델 학습 + 평가 (GPU 권장) → models/*.pt, outputs/*
python -m scripts.train_kobert
python -m scripts.train_koelectra

# 3. 성능 비교
python -m scripts.compare_models
```
각 스크립트의 하이퍼파라미터는 `--help`로 확인할 수 있습니다. 모든 명령은 프로젝트 루트에서 실행합니다.

직접 학습한 가중치를 Hugging Face Hub에 올려 앱에서 쓰려면:
```bash
hf auth login                    # 최초 1회, write 권한 토큰 필요
python -m scripts.upload_models --repo-id <HF아이디>/<저장소이름>
```
올린 저장소를 앱에서 쓰려면 `.env`에 `HF_MODEL_REPO=<HF아이디>/<저장소이름>` 을 추가합니다.
