# data/

원본 데이터는 각 제공처의 이용 약관상 재배포가 제한되므로 저장소에 포함하지 않습니다.
아래 데이터를 직접 받아 `data/raw/` 에 **표의 파일명 그대로** 넣어주세요.

| 데이터셋 | 출처 | `data/raw/` 파일명 |
|---|---|---|
| 감성 대화 말뭉치 | [AI Hub](https://aihub.or.kr/aihubdata/data/view.do?dataSetSn=86) | `감성대화말뭉치(최종데이터)_Training.xlsx`<br>`감성대화말뭉치(최종데이터)_Validation.xlsx` |
| 한국어 감정 정보가 포함된 연속적 대화 데이터셋 | [AI Hub](https://aihub.or.kr/aihubdata/data/view.do?dataSetSn=271) | `한국어_연속적_대화_데이터셋.xlsx` |
| 한국어 혐오 표현 데이터셋 | [smilegate-ai/korean_unsmile_dataset](https://github.com/smilegate-ai/korean_unsmile_dataset) | `혐오 데이터.csv` (cp949 인코딩) |

## 전처리

```bash
python -m scripts.preprocess
```

세 데이터셋을 7개 감정(공포·놀람·분노·슬픔·중립·행복·혐오)으로 정리·결합해 `data/processed/data.csv`(`Sentence`, `Emotion` 컬럼)를 만듭니다.

```
data/
├── raw/          # 원본 (git 제외)
└── processed/    # 전처리 결과 data.csv (git 제외)
```
