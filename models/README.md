# models/

학습된 감정 분류 모델 가중치가 위치하는 폴더입니다. `.pt` 파일은 용량이 커서(각 350~430MB) git에 올리지 않고 **Hugging Face Hub**에 보관합니다.

| 파일 | 모델 |
|---|---|
| `kobert_emotion_model_state_dict.pt` | KoBERT (`skt/kobert-base-v1`) + 선형 분류기 |
| `koelectra_emotion_model_state_dict.pt` | KoELECTRA (`monologg/koelectra-base-v3-discriminator`) |

## 받는 방법

```bash
python -m scripts.download_models
```

`streamlit run app.py` 로 앱을 실행할 때 파일이 없으면 자동으로 받습니다.
받아올 저장소는 [`src/config.py`](../src/config.py)의 `HF_MODEL_REPO`(또는 환경변수 `HF_MODEL_REPO`)로 정합니다.

## 직접 학습한 가중치 올리기

```bash
hf auth login
python -m scripts.upload_models --repo-id <HF아이디>/<저장소이름>
```
