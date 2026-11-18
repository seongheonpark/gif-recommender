"""Hugging Face Hub에서 학습된 가중치(.pt)를 models/ 로 내려받는다.

사용법 (프로젝트 루트에서):
    python -m scripts.download_models

저장소는 src/config.py 의 HF_MODEL_REPO (또는 환경변수 HF_MODEL_REPO)를 따른다.
앱(app.py)도 가중치가 없으면 처음 실행할 때 같은 방식으로 자동 다운로드한다.
"""

from src import config
from src.models import BACKENDS, resolve_weights, weights_filename


def main():
    print(f"저장소: {config.HF_MODEL_REPO}")
    for backend in BACKENDS:
        path = resolve_weights(weights_filename(backend))
        print(f"[{backend}] {path}")


if __name__ == "__main__":
    main()
