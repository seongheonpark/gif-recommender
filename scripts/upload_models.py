"""models/ 의 학습된 가중치(.pt)를 Hugging Face Hub 모델 저장소에 업로드한다.

사용법 (프로젝트 루트에서):
    hf auth login                  # 최초 1회, write 권한 토큰 필요
    python -m scripts.upload_models [--repo-id <HF아이디>/<저장소이름>] [--private]

--repo-id 를 생략하면 src/config.py 의 HF_MODEL_REPO 를 사용한다.
저장소가 없으면 새로 만든다.
"""

import argparse

from huggingface_hub import HfApi

from src import config
from src.models import BACKENDS, weights_filename


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo-id", default=config.HF_MODEL_REPO)
    parser.add_argument("--private", action="store_true", help="비공개 저장소로 생성")
    args = parser.parse_args()

    files = [config.MODELS_DIR / weights_filename(b) for b in BACKENDS]
    missing = [str(f) for f in files if not f.exists()]
    if missing:
        raise SystemExit(f"업로드할 파일이 없습니다: {missing}")

    api = HfApi()
    api.create_repo(repo_id=args.repo_id, repo_type="model", private=args.private, exist_ok=True)
    for path in files:
        print(f"업로드 중: {path.name} → {args.repo_id}")
        api.upload_file(
            path_or_fileobj=str(path),
            path_in_repo=path.name,
            repo_id=args.repo_id,
            repo_type="model",
            commit_message=f"Upload {path.name}",
        )
    print(f"완료: https://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    main()
