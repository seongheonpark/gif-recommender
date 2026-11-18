"""학습 스크립트가 저장한 outputs/*-metrics.json 을 읽어 두 모델 성능을 비교한다.

사용법 (프로젝트 루트에서):
    python -m scripts.compare_models
"""

import json

from src import config


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


if __name__ == "__main__":
    main()
