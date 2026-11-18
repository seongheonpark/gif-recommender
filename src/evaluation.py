"""평가 지표 계산과 결과 그림/리포트 저장 (학습 스크립트 전용)."""

import json
from pathlib import Path
from typing import Optional

import matplotlib

matplotlib.use("Agg")  # 화면 없이 파일로만 저장

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

from . import config  # noqa: E402

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


def compute_metrics(y_true, y_pred):
    return accuracy_score(y_true, y_pred), f1_score(y_true, y_pred, average="macro")


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
