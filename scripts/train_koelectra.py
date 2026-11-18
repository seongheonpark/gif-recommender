"""KoELECTRA 감정 분류 모델 학습 + 평가.

사용법 (프로젝트 루트에서):
    python -m scripts.train_koelectra [--epochs 1] [--batch-size 32] [--lr 2e-5] [--max-len 64]

결과:
    models/koelectra_emotion_model_state_dict.pt
    outputs/koelectra-{confusion-matrix,f1-score}.png, outputs/koelectra-metrics.json
"""

import argparse

import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src import config
from src.datasets import EmotionDataset, load_splits
from src.evaluation import compute_metrics, save_evaluation_report
from src.models import build_koelectra, load_koelectra_tokenizer
from src.utils import get_device, set_seed


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


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max-len", type=int, default=config.KOELECTRA_MAX_LEN)
    args = parser.parse_args()

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

    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    weights_path = config.MODELS_DIR / config.KOELECTRA_WEIGHTS
    torch.save(model.state_dict(), weights_path)
    print("가중치 저장 완료:", weights_path)

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


if __name__ == "__main__":
    main()
