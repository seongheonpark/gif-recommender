"""KoBERT 감정 분류 모델 학습 + 평가.

사용법 (프로젝트 루트에서):
    python -m scripts.train_kobert [--epochs 1] [--batch-size 64] [--lr 5e-5] [--max-len 100]

결과:
    models/kobert_emotion_model_state_dict.pt
    outputs/kobert-{confusion-matrix,f1-score}.png, outputs/kobert-metrics.json
"""

import argparse

import numpy as np
import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers.optimization import get_cosine_schedule_with_warmup

from src import config
from src.datasets import BERTDataset, load_splits
from src.evaluation import compute_metrics, save_evaluation_report
from src.models import build_kobert, load_kobert_tokenizer
from src.utils import get_device, set_seed


def predict_all(model, dataloader, device):
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for token_ids, valid_length, segment_ids, label in dataloader:
            logits = model(token_ids.long().to(device), valid_length, segment_ids.long().to(device))
            preds.extend(torch.argmax(logits, dim=-1).cpu().tolist())
            trues.extend(label.long().tolist())
    return np.array(trues), np.array(preds)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--max-len", type=int, default=config.KOBERT_MAX_LEN)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--log-interval", type=int, default=200)
    args = parser.parse_args()

    set_seed(config.SEED)
    device = get_device()
    print("사용 device:", device)

    train_df, valid_df = load_splits()

    tokenizer = load_kobert_tokenizer()
    model = build_kobert(pretrained=True).to(device)

    train_set = BERTDataset(train_df["sentence"], train_df["emotion"], tokenizer, max_len=args.max_len)
    valid_set = BERTDataset(valid_df["sentence"], valid_df["emotion"], tokenizer, max_len=args.max_len)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=0)
    valid_loader = DataLoader(valid_set, batch_size=args.batch_size, shuffle=False, num_workers=0)
    print(f"Train batch 수: {len(train_loader)}, Valid batch 수: {len(valid_loader)}")

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

    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    weights_path = config.MODELS_DIR / config.KOBERT_WEIGHTS
    torch.save(model.state_dict(), weights_path)
    print("가중치 저장 완료:", weights_path)

    y_true, y_pred = predict_all(model, valid_loader, device)
    save_evaluation_report("KoBERT", y_true, y_pred, extra={"args": vars(args)})


if __name__ == "__main__":
    main()
