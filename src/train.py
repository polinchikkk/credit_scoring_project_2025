import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import roc_auc_score

from .data import read_raw, make_loaders
from .model import GRUWithAttention


def epoch_loop(model, loader, optimizer=None, scaler=None, device="cpu"):
    model.train(optimizer is not None)
    preds, trues, losses = [], [], []

    for seq, mask, y in loader:
        seq, mask, y = seq.to(device), mask.to(device), y.to(device)
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            logit, _ = model(seq, mask)
            loss = F.binary_cross_entropy_with_logits(logit, y)
        if optimizer:
            optimizer.zero_grad()
            if scaler:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
        preds.append(torch.sigmoid(logit).detach().cpu())
        trues.append(y.cpu())
        losses.append(loss.detach().cpu())

    preds = torch.cat(preds).numpy()
    trues = torch.cat(trues).numpy()
    return float(torch.tensor(losses).mean()), roc_auc_score(trues, preds)


def main(args):
    df = read_raw(Path(args.data))
    df_train = df[df["fold"] == "train"].copy()
    df_val = df[df["fold"] == "val"].copy()

    train_dl, val_dl, vocab = make_loaders(df_train, df_val, args.batch_size)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = GRUWithAttention(len(vocab)).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    sch = ReduceLROnPlateau(opt, mode="max", patience=2, factor=0.5, verbose=True)
    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

    writer = SummaryWriter()

    best_auc = 0.0
    Path(args.out).mkdir(exist_ok=True, parents=True)

    for epoch in range(args.epochs):
        train_loss, train_auc = epoch_loop(model, train_dl, opt, scaler, device)
        val_loss, val_auc = epoch_loop(model, val_dl, device=device)

        writer.add_scalars(
            "Loss", {"train": train_loss, "val": val_loss}, epoch
        )
        writer.add_scalars(
            "AUC", {"train": train_auc, "val": val_auc}, epoch
        )

        sch.step(val_auc)

        print(
            f"Epoch {epoch:02d} | "
            f"train_auc={train_auc:.4f} val_auc={val_auc:.4f} "
            f"(best={best_auc:.4f})"
        )

        if val_auc > best_auc:
            best_auc = val_auc
            torch.save(
                {"state_dict": model.state_dict(), "vocab": vocab},
                Path(args.out) / "best_model.pt",
            )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=str, required=True, help="CSV/Parquet с фолдами")
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--out", type=str, default="artifacts")
    main(p.parse_args())
