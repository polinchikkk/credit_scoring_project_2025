"""
Пример запуска:

python -m src.infer --model artifacts/best_model.pt --json sample_client.json
"""

import argparse
import json
from pathlib import Path

import torch
import pandas as pd

from .data import build_vocab, encode
from .model import GRUWithAttention


def load_model(checkpoint):
    ckpt = torch.load(checkpoint, map_location="cpu")
    model = GRUWithAttention(len(ckpt["vocab"]))
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt["vocab"]


def make_tensor(history_json, vocab):
    df = pd.DataFrame(history_json)
    seq = torch.tensor([encode(df["product_type"], vocab)], dtype=torch.long)
    mask = (seq != 0).int()
    return seq, mask


def main(args):
    model, vocab = load_model(Path(args.model))
    with open(args.json) as fp:
        hist = json.load(fp)

    seq, mask = make_tensor(hist, vocab)
    with torch.no_grad():
        logit, alpha = model(seq, mask)
        prob = torch.sigmoid(logit).item()
    print(f"Вероятность дефолта: {prob:.4f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, required=True)
    p.add_argument("--json", type=str, required=True)
    main(p.parse_args())