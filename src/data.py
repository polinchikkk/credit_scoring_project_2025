from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

def read_raw(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
    return df


def build_vocab(series: pd.Series) -> Dict[str, int]:
    vocab = {"<PAD>": 0, "<UNK>": 1}
    for token in series.unique():
        vocab[token] = len(vocab)
    return vocab


def encode(series: pd.Series, vocab: Dict[str, int]) -> List[int]:
    return [vocab.get(tok, vocab["<UNK>"]) for tok in series]


class CreditHistoryDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, vocab: Dict[str, int]) -> None:
        self.grouped = list(frame.groupby("client_id"))
        self.vocab = vocab

    def __len__(self) -> int:
        return len(self.grouped)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        client_id, df = self.grouped[idx]
        seq = torch.tensor(encode(df["product_type"], self.vocab))
        label = torch.tensor(df["target"].iloc[0], dtype=torch.float32)
        return seq, label


def collate(batch):
    seqs, labels = zip(*batch)
    pad_seqs = pad_sequence(seqs, batch_first=True, padding_value=0)
    attn_mask = (pad_seqs != 0).int()  # 0-ы — паддинг
    return pad_seqs, attn_mask, torch.stack(labels)


def make_loaders(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    batch_size: int = 256,
) -> Tuple[DataLoader, DataLoader, Dict[str, int]]:
    vocab = build_vocab(pd.concat([df_train["product_type"], df_val["product_type"]]))
    train_ds = CreditHistoryDataset(df_train, vocab)
    val_ds = CreditHistoryDataset(df_val, vocab)
    train_dl = DataLoader(train_ds, batch_size, shuffle=True, collate_fn=collate)
    val_dl = DataLoader(val_ds, batch_size, shuffle=False, collate_fn=collate)
    return train_dl, val_dl, vocab
