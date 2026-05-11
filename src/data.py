from __future__ import annotations

import os
import random
from dataclasses import dataclass

import numpy as np
import torch
from datasets import Dataset, load_dataset
from torch.utils.data import DataLoader
from transformers import DataCollatorWithPadding, PreTrainedTokenizerBase


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


@dataclass
class SST2Splits:
    train: Dataset
    val: Dataset
    num_labels: int = 2


def load_sst2(
    tokenizer: PreTrainedTokenizerBase,
    max_length: int = 128,
    max_train_samples: int | None = None,
    max_val_samples: int | None = None,
) -> SST2Splits:
    # GLUE SST-2 test split has hidden labels, so use validation as held-out
    raw = load_dataset("glue", "sst2")

    def tokenize(batch):
        return tokenizer(batch["sentence"], truncation=True, max_length=max_length)

    train = raw["train"]
    val = raw["validation"]
    if max_train_samples is not None:
        train = train.select(range(min(max_train_samples, len(train))))
    if max_val_samples is not None:
        val = val.select(range(min(max_val_samples, len(val))))

    train = train.map(tokenize, batched=True, remove_columns=["sentence", "idx"])
    val = val.map(tokenize, batched=True, remove_columns=["sentence", "idx"])
    train = train.rename_column("label", "labels")
    val = val.rename_column("label", "labels")
    train.set_format("torch")
    val.set_format("torch")

    return SST2Splits(train=train, val=val)


def make_dataloaders(
    splits: SST2Splits,
    tokenizer: PreTrainedTokenizerBase,
    batch_size: int = 32,
    eval_batch_size: int | None = None,
    num_workers: int = 2,
) -> tuple[DataLoader, DataLoader]:
    collator = DataCollatorWithPadding(tokenizer=tokenizer)
    eval_bs = eval_batch_size or batch_size
    train_loader = DataLoader(
        splits.train,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collator,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        splits.val,
        batch_size=eval_bs,
        shuffle=False,
        collate_fn=collator,
        num_workers=num_workers,
        pin_memory=True,
    )
    return train_loader, val_loader
