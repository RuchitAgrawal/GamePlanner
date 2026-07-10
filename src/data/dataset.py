"""
PyTorch Dataset for training collaborative filtering models on implicit feedback.

Handles on-the-fly negative sampling to keep memory usage low.
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.utils.config import NUM_NEGATIVES


class SteamDataset(Dataset):
    """
    Dataset for implicit feedback training with negative sampling.

    For each positive (user, item) pair, samples NUM_NEGATIVES items
    the user has not interacted with. Returns (user_idx, item_idx, label).

    Args:
        df: DataFrame with user_idx, item_idx, and optionally confidence columns.
        num_items: total number of items in the catalog.
        num_negatives: negative samples per positive interaction.
        use_confidence: whether to use log-playtime confidence as sample weight.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        num_items: int,
        num_negatives: int = NUM_NEGATIVES,
        use_confidence: bool = False,
    ):
        self.num_items = num_items
        self.num_negatives = num_negatives
        self.use_confidence = use_confidence

        self.users = df["user_idx"].values.astype(np.int64)
        self.items = df["item_idx"].values.astype(np.int64)
        self.confidences = df["confidence"].values.astype(np.float32) if use_confidence else None

        # Build a per-user set of positive item indices for fast negative sampling
        self.user_positives: dict[int, set[int]] = {}
        for u, i in zip(self.users, self.items):
            self.user_positives.setdefault(int(u), set()).add(int(i))

    def __len__(self) -> int:
        # One entry per positive interaction (negatives are generated on the fly)
        return len(self.users)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        user = int(self.users[idx])
        pos_item = int(self.items[idx])
        conf = float(self.confidences[idx]) if self.use_confidence else 1.0

        # Positive sample
        user_list = [user]
        item_list = [pos_item]
        label_list = [1.0]
        weight_list = [conf]

        # Negative samples: draw random items until we have enough non-interacted ones
        positives = self.user_positives.get(user, set())
        count = 0
        while count < self.num_negatives:
            neg_item = np.random.randint(0, self.num_items)
            if neg_item not in positives:
                user_list.append(user)
                item_list.append(neg_item)
                label_list.append(0.0)
                weight_list.append(1.0)
                count += 1

        return {
            "user": torch.tensor(user_list, dtype=torch.long),
            "item": torch.tensor(item_list, dtype=torch.long),
            "label": torch.tensor(label_list, dtype=torch.float32),
            "weight": torch.tensor(weight_list, dtype=torch.float32),
        }


def collate_fn(batch: list[dict]) -> dict[str, torch.Tensor]:
    """Flatten the batch so each sample is one (user, item, label) tuple."""
    return {
        "user": torch.cat([b["user"] for b in batch]),
        "item": torch.cat([b["item"] for b in batch]),
        "label": torch.cat([b["label"] for b in batch]),
        "weight": torch.cat([b["weight"] for b in batch]),
    }
