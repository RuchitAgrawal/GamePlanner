"""
Generalized Matrix Factorization (GMF) model for collaborative filtering.

User and item embeddings are combined via element-wise product (GMF path).
Includes user and item bias terms for better calibration on implicit feedback.

Reference: He et al., "Neural Collaborative Filtering" (WWW 2017)
"""

import torch
import torch.nn as nn

from src.utils.config import EMBEDDING_DIM


class GMF(nn.Module):
    """
    Generalized Matrix Factorization.

    Learns separate user and item embedding matrices. The dot product of
    a user vector and an item vector approximates the interaction probability.
    Adding bias terms per user and per item improves calibration.

    Args:
        num_users: number of unique users in the training set.
        num_items: number of unique items in the training set.
        emb_dim: embedding dimension for users and items.
    """

    def __init__(self, num_users: int, num_items: int, emb_dim: int = EMBEDDING_DIM):
        super().__init__()
        self.user_emb = nn.Embedding(num_users, emb_dim)
        self.item_emb = nn.Embedding(num_items, emb_dim)
        self.user_bias = nn.Embedding(num_users, 1)
        self.item_bias = nn.Embedding(num_items, 1)
        self.output = nn.Linear(emb_dim, 1, bias=True)

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.normal_(self.user_emb.weight, std=0.01)
        nn.init.normal_(self.item_emb.weight, std=0.01)
        nn.init.zeros_(self.user_bias.weight)
        nn.init.zeros_(self.item_bias.weight)
        nn.init.kaiming_uniform_(self.output.weight, nonlinearity="sigmoid")

    def forward(self, user: torch.Tensor, item: torch.Tensor) -> torch.Tensor:
        """
        Args:
            user: (batch,) user index tensor.
            item: (batch,) item index tensor.
        Returns:
            (batch,) prediction logits (pre-sigmoid).
        """
        u = self.user_emb(user)    # (batch, emb_dim)
        i = self.item_emb(item)    # (batch, emb_dim)
        ub = self.user_bias(user).squeeze(1)   # (batch,)
        ib = self.item_bias(item).squeeze(1)   # (batch,)

        product = u * i                        # (batch, emb_dim)
        logit = self.output(product).squeeze(1) + ub + ib
        return logit

    def get_user_embedding(self, user: torch.Tensor) -> torch.Tensor:
        return self.user_emb(user)

    def get_item_embedding(self, item: torch.Tensor) -> torch.Tensor:
        return self.item_emb(item)

    @torch.no_grad()
    def score_all_items(
        self, user: torch.Tensor, num_items: int, device: torch.device
    ) -> torch.Tensor:
        """
        Score all items for a single user in batches.
        Returns a (num_items,) tensor of logits, ready for top-k selection.
        """
        self.eval()
        all_items = torch.arange(num_items, dtype=torch.long, device=device)
        scores = []
        for start in range(0, num_items, 2048):
            batch = all_items[start:start + 2048]
            u = user.expand(len(batch))
            scores.append(self(u, batch).cpu())
        return torch.cat(scores)
