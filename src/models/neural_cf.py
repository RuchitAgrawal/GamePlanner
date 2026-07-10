"""
Neural Collaborative Filtering (NeuMF) model.

Combines a Generalized Matrix Factorization (GMF) branch with an MLP branch.
Separate embedding tables for each branch let them learn different representations.
The final prediction merges both branches before a sigmoid output layer.

Pre-training strategy: train GMF and MLP branches separately, then initialize
NeuMF from those weights before joint fine-tuning. This consistently outperforms
random initialization (see original paper, Section 4.3).

Reference: He et al., "Neural Collaborative Filtering" (WWW 2017)
"""

import torch
import torch.nn as nn

from src.utils.config import EMBEDDING_DIM, MLP_LAYERS, DROPOUT


class MLPBranch(nn.Module):
    """
    Standalone MLP branch for pre-training.
    Can be loaded into NeuMF after pre-training.
    """

    def __init__(self, num_users: int, num_items: int,
                 emb_dim: int = EMBEDDING_DIM,
                 layers: list[int] = MLP_LAYERS,
                 dropout: float = DROPOUT):
        super().__init__()
        self.user_emb = nn.Embedding(num_users, emb_dim)
        self.item_emb = nn.Embedding(num_items, emb_dim)

        mlp_in = emb_dim * 2
        mlp_modules = []
        for out_size in layers:
            mlp_modules.extend([
                nn.Linear(mlp_in, out_size),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            mlp_in = out_size
        self.mlp = nn.Sequential(*mlp_modules)
        self.output = nn.Linear(layers[-1], 1)

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.normal_(self.user_emb.weight, std=0.01)
        nn.init.normal_(self.item_emb.weight, std=0.01)
        for layer in self.mlp:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, user: torch.Tensor, item: torch.Tensor) -> torch.Tensor:
        u = self.user_emb(user)
        i = self.item_emb(item)
        x = torch.cat([u, i], dim=1)
        return self.output(self.mlp(x)).squeeze(1)

    def get_mlp_features(self, user: torch.Tensor, item: torch.Tensor) -> torch.Tensor:
        """Return final MLP hidden features (used when loading into NeuMF)."""
        u = self.user_emb(user)
        i = self.item_emb(item)
        x = torch.cat([u, i], dim=1)
        return self.mlp(x)


class NeuMF(nn.Module):
    """
    NeuMF: Neural Matrix Factorization.

    GMF branch: element-wise product of separate user/item embeddings.
    MLP branch: concatenated user/item embeddings passed through deep layers.
    Both branches are concatenated and passed to a final prediction layer.

    Args:
        num_users: number of unique users.
        num_items: number of unique items.
        emb_dim: embedding dimension for both branches.
        layers: MLP hidden layer sizes (e.g. [128, 64, 32]).
        dropout: dropout rate after each MLP activation.
    """

    def __init__(
        self,
        num_users: int,
        num_items: int,
        emb_dim: int = EMBEDDING_DIM,
        layers: list[int] = MLP_LAYERS,
        dropout: float = DROPOUT,
    ):
        super().__init__()

        # GMF branch
        self.gmf_user_emb = nn.Embedding(num_users, emb_dim)
        self.gmf_item_emb = nn.Embedding(num_items, emb_dim)

        # MLP branch
        self.mlp_user_emb = nn.Embedding(num_users, emb_dim)
        self.mlp_item_emb = nn.Embedding(num_items, emb_dim)

        mlp_in = emb_dim * 2
        mlp_modules = []
        for out_size in layers:
            mlp_modules.extend([
                nn.Linear(mlp_in, out_size),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            mlp_in = out_size
        self.mlp = nn.Sequential(*mlp_modules)

        # Merge and predict
        # GMF contributes emb_dim features, MLP contributes layers[-1] features
        self.predict = nn.Linear(emb_dim + layers[-1], 1)

        self._init_weights()

    def _init_weights(self) -> None:
        for emb in [self.gmf_user_emb, self.gmf_item_emb,
                    self.mlp_user_emb, self.mlp_item_emb]:
            nn.init.normal_(emb.weight, std=0.01)
        for layer in self.mlp:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)
        nn.init.kaiming_uniform_(self.predict.weight, nonlinearity="sigmoid")

    def forward(self, user: torch.Tensor, item: torch.Tensor) -> torch.Tensor:
        """
        Args:
            user: (batch,) user index tensor.
            item: (batch,) item index tensor.
        Returns:
            (batch,) prediction logits (pre-sigmoid).
        """
        # GMF branch
        gmf_u = self.gmf_user_emb(user)
        gmf_i = self.gmf_item_emb(item)
        gmf_out = gmf_u * gmf_i    # (batch, emb_dim)

        # MLP branch
        mlp_u = self.mlp_user_emb(user)
        mlp_i = self.mlp_item_emb(item)
        mlp_in = torch.cat([mlp_u, mlp_i], dim=1)
        mlp_out = self.mlp(mlp_in)  # (batch, layers[-1])

        # Merge
        merged = torch.cat([gmf_out, mlp_out], dim=1)
        logit = self.predict(merged).squeeze(1)
        return logit

    def load_pretrained_weights(self, gmf_state: dict, mlp_state: dict) -> None:
        """
        Initialize from separately pre-trained GMF and MLP weights.
        Call this before joint fine-tuning to replicate the NeuMF paper's training protocol.

        gmf_state: state_dict of a trained GMF model.
        mlp_state: state_dict of a trained MLPBranch model.
        """
        self.gmf_user_emb.weight.data.copy_(gmf_state["user_emb.weight"])
        self.gmf_item_emb.weight.data.copy_(gmf_state["item_emb.weight"])

        self.mlp_user_emb.weight.data.copy_(mlp_state["user_emb.weight"])
        self.mlp_item_emb.weight.data.copy_(mlp_state["item_emb.weight"])
        self.mlp.load_state_dict(
            {k.replace("mlp.", ""): v for k, v in mlp_state.items() if k.startswith("mlp.")}
        )

    def score_all_items(self, user: torch.Tensor, num_items: int,
                        device: torch.device, batch_size: int = 2048) -> torch.Tensor:
        """
        Score a single user against all items in batches.
        Returns a (num_items,) tensor of logits. Used during evaluation.
        """
        all_scores = []
        item_range = torch.arange(num_items, device=device)
        for start in range(0, num_items, batch_size):
            batch_items = item_range[start:start + batch_size]
            batch_users = user.expand(len(batch_items))
            with torch.no_grad():
                scores = self.forward(batch_users, batch_items)
            all_scores.append(scores)
        return torch.cat(all_scores)
