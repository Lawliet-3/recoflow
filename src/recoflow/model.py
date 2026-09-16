from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class TwoTowerModel(nn.Module):
    """ID-based retrieval model; metadata encoders can be added behind each tower."""

    def __init__(self, num_users: int, num_items: int, embedding_dim: int = 64) -> None:
        super().__init__()
        self.user_tower = nn.Embedding(num_users, embedding_dim)
        self.item_tower = nn.Embedding(num_items, embedding_dim)
        nn.init.normal_(self.user_tower.weight, std=0.02)
        nn.init.normal_(self.item_tower.weight, std=0.02)

    def encode_users(self, user_ids: Tensor) -> Tensor:
        return F.normalize(self.user_tower(user_ids), dim=-1)

    def encode_items(self, item_ids: Tensor) -> Tensor:
        return F.normalize(self.item_tower(item_ids), dim=-1)

    def forward(self, user_ids: Tensor, item_ids: Tensor) -> Tensor:
        """Return in-batch logits; the diagonal contains observed positive pairs."""
        users = self.encode_users(user_ids)
        items = self.encode_items(item_ids)
        return users @ items.T


def in_batch_softmax_loss(logits: Tensor, temperature: float = 0.07) -> Tensor:
    if logits.ndim != 2 or logits.shape[0] != logits.shape[1]:
        raise ValueError("logits must be a square user-item matrix")
    targets = torch.arange(logits.shape[0], device=logits.device)
    return F.cross_entropy(logits / temperature, targets)

