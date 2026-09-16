from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class MatrixFactorization(nn.Module):
    """Collaborative-filtering baseline trained with sampled BPR loss."""

    def __init__(self, num_users: int, num_items: int, embedding_dim: int = 64) -> None:
        super().__init__()
        self.user_embeddings = nn.Embedding(num_users, embedding_dim)
        self.item_embeddings = nn.Embedding(num_items + 1, embedding_dim, padding_idx=0)
        nn.init.normal_(self.user_embeddings.weight, std=0.02)
        nn.init.normal_(self.item_embeddings.weight, std=0.02)

    def encode_users(self, user_ids: Tensor) -> Tensor:
        return F.normalize(self.user_embeddings(user_ids), dim=-1)

    def encode_items(self, item_ids: Tensor) -> Tensor:
        return F.normalize(self.item_embeddings(item_ids), dim=-1)

    def score(self, user_ids: Tensor, item_ids: Tensor) -> Tensor:
        return (self.encode_users(user_ids) * self.encode_items(item_ids)).sum(dim=-1)


def bpr_loss(positive_scores: Tensor, negative_scores: Tensor) -> Tensor:
    """Prefer each observed item over a randomly sampled non-target item."""
    return -F.logsigmoid(positive_scores - negative_scores).mean()


class TwoTowerModel(nn.Module):
    """History-aware user tower and metadata-aware item tower."""

    def __init__(
        self,
        num_users: int,
        num_items: int,
        metadata_cardinalities: list[int],
        embedding_dim: int = 64,
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.user_id_embedding = nn.Embedding(num_users, embedding_dim)
        self.history_item_embedding = nn.Embedding(
            num_items + 1, embedding_dim, padding_idx=0
        )
        self.item_id_embedding = nn.Embedding(num_items + 1, embedding_dim, padding_idx=0)
        self.metadata_embeddings = nn.ModuleList(
            nn.Embedding(cardinality, embedding_dim, padding_idx=0)
            for cardinality in metadata_cardinalities
        )
        self.user_projection = nn.Sequential(
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim),
        )
        self.item_projection = nn.Sequential(
            nn.Linear(embedding_dim * (1 + len(metadata_cardinalities)), embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim),
        )
        self.apply(self._initialize)

    @staticmethod
    def _initialize(module: nn.Module) -> None:
        if isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=0.02)
            if module.padding_idx is not None:
                with torch.no_grad():
                    module.weight[module.padding_idx].zero_()

    def encode_users(self, user_ids: Tensor, history_ids: Tensor, history_mask: Tensor) -> Tensor:
        history = self.history_item_embedding(history_ids)
        weights = history_mask.unsqueeze(-1).to(history.dtype)
        history_mean = (history * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1)
        combined = torch.cat([self.user_id_embedding(user_ids), history_mean], dim=-1)
        return F.normalize(self.user_projection(combined), dim=-1)

    def encode_items(self, item_ids: Tensor, metadata: Tensor) -> Tensor:
        parts = [self.item_id_embedding(item_ids)]
        parts.extend(
            embedding(metadata[:, column])
            for column, embedding in enumerate(self.metadata_embeddings)
        )
        return F.normalize(self.item_projection(torch.cat(parts, dim=-1)), dim=-1)

    def forward(
        self,
        user_ids: Tensor,
        item_ids: Tensor,
        history_ids: Tensor,
        history_mask: Tensor,
        item_metadata: Tensor,
    ) -> Tensor:
        """Return in-batch logits; diagonal entries are observed positive pairs."""
        users = self.encode_users(user_ids, history_ids, history_mask)
        items = self.encode_items(item_ids, item_metadata)
        return users @ items.T


def in_batch_softmax_loss(logits: Tensor, temperature: float = 0.07) -> Tensor:
    if logits.ndim != 2 or logits.shape[0] != logits.shape[1]:
        raise ValueError("logits must be a square user-item matrix")
    targets = torch.arange(logits.shape[0], device=logits.device)
    return F.cross_entropy(logits / temperature, targets)
