from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from recoflow.artifacts import save_embeddings
from recoflow.baseline import PopularityRecommender
from recoflow.data import (
    build_training_examples,
    encode_item_metadata,
    load_articles,
    load_transactions,
    temporal_split,
)
from recoflow.evaluation import (
    exact_recommendations,
    ground_truth_by_user,
    popularity_recommendations,
    ranking_metrics,
    seen_items_by_user,
)
from recoflow.model import MatrixFactorization, TwoTowerModel, bpr_loss, in_batch_softmax_loss


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _train_matrix_factorization(
    examples: TensorDataset,
    num_users: int,
    num_items: int,
    embedding_dim: int,
    epochs: int,
    batch_size: int,
) -> MatrixFactorization:
    model = MatrixFactorization(num_users, num_items, embedding_dim)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loader = DataLoader(examples, batch_size=batch_size, shuffle=True)
    model.train()
    for _ in range(epochs):
        for user_ids, positive_ids, *_ in loader:
            negative_ids = torch.randint(1, num_items + 1, positive_ids.shape)
            collisions = negative_ids == positive_ids
            while collisions.any():
                negative_ids[collisions] = torch.randint(1, num_items + 1, (int(collisions.sum()),))
                collisions = negative_ids == positive_ids
            loss = bpr_loss(
                model.score(user_ids, positive_ids),
                model.score(user_ids, negative_ids),
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return model.eval()


def _train_two_tower(
    examples: TensorDataset,
    metadata: torch.Tensor,
    cardinalities: list[int],
    num_users: int,
    num_items: int,
    embedding_dim: int,
    epochs: int,
    batch_size: int,
) -> TwoTowerModel:
    model = TwoTowerModel(
        num_users,
        num_items,
        metadata_cardinalities=cardinalities,
        embedding_dim=embedding_dim,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loader = DataLoader(examples, batch_size=batch_size, shuffle=True)
    model.train()
    for _ in range(epochs):
        for user_ids, item_ids, history_ids, history_mask in loader:
            if len(user_ids) < 2:
                continue
            logits = model(
                user_ids,
                item_ids,
                history_ids,
                history_mask,
                metadata[item_ids],
            )
            loss = in_batch_softmax_loss(logits, item_ids=item_ids)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return model.eval()


def train(
    transactions_path: str,
    articles_path: str,
    output_dir: str = "artifacts",
    epochs: int = 3,
    embedding_dim: int = 64,
    batch_size: int = 512,
    max_history: int = 20,
    evaluation_k: int = 100,
    max_eval_users: int = 5_000,
    max_rows: int | None = None,
    seed: int = 42,
) -> dict[str, dict[str, float | int]]:
    """Train, export, and evaluate all V1 recommenders."""
    _seed_everything(seed)
    transactions = load_transactions(transactions_path, max_rows=max_rows)
    articles = load_articles(articles_path)
    split = temporal_split(transactions)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    users = sorted(split.train["customer_id"].astype(str).unique())
    items = sorted(split.train["article_id"].astype(str).unique())
    if len(items) < 2:
        raise ValueError("training requires at least two distinct items")
    user_to_index = {value: index for index, value in enumerate(users)}
    item_to_index = {value: index + 1 for index, value in enumerate(items)}
    encoded = build_training_examples(
        split.train, user_to_index, item_to_index, max_history=max_history
    )
    metadata_array, cardinalities = encode_item_metadata(articles, items)
    metadata = torch.from_numpy(metadata_array)
    dataset = TensorDataset(
        torch.from_numpy(encoded.user_indices),
        torch.from_numpy(encoded.item_indices),
        torch.from_numpy(encoded.history_indices),
        torch.from_numpy(encoded.history_mask),
    )

    popularity = PopularityRecommender().fit(split.train)
    (output / "popularity.json").write_text(json.dumps(popularity.ranked_items))

    matrix_factorization = _train_matrix_factorization(
        dataset, len(users), len(items), embedding_dim, epochs, batch_size
    )
    two_tower = _train_two_tower(
        dataset,
        metadata,
        cardinalities,
        len(users),
        len(items),
        embedding_dim,
        epochs,
        batch_size,
    )

    all_user_indices = torch.arange(len(users))
    all_item_indices = torch.arange(1, len(items) + 1)
    with torch.no_grad():
        mf_user_vectors = matrix_factorization.encode_users(all_user_indices).numpy()
        mf_item_vectors = matrix_factorization.encode_items(all_item_indices).numpy()
        two_tower_user_vectors = two_tower.encode_users(
            all_user_indices,
            torch.from_numpy(encoded.latest_histories),
            torch.from_numpy(encoded.latest_history_mask),
        ).numpy()
        two_tower_item_vectors = two_tower.encode_items(
            all_item_indices, metadata[all_item_indices]
        ).numpy()

    save_embeddings(output / "user_embeddings.npz", users, two_tower_user_vectors)
    save_embeddings(output / "item_embeddings.npz", items, two_tower_item_vectors)
    torch.save(
        {
            "state_dict": matrix_factorization.state_dict(),
            "users": users,
            "items": items,
            "embedding_dim": embedding_dim,
        },
        output / "matrix_factorization.pt",
    )
    torch.save(
        {
            "state_dict": two_tower.state_dict(),
            "users": users,
            "items": items,
            "metadata_cardinalities": cardinalities,
            "embedding_dim": embedding_dim,
            "max_history": max_history,
        },
        output / "two_tower.pt",
    )

    truth = ground_truth_by_user(split.validation, set(users), set(items))
    eval_users = sorted(truth)[:max_eval_users]
    truth = {user: truth[user] for user in eval_users}
    seen = seen_items_by_user(split.train)
    user_positions = [user_to_index[user] for user in eval_users]
    popularity_rankings = popularity_recommendations(
        eval_users, popularity.ranked_items, seen, evaluation_k
    )
    mf_rankings = exact_recommendations(
        eval_users,
        mf_user_vectors[user_positions],
        items,
        mf_item_vectors,
        seen,
        evaluation_k,
    )
    two_tower_rankings = exact_recommendations(
        eval_users,
        two_tower_user_vectors[user_positions],
        items,
        two_tower_item_vectors,
        seen,
        evaluation_k,
    )
    metrics = {
        "popularity": ranking_metrics(popularity_rankings, truth, evaluation_k),
        "matrix_factorization": ranking_metrics(mf_rankings, truth, evaluation_k),
        "two_tower": ranking_metrics(two_tower_rankings, truth, evaluation_k),
    }
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2))
    pd.DataFrame(
        {
            "split": ["train", "validation", "test"],
            "rows": [len(split.train), len(split.validation), len(split.test)],
        }
    ).to_json(output / "split_summary.json", orient="records", indent=2)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and evaluate RecoFlow V1")
    parser.add_argument("transactions", help="H&M transactions_train.csv or Parquet")
    parser.add_argument("articles", help="H&M articles.csv")
    parser.add_argument("--output-dir", default="artifacts")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--max-history", type=int, default=20)
    parser.add_argument("--evaluation-k", type=int, default=100)
    parser.add_argument("--max-eval-users", type=int, default=5_000)
    parser.add_argument("--max-rows", type=int)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    metrics = train(
        transactions_path=args.transactions,
        articles_path=args.articles,
        output_dir=args.output_dir,
        epochs=args.epochs,
        embedding_dim=args.embedding_dim,
        batch_size=args.batch_size,
        max_history=args.max_history,
        evaluation_k=args.evaluation_k,
        max_eval_users=args.max_eval_users,
        max_rows=args.max_rows,
        seed=args.seed,
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
