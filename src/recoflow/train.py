from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from recoflow.baseline import PopularityRecommender
from recoflow.data import load_transactions, temporal_split
from recoflow.model import TwoTowerModel, in_batch_softmax_loss


def train(transactions_path: str, output_dir: str, epochs: int, embedding_dim: int) -> None:
    transactions = load_transactions(transactions_path)
    split = temporal_split(transactions)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    popularity = PopularityRecommender().fit(split.train)
    (output / "popularity.json").write_text(json.dumps(popularity.ranked_items))

    users = sorted(split.train["customer_id"].astype(str).unique())
    items = sorted(split.train["article_id"].astype(str).unique())
    user_to_index = {value: index for index, value in enumerate(users)}
    item_to_index = {value: index for index, value in enumerate(items)}
    user_tensor = torch.tensor(split.train["customer_id"].astype(str).map(user_to_index).to_numpy())
    item_tensor = torch.tensor(split.train["article_id"].astype(str).map(item_to_index).to_numpy())
    loader = DataLoader(TensorDataset(user_tensor, item_tensor), batch_size=512, shuffle=True)

    model = TwoTowerModel(len(users), len(items), embedding_dim)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    model.train()
    for _ in range(epochs):
        for user_batch, item_batch in loader:
            loss = in_batch_softmax_loss(model(user_batch, item_batch))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    torch.save(
        {"state_dict": model.state_dict(), "users": users, "items": items},
        output / "two_tower.pt",
    )
    pd.DataFrame(
        {"split": ["train", "validation", "test"],
         "rows": [len(split.train), len(split.validation), len(split.test)]}
    ).to_json(output / "split_summary.json", orient="records", indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train RecoFlow V1 retrieval artifacts")
    parser.add_argument("transactions")
    parser.add_argument("--output-dir", default="artifacts")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--embedding-dim", type=int, default=64)
    args = parser.parse_args()
    train(args.transactions, args.output_dir, args.epochs, args.embedding_dim)


if __name__ == "__main__":
    main()

