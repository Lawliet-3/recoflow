import json

import pandas as pd

from recoflow.artifacts import load_embeddings
from recoflow.train import train


def test_training_exports_serving_and_evaluation_artifacts(tmp_path):
    transactions = []
    article_ids = [f"{index:010d}" for index in range(1, 7)]
    for day, date in enumerate(pd.date_range("2024-01-01", periods=24, freq="D")):
        for user in range(3):
            transactions.append(
                {
                    "t_dat": date.strftime("%Y-%m-%d"),
                    "customer_id": f"u{user}",
                    "article_id": article_ids[(day + user) % len(article_ids)],
                }
            )
    transaction_path = tmp_path / "transactions.csv"
    pd.DataFrame(transactions).to_csv(transaction_path, index=False)
    article_path = tmp_path / "articles.csv"
    pd.DataFrame(
        {
            "article_id": article_ids,
            "product_type_no": [1, 1, 2, 2, 3, 3],
            "colour_group_code": [1, 2, 1, 2, 1, 2],
            "department_no": [1, 1, 1, 2, 2, 2],
            "index_group_no": [1, 1, 1, 1, 2, 2],
        }
    ).to_csv(article_path, index=False)

    output = tmp_path / "artifacts"
    metrics = train(
        str(transaction_path),
        str(article_path),
        output_dir=str(output),
        epochs=1,
        embedding_dim=8,
        batch_size=8,
        max_history=3,
        evaluation_k=3,
        max_eval_users=3,
    )

    assert set(metrics) == {"popularity", "matrix_factorization", "two_tower"}
    assert json.loads((output / "metrics.json").read_text()) == metrics
    user_ids, user_vectors = load_embeddings(output / "user_embeddings.npz")
    item_ids, item_vectors = load_embeddings(output / "item_embeddings.npz")
    assert len(user_ids) == len(user_vectors) == 3
    assert len(item_ids) == len(item_vectors) == 6
    assert (output / "matrix_factorization.pt").exists()
    assert (output / "two_tower.pt").exists()
