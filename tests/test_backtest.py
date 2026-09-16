import pandas as pd

from recoflow.backtest import make_backtest_folds, run_backtest


def _fixture_files(tmp_path):
    article_ids = [f"{index:010d}" for index in range(1, 9)]
    transactions = []
    for day, date in enumerate(pd.date_range("2024-01-01", periods=35, freq="D")):
        for user in range(4):
            transactions.append(
                {
                    "t_dat": date.strftime("%Y-%m-%d"),
                    "customer_id": f"u{user}",
                    "article_id": article_ids[(day + user) % len(article_ids)],
                }
            )
        if day >= 28:
            transactions.append(
                {
                    "t_dat": date.strftime("%Y-%m-%d"),
                    "customer_id": "cold-user",
                    "article_id": article_ids[day % len(article_ids)],
                }
            )
    transaction_path = tmp_path / "transactions.csv"
    pd.DataFrame(transactions).to_csv(transaction_path, index=False)
    article_path = tmp_path / "articles.csv"
    pd.DataFrame(
        {
            "article_id": article_ids,
            "product_type_no": [1, 1, 2, 2, 3, 3, 4, 4],
            "colour_group_code": [1, 2, 1, 2, 1, 2, 1, 2],
            "department_no": [1, 1, 1, 2, 2, 2, 3, 3],
            "index_group_no": [1, 1, 1, 1, 2, 2, 2, 2],
        }
    ).to_csv(article_path, index=False)
    return transaction_path, article_path


def test_folds_keep_final_period_gated_until_requested():
    frame = pd.DataFrame(
        {
            "t_dat": pd.date_range("2024-01-01", periods=35, freq="D"),
            "customer_id": ["u"] * 35,
            "article_id": ["i"] * 35,
        }
    )
    validation = make_backtest_folds(frame, validation_folds=2)
    assert [fold.name for fold in validation] == ["validation_1", "validation_2"]
    assert all(not fold.is_holdout for fold in validation)
    with_holdout = make_backtest_folds(frame, validation_folds=2, include_holdout=True)
    assert with_holdout[-1].name == "final_holdout"
    assert with_holdout[-1].is_holdout
    assert validation[-1].evaluation_end == with_holdout[-1].evaluation_start


def test_backtest_reports_cold_users_policies_segments_and_qdrant(tmp_path):
    transactions, articles = _fixture_files(tmp_path)
    output = tmp_path / "backtest.json"
    report = run_backtest(
        str(transactions),
        str(articles),
        output_path=str(output),
        validation_folds=1,
        include_holdout=True,
        epochs=1,
        embedding_dim=8,
        batch_size=16,
        max_history=4,
        k=3,
        max_eval_users=10,
        bootstrap_samples=20,
        qdrant_parity_users=2,
    )
    assert output.exists()
    assert report["protocol"]["holdout_consumed"] is True
    assert report["validation_summary"]["popularity"]["overall"]["ndcg@3"] is not None
    assert report["holdout_summary"]["popularity"]["overall"]["ndcg@3"] is not None
    assert [fold["name"] for fold in report["folds"]] == [
        "validation_1",
        "final_holdout",
    ]
    holdout = report["folds"][-1]
    assert holdout["segment_users"]["cold"] == 1
    assert holdout["warm_user_rate"] < 1.0
    assert holdout["qdrant_parity"]["evaluated_users"] == 2
    for model in ("popularity", "matrix_factorization", "two_tower"):
        assert set(holdout["models"][model]) == {"overall", "discovery", "repeat"}
