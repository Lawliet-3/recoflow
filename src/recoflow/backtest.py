from __future__ import annotations

import argparse
import gc
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset

from recoflow.baseline import PopularityRecommender
from recoflow.data import (
    build_training_examples,
    encode_item_metadata,
    load_articles,
    load_transactions,
)
from recoflow.evaluation import (
    exact_recommendations,
    metrics_with_confidence,
    popularity_recommendations,
    seen_items_by_user,
)
from recoflow.train import _seed_everything, _train_matrix_factorization, _train_two_tower
from recoflow.vector_store import ProductVectorStore


@dataclass(frozen=True)
class BacktestFold:
    name: str
    evaluation_start: pd.Timestamp
    evaluation_end: pd.Timestamp
    is_holdout: bool = False


def make_backtest_folds(
    transactions: pd.DataFrame,
    validation_folds: int = 3,
    evaluation_days: int = 7,
    holdout_days: int = 7,
    include_holdout: bool = False,
) -> list[BacktestFold]:
    """Build expanding-window validation folds before a separately gated holdout."""
    if validation_folds < 0:
        raise ValueError("validation_folds cannot be negative")
    if validation_folds == 0 and not include_holdout:
        raise ValueError("request at least one validation fold or include the holdout")
    if evaluation_days < 1 or holdout_days < 1:
        raise ValueError("evaluation_days and holdout_days must be positive")
    end = pd.to_datetime(transactions["t_dat"]).max().normalize() + pd.Timedelta(days=1)
    holdout_start = end - pd.Timedelta(days=holdout_days)
    folds: list[BacktestFold] = []
    for index in range(validation_folds):
        periods_before_holdout = validation_folds - index
        evaluation_end = holdout_start - pd.Timedelta(
            days=evaluation_days * (periods_before_holdout - 1)
        )
        evaluation_start = evaluation_end - pd.Timedelta(days=evaluation_days)
        folds.append(
            BacktestFold(
                name=f"validation_{index + 1}",
                evaluation_start=evaluation_start,
                evaluation_end=evaluation_end,
            )
        )
    if include_holdout:
        folds.append(
            BacktestFold(
                name="final_holdout",
                evaluation_start=holdout_start,
                evaluation_end=end,
                is_holdout=True,
            )
        )
    first_event = pd.to_datetime(transactions["t_dat"]).min()
    if folds and folds[0].evaluation_start <= first_event:
        raise ValueError("date range is too short for the requested rolling folds")
    return folds


def _ground_truth(transactions: pd.DataFrame) -> dict[str, set[str]]:
    return {
        str(user): set(group["article_id"].astype(str))
        for user, group in transactions.groupby("customer_id")
    }


def _customer_segment(history_count: int) -> str:
    if history_count == 0:
        return "cold"
    if history_count < 5:
        return "light_1_4"
    if history_count < 20:
        return "regular_5_19"
    return "heavy_20_plus"


def _policy_truth(
    truth: dict[str, set[str]],
    seen: dict[str, set[str]],
    policy: str,
) -> dict[str, set[str]]:
    if policy == "overall":
        return truth
    if policy == "discovery":
        return {user: items.difference(seen.get(user, set())) for user, items in truth.items()}
    if policy == "repeat":
        return {user: items.intersection(seen.get(user, set())) for user, items in truth.items()}
    raise ValueError(f"unknown policy: {policy}")


def _rankings_for_policy(
    policy: str,
    user_ids: list[str],
    popular_items: list[str],
    seen: dict[str, set[str]],
    k: int,
    warm_user_ids: list[str] | None = None,
    warm_user_vectors: np.ndarray | None = None,
    item_ids: list[str] | None = None,
    item_vectors: np.ndarray | None = None,
) -> dict[str, list[str]]:
    excluded = seen if policy == "discovery" else {}
    included = seen if policy == "repeat" else None
    fallback = popularity_recommendations(
        user_ids,
        popular_items,
        excluded,
        k,
        include_items=included,
    )
    if warm_user_ids is None or not warm_user_ids:
        return fallback
    assert warm_user_vectors is not None and item_ids is not None and item_vectors is not None
    personalized = exact_recommendations(
        warm_user_ids,
        warm_user_vectors,
        item_ids,
        item_vectors,
        excluded,
        k,
        include_items=included,
    )
    fallback.update(personalized)
    return fallback


def _evaluate_rankings(
    rankings: dict[str, list[str]],
    truth: dict[str, set[str]],
    segments: dict[str, str],
    k: int,
    bootstrap_samples: int,
    seed: int,
) -> dict[str, object]:
    by_segment: dict[str, object] = {}
    segment_names = ("cold", "warm", "light_1_4", "regular_5_19", "heavy_20_plus")
    for segment in segment_names:
        segment_truth = {
            user: relevant
            for user, relevant in truth.items()
            if (segments[user] != "cold" if segment == "warm" else segments[user] == segment)
            and relevant
        }
        by_segment[segment] = metrics_with_confidence(
            rankings,
            segment_truth,
            k,
            bootstrap_samples=bootstrap_samples,
            seed=seed,
        )
    return {
        "all": metrics_with_confidence(
            rankings,
            {user: relevant for user, relevant in truth.items() if relevant},
            k,
            bootstrap_samples=bootstrap_samples,
            seed=seed,
        ),
        "segments": by_segment,
    }


def _qdrant_parity(
    user_ids: list[str],
    user_vectors: np.ndarray,
    item_ids: list[str],
    item_vectors: np.ndarray,
    k: int,
) -> dict[str, float | int]:
    if not user_ids:
        return {"evaluated_users": 0, f"mean_top_{k}_overlap": 0.0}
    store = ProductVectorStore(":memory:", "backtest-parity", item_vectors.shape[1])
    store.ensure_collection()
    for start in range(0, len(item_ids), 1_000):
        end = start + 1_000
        store.upsert(item_ids[start:end], item_vectors[start:end].tolist(), start_index=start)
    exact = exact_recommendations(user_ids, user_vectors, item_ids, item_vectors, {}, k)
    overlaps: list[float] = []
    for user_id, vector in zip(user_ids, user_vectors, strict=True):
        qdrant_ids = {
            str(item["article_id"]) for item in store.search(vector.tolist(), limit=k)
        }
        exact_ids = set(exact[user_id])
        overlaps.append(len(qdrant_ids.intersection(exact_ids)) / max(len(exact_ids), 1))
    store.client.close()
    return {
        "evaluated_users": len(overlaps),
        f"mean_top_{k}_overlap": float(np.mean(overlaps)),
        f"minimum_top_{k}_overlap": float(np.min(overlaps)),
    }


def _evaluate_fold(
    transactions: pd.DataFrame,
    articles: pd.DataFrame,
    fold: BacktestFold,
    epochs: int,
    embedding_dim: int,
    batch_size: int,
    max_history: int,
    k: int,
    max_eval_users: int,
    bootstrap_samples: int,
    qdrant_parity_users: int,
    seed: int,
) -> dict[str, object]:
    _seed_everything(seed)
    dates = pd.to_datetime(transactions["t_dat"])
    train = transactions.loc[dates < fold.evaluation_start].copy()
    evaluation = transactions.loc[
        (dates >= fold.evaluation_start) & (dates < fold.evaluation_end)
    ].copy()
    if train.empty or evaluation.empty:
        raise ValueError(f"{fold.name} has an empty train or evaluation window")

    users = sorted(train["customer_id"].astype(str).unique())
    items = sorted(train["article_id"].astype(str).unique())
    user_to_index = {value: index for index, value in enumerate(users)}
    item_to_index = {value: index + 1 for index, value in enumerate(items)}
    encoded = build_training_examples(train, user_to_index, item_to_index, max_history)
    metadata_array, cardinalities = encode_item_metadata(articles, items)
    metadata = torch.from_numpy(metadata_array)
    dataset = TensorDataset(
        torch.from_numpy(encoded.user_indices),
        torch.from_numpy(encoded.item_indices),
        torch.from_numpy(encoded.history_indices),
        torch.from_numpy(encoded.history_mask),
    )
    popularity = PopularityRecommender().fit(train)
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

    with torch.no_grad():
        all_user_indices = torch.arange(len(users))
        all_item_indices = torch.arange(1, len(items) + 1)
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

    truth = _ground_truth(evaluation)
    evaluation_rng = np.random.default_rng(seed)
    eval_users = np.asarray(sorted(truth))
    if len(eval_users) > max_eval_users:
        eval_users = np.sort(
            evaluation_rng.choice(eval_users, size=max_eval_users, replace=False)
        )
    eval_user_ids = eval_users.tolist()
    truth = {user: truth[user] for user in eval_user_ids}
    seen = seen_items_by_user(train)
    history_counts = train["customer_id"].astype(str).value_counts().to_dict()
    segments = {
        user: _customer_segment(int(history_counts.get(user, 0))) for user in eval_user_ids
    }
    warm_user_ids = [user for user in eval_user_ids if user in user_to_index]
    warm_positions = [user_to_index[user] for user in warm_user_ids]

    model_vectors = {
        "matrix_factorization": (mf_user_vectors[warm_positions], mf_item_vectors),
        "two_tower": (two_tower_user_vectors[warm_positions], two_tower_item_vectors),
    }
    results: dict[str, object] = {}
    for model_name in ("popularity", "matrix_factorization", "two_tower"):
        policy_results: dict[str, object] = {}
        for policy in ("overall", "discovery", "repeat"):
            policy_truth = _policy_truth(truth, seen, policy)
            if model_name == "popularity":
                rankings = _rankings_for_policy(
                    policy, eval_user_ids, popularity.ranked_items, seen, k
                )
            else:
                warm_vectors, model_item_vectors = model_vectors[model_name]
                rankings = _rankings_for_policy(
                    policy,
                    eval_user_ids,
                    popularity.ranked_items,
                    seen,
                    k,
                    warm_user_ids,
                    warm_vectors,
                    items,
                    model_item_vectors,
                )
            policy_results[policy] = _evaluate_rankings(
                rankings,
                policy_truth,
                segments,
                k,
                bootstrap_samples,
                seed,
            )
        results[model_name] = policy_results

    relevant_items = sum((len(relevant) for relevant in truth.values()), start=0)
    eligible_relevant_items = sum(
        len(relevant.intersection(item_to_index)) for relevant in truth.values()
    )
    repeat_items = sum(
        len(relevant.intersection(seen.get(user, set()))) for user, relevant in truth.items()
    )
    parity_count = min(qdrant_parity_users, len(warm_user_ids))
    parity = _qdrant_parity(
        warm_user_ids[:parity_count],
        two_tower_user_vectors[warm_positions[:parity_count]],
        items,
        two_tower_item_vectors,
        min(k, 100),
    )
    return {
        "name": fold.name,
        "is_holdout": fold.is_holdout,
        "evaluation_start": fold.evaluation_start.date().isoformat(),
        "evaluation_end_exclusive": fold.evaluation_end.date().isoformat(),
        "train_rows": len(train),
        "evaluation_rows": len(evaluation),
        "sampled_users": len(eval_user_ids),
        "warm_user_rate": len(warm_user_ids) / max(len(eval_user_ids), 1),
        "eligible_target_rate": eligible_relevant_items / max(relevant_items, 1),
        "repeat_target_rate": repeat_items / max(relevant_items, 1),
        "segment_users": {
            segment: sum(
                value != "cold" if segment == "warm" else value == segment
                for value in segments.values()
            )
            for segment in ("cold", "warm", "light_1_4", "regular_5_19", "heavy_20_plus")
        },
        "qdrant_parity": parity,
        "models": results,
    }


def _mean_summary(folds: list[dict[str, object]], k: int) -> dict[str, object]:
    summary: dict[str, object] = {}
    for model in ("popularity", "matrix_factorization", "two_tower"):
        summary[model] = {}
        for policy in ("overall", "discovery", "repeat"):
            model_policy = summary[model]
            assert isinstance(model_policy, dict)
            model_policy[policy] = {
                metric: float(
                    np.mean(
                        [
                            fold["models"][model][policy]["all"][metric]["mean"]
                            for fold in folds
                        ]
                    )
                )
                if folds
                else None
                for metric in (f"recall@{k}", f"ndcg@{k}", "mrr")
            }
    return summary


def run_backtest(
    transactions_path: str,
    articles_path: str,
    output_path: str = "artifacts/backtest.json",
    validation_folds: int = 3,
    evaluation_days: int = 7,
    holdout_days: int = 7,
    include_holdout: bool = False,
    epochs: int = 3,
    embedding_dim: int = 64,
    batch_size: int = 512,
    max_history: int = 20,
    k: int = 100,
    max_eval_users: int = 5_000,
    bootstrap_samples: int = 500,
    qdrant_parity_users: int = 25,
    max_rows: int | None = None,
    seed: int = 42,
) -> dict[str, object]:
    transactions = load_transactions(transactions_path, max_rows=max_rows)
    articles = load_articles(articles_path)
    folds = make_backtest_folds(
        transactions,
        validation_folds=validation_folds,
        evaluation_days=evaluation_days,
        holdout_days=holdout_days,
        include_holdout=include_holdout,
    )
    fold_results: list[dict[str, object]] = []
    for index, fold in enumerate(folds):
        print(
            f"Running {fold.name}: {fold.evaluation_start.date()} to "
            f"{fold.evaluation_end.date()} (exclusive)"
        )
        fold_result = _evaluate_fold(
            transactions,
            articles,
            fold,
            epochs,
            embedding_dim,
            batch_size,
            max_history,
            k,
            max_eval_users,
            bootstrap_samples,
            qdrant_parity_users,
            seed + index,
        )
        fold_results.append(fold_result)
        print(
            f"Completed {fold.name}: {fold_result['sampled_users']} users, "
            f"warm-user rate {fold_result['warm_user_rate']:.1%}"
        )
        gc.collect()
    report = {
        "protocol": {
            "validation_folds": validation_folds,
            "evaluation_days": evaluation_days,
            "holdout_days": holdout_days,
            "holdout_consumed": include_holdout,
            "catalog": "products observed before each evaluation window",
            "cold_user_policy": "popularity fallback",
            "policies": {
                "overall": "seen and unseen products are eligible, matching the current API",
                "discovery": "previously purchased products are excluded",
                "repeat": "only previously purchased products are eligible",
            },
            "seed": seed,
        },
        "validation_summary": _mean_summary(
            [fold for fold in fold_results if not fold["is_holdout"]], k
        ),
        "holdout_summary": _mean_summary(
            [fold for fold in fold_results if fold["is_holdout"]], k
        ),
        "folds": fold_results,
    }
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Run production-shaped rolling evaluation")
    parser.add_argument("transactions")
    parser.add_argument("articles")
    parser.add_argument("--output", default="artifacts/backtest.json")
    parser.add_argument("--validation-folds", type=int, default=3)
    parser.add_argument("--evaluation-days", type=int, default=7)
    parser.add_argument("--holdout-days", type=int, default=7)
    parser.add_argument("--include-holdout", action="store_true")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--max-history", type=int, default=20)
    parser.add_argument("--k", type=int, default=100)
    parser.add_argument("--max-eval-users", type=int, default=5_000)
    parser.add_argument("--bootstrap-samples", type=int, default=500)
    parser.add_argument("--qdrant-parity-users", type=int, default=25)
    parser.add_argument("--max-rows", type=int)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    report = run_backtest(
        args.transactions,
        args.articles,
        output_path=args.output,
        validation_folds=args.validation_folds,
        evaluation_days=args.evaluation_days,
        holdout_days=args.holdout_days,
        include_holdout=args.include_holdout,
        epochs=args.epochs,
        embedding_dim=args.embedding_dim,
        batch_size=args.batch_size,
        max_history=args.max_history,
        k=args.k,
        max_eval_users=args.max_eval_users,
        bootstrap_samples=args.bootstrap_samples,
        qdrant_parity_users=args.qdrant_parity_users,
        max_rows=args.max_rows,
        seed=args.seed,
    )
    displayed_summary = (
        report["validation_summary"] if args.validation_folds else report["holdout_summary"]
    )
    print(json.dumps(displayed_summary, indent=2))


if __name__ == "__main__":
    main()
