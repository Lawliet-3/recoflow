# V1.1 evaluation protocol

This protocol answers a narrower question than an online experiment: if RecoFlow had
been retrained at a historical date, how well would its recommendations have matched
purchases in the following week?

## 1. Rolling validation

The harness creates expanding training windows followed by non-overlapping evaluation
weeks. A three-fold run near the end of the dataset looks like this:

~~~text
fold 1  [training────────────][evaluate]
fold 2  [training────────────────────][evaluate]
fold 3  [training────────────────────────────][evaluate]
holdout [kept closed until model choices are frozen──────────][final test]
~~~

Every fold trains new popularity, matrix-factorization, and two-tower models. No event
from an evaluation week can enter that fold's training histories or embeddings.

The final holdout is excluded by default. `--include-holdout` is an explicit record
that the period has been consumed and must no longer be used for tuning.

## 2. Real serving population

Evaluation samples from every customer who purchased during the evaluation week. It
does not discard new customers. Customers absent from training use the same popularity
fallback as the API.

Results are shown for:

- cold customers with no training history;
- all warm customers;
- light customers with 1–4 prior transactions;
- regular customers with 5–19 prior transactions;
- heavy customers with at least 20 prior transactions.

`warm_user_rate` makes the personalized model's reachable share visible. The catalog
contains products observed before the evaluation window. `eligible_target_rate` shows
how much future ground truth that historical catalog could possibly retrieve.

H&M does not provide daily stock snapshots. “Observed before the window” is therefore
a reproducible catalog proxy, not proof that an item was actually in stock.

## 3. Three recommendation policies

The report avoids mixing two different product jobs into one unexplained score:

- **Overall** allows seen and unseen products. This matches the current API behavior.
- **Discovery** removes previously purchased products from the candidate set and target.
- **Repeat** limits candidates and targets to previously purchased products.

At `k=100`, repeat recall can saturate when a customer has fewer than 100 historical
products. NDCG and MRR remain useful because they measure how early the repeated item
appears. A later ranking system may use a smaller `k` for repeat-oriented surfaces.

## 4. Metrics and uncertainty

Recall measures the fraction of purchased products retrieved. NDCG rewards relevant
products near the top. MRR measures the position of the first relevant product.

Metrics are first calculated per customer and then macro-averaged, so highly active
customers do not dominate the report. A deterministic user bootstrap produces a 95%
interval around every aggregate. Intervals communicate sampling uncertainty; they do
not correct for missing exposure, stock, price, promotion, or causal effects.

## 5. Retrieval parity

Offline model scoring uses exact dot products. The harness also uploads item vectors to
an in-memory Qdrant collection and compares its top results with exact retrieval for a
sample of warm customers. This detects indexing, distance-metric, or ID-alignment bugs.

## 6. Reading the result

Use rolling validation to choose models and settings. Prefer a challenger only when it
wins across multiple weeks and important customer segments, and its confidence ranges
do not suggest a fragile one-week result. Then run the final holdout once.

Even a strong holdout result is not a conversion or revenue estimate. Live impact still
requires an online experiment with impression, click, purchase, stock, and business
outcome logging.
