# RecoFlow

RecoFlow is a production-shaped recommendation retrieval service built around the
[H&M Personalized Fashion Recommendations](https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations/data)
dataset. V1 deliberately focuses on a measurable backbone: time-safe evaluation, a
popularity benchmark, two-tower candidate retrieval, Qdrant indexing, and a small
FastAPI serving layer.

## V1 architecture

```text
H&M transactions.csv
        │
        ├── temporal train / validation / test split
        ├── popularity baseline ───────────────────────┐
        └── PyTorch two-tower model                    │
                    │                                  │
             product embeddings → Qdrant              │
                    │                                  │
                    └──────── FastAPI retrieval ←──────┘
                               personalized       cold start
```

The split is based on event time rather than random rows, so future purchases cannot
leak into training. The popularity model is both an honest offline baseline and the
serving fallback for unknown users or an unavailable vector store.

## Run locally

Prerequisites: Python 3.11+ and Docker.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
docker compose up -d qdrant
pytest
```

Download the Kaggle data separately and keep it under `data/raw/` (ignored by Git).
Train the first artifacts with:

```bash
recoflow-train data/raw/transactions_train.csv --epochs 3
```

Start the service:

```bash
uvicorn recoflow.api:app --reload
curl 'http://localhost:8000/v1/recommendations/customer-id?limit=10'
```

Environment variables use the `RECOFLOW_` prefix; for example,
`RECOFLOW_QDRANT_URL`, `RECOFLOW_EMBEDDING_DIM`, and `RECOFLOW_POPULARITY_PATH`.

## Repository map

- `src/recoflow/data.py` validates transactions and creates temporal splits.
- `src/recoflow/baseline.py` provides the popularity benchmark and fallback.
- `src/recoflow/model.py` contains normalized user/item towers and in-batch loss.
- `src/recoflow/vector_store.py` owns Qdrant collection, indexing, and search.
- `src/recoflow/api.py` exposes health and recommendation endpoints.
- `src/recoflow/train.py` builds reproducible V1 artifacts.
- `tests/` covers leakage boundaries, ranking, model behavior, and API fallback.

## V1 completion path

The next increments are: export learned product and user embeddings, index products in
Qdrant, add Recall@K/MRR evaluation against the held-out weeks, enrich the item tower
with H&M metadata and image/text features, and add CI plus latency and drift metrics.

Large datasets, checkpoints, and generated artifacts must remain outside Git.
