# RecoFlow

RecoFlow V1 is an end-to-end recommendation retrieval system built around the
[H&M Personalized Fashion Recommendations](https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations/data)
dataset. It trains three progressively stronger recommenders, evaluates them on future
purchases, exports serving artifacts, indexes product vectors in Qdrant, and serves
recommendations through FastAPI.

## What V1 proves

~~~text
transactions_train.csv + articles.csv
                 │
        validate and sort by time
                 │
       train / validation / test
                 │
     ┌───────────┼──────────────────┐
     │           │                  │
 popularity   matrix           two-tower
 baseline     factorization    history + metadata
     │           │                  │
     └───────────┴──── offline evaluation
                                │
                    user and item embeddings
                                │
                    Qdrant cosine retrieval
                                │
                    FastAPI recommendation API
~~~

This separation matters. Training learns representations, Qdrant performs fast
nearest-neighbour retrieval, and the API owns serving and fallback behavior.

## Set up

Prerequisites are Python 3.11+ and Docker.

~~~bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
docker compose up -d qdrant
pytest
~~~

Download transactions_train.csv and articles.csv from Kaggle and place them in
data/raw/. Raw data and generated artifacts are ignored by Git.

## Step 1: train, compare, and export

Start with a bounded run while reviewing the pipeline:

~~~bash
recoflow-train \
  data/raw/transactions_train.csv \
  data/raw/articles.csv \
  --max-rows 1000000 \
  --epochs 1
~~~

Remove --max-rows and increase the epochs after the small run is healthy.

~~~bash
recoflow-train \
  data/raw/transactions_train.csv \
  data/raw/articles.csv \
  --epochs 3 \
  --embedding-dim 64
~~~

Training produces:

| Artifact | Purpose |
|---|---|
| popularity.json | Cold-start and dependency-failure fallback |
| matrix_factorization.pt | Collaborative-filtering comparison model |
| two_tower.pt | Reproducible model checkpoint and vocabulary |
| user_embeddings.npz | User IDs aligned with serving vectors |
| item_embeddings.npz | Product IDs aligned with Qdrant vectors |
| metrics.json | Popularity, MF, and two-tower Recall/NDCG/MRR |
| split_summary.json | Auditable row counts for each time window |

## Step 2: index products

~~~bash
recoflow-index --recreate
~~~

The index command reads item_embeddings.npz, creates the cosine-distance collection,
and uploads vectors in batches. --recreate prevents products from an older artifact
version remaining in the collection.

## Step 3: serve recommendations

~~~bash
uvicorn recoflow.api:app --reload
~~~

~~~bash
curl 'http://localhost:8000/health'
curl 'http://localhost:8000/v1/recommendations/CUSTOMER_ID?limit=10'
~~~

Known customers use their two-tower vector to search Qdrant. Unknown customers, or
requests made while Qdrant is unavailable, receive the popularity fallback.

With the API running, verify the entire trained-artifact → Qdrant → API path:

~~~bash
recoflow-smoke
~~~

The smoke command selects a known user from the exported artifact and fails unless
the live endpoint returns a personalized two-tower result.

Configuration uses the RECOFLOW_ prefix. Copy .env.example to .env to override the
Qdrant address, collection, model version, artifact paths, or default result count.

## How the models differ

- Popularity has no personalization. It answers whether learned models beat a sensible
  default.
- Matrix factorization learns only user and item IDs. It measures the value of
  collaborative behavior.
- The two-tower model combines user ID with prior purchases, and product ID with H&M
  product type, colour, department, and index-group metadata.

Every training example uses purchases that happened before its target purchase.
Validation comes from a later week, and the final week remains reserved as a test set.
Previously purchased products are removed during offline retrieval.

Read [the V1 walkthrough](docs/v1-walkthrough.md) for the reasoning behind every stage
and the meaning of the evaluation metrics.

## Repository map

- src/recoflow/data.py validates H&M files, splits time, and builds prefix histories.
- src/recoflow/model.py defines matrix factorization and the two-tower encoders.
- src/recoflow/evaluation.py performs exact offline retrieval and ranking metrics.
- src/recoflow/artifacts.py defines the training-to-serving embedding contract.
- src/recoflow/train.py trains, evaluates, checkpoints, and exports.
- src/recoflow/index.py uploads exported product vectors into Qdrant.
- src/recoflow/vector_store.py owns Qdrant collection, batch upsert, and search.
- src/recoflow/api.py serves personalized results with a popularity fallback.
- tests/ covers split leakage, histories, models, metrics, artifacts, Qdrant, API, and
  a small end-to-end training run.

## V1 boundaries

V1 uses static offline user vectors and keeps preprocessing in memory. Offline
evaluation uses exact dot-product search as the quality reference, while production
retrieval uses Qdrant. Image and text encoders, learned ranking, real-time events,
session features, monitoring, and orchestration belong to later versions.
