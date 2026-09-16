# RecoFlow V1 walkthrough

This document follows one recommendation from raw events to the API response.

## 1. Validate and order the events

load_transactions requires a date, customer ID, and article ID. IDs are kept as
strings because H&M article IDs contain meaningful leading zeroes. Events are sorted
by timestamp before any history is constructed.

Why: treating IDs as numbers can silently change them, and unsorted interactions would
let a user representation learn from the future.

## 2. Split by time

The final seven days become the test set and the preceding seven days become
validation. Everything earlier is training data.

~~~text
past                                future
|----------- train -----------| validation | test |
~~~

Validation guides development. The test window remains untouched until a version is
ready for a final report. A random row split would allow later purchases from the same
customer to influence an earlier prediction.

## 3. Establish two baselines

Popularity ranks products by training-window purchase count. It is difficult to beat
for new customers and is safe when personalization dependencies fail.

Matrix factorization gives every known user and product a learned vector. Bayesian
Personalized Ranking loss asks the observed product to score above a sampled
non-target product. This isolates the value of collaborative behavior before metadata
or history is introduced.

## 4. Build leakage-safe user histories

For each training purchase, the user tower receives only preceding purchases:

~~~text
target A → history []
target B → history [A]
target C → history [A, B]
~~~

Histories are truncated to the most recent 20 items by default and padded for batch
training. A mask ensures padding contributes nothing to the average history vector.

## 5. Add item metadata

The item tower combines embeddings for:

- Article ID
- Product type
- Colour group
- Department
- Index group

Each categorical value becomes an integer vocabulary entry. Zero is reserved for
missing or unknown values. The embeddings are concatenated and projected into the
same vector space as users.

## 6. Train the two towers

The user tower joins the customer-ID embedding with the average prior-purchase
embedding. The item tower joins the product-ID and metadata embeddings. Both outputs
are normalized, so their dot product is cosine similarity.

A batch of observed user-product pairs provides in-batch negatives. The diagonal of
the similarity matrix contains positive pairs; other products in the batch act as
negative examples. When an item occurs more than once in a batch, every matching copy
is treated as positive rather than as a false negative. Cross-entropy pulls the
positive pairs together.

## 7. Evaluate future retrieval

The evaluator embeds validation users and every training-catalog product, computes
exact dot products, removes previously purchased products, and returns the top K.
Exact search is intentional here: it is the reference used to tell whether an
approximate Qdrant index loses relevant results.

- Recall@K measures how many future purchases appear in the first K results.
- NDCG@K rewards relevant products more when they appear near the top.
- MRR measures the reciprocal position of the first relevant result.

Popularity, matrix factorization, and two-tower retrieval use the same users, product
catalog, exclusions, and ground truth.

## 8. Export a serving contract

Training writes compressed archives with two arrays:

~~~text
ids[i] ↔ vectors[i]
~~~

The item archive feeds Qdrant. The user archive feeds the API. Keeping IDs and vectors
in the same artifact prevents ordering differences between training and serving.
Checkpoints are also saved for reproducibility and later retraining.

## 9. Index products in Qdrant

recoflow-index reads the product artifact, creates a cosine collection, and uploads
vectors in batches. Each Qdrant point stores the article ID as payload, so a vector
result can be translated back to a catalog product.

## 10. Serve and fail safely

For a known customer, FastAPI loads the exported user vector and asks Qdrant for the
nearest products. The response reports source=two_tower and a model version.

An unknown customer has no learned vector. If Qdrant is temporarily unavailable, the
service also cannot complete personalized retrieval. Both cases use the popularity
artifact and report source=popularity.

This fallback is part of the model design: a recommendation endpoint should remain
useful when personalization is impossible.
