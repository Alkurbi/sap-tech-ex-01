# Product Similarity Search

A service that finds similar products in the Amazon India fashion dataset (~30,000 items).
Given a product ID, it returns the `N` most similar product IDs.

It combines three kinds of signal — text, numeric, and image — into a single similarity
score with configurable weights, and falls back gracefully when a signal is missing. It
also includes an approximate nearest-neighbour index (HNSW) so search stays fast on the
full dataset.

## How it works

Each product is turned into vectors:

- text: product name, meta keywords, brand, category, and colour, combined into one
  string and turned into a vector with TF-IDF + SVD (or, optionally, transformer embeddings).
- numeric: `log(sales_price)` and `rating`, standardized.
- image: features from the product image using CLIP or ResNet (optional).

To compare two products, I compute a cosine similarity for each part separately and then
take a weighted average of those scores. This is called late fusion. If a product has no
usable image, the image part is dropped and the remaining weights are renormalized, so the
product is still scored fairly on the signals it does have.

For speed, the parts are also concatenated into a single vector and indexed with HNSW. A
query first pulls a small set of candidates from the index, and then those candidates are
re-scored with the weighted late-fusion score. This keeps the fast index lookup while still
applying the per-query weights and fallback.

## Project layout

```
product_similarity/      core package
  config.py              settings (backends, weights, file paths)
  clean.py               raw .ldjson  ->  tidy parquet
  features.py            builds text / numeric / image vectors, fuses them
  image_features.py      downloads images and embeds them (ResNet / CLIP)
  search.py              Part 1: exact search over all products
  ann.py                 Part 3: HNSW index + re-rank (the main search path)
api/
  app.py                 FastAPI service
k8s/                     Kubernetes deployment + service manifests
Dockerfile               lightweight image (TF-IDF, no torch)
Dockerfile.full          full image (transformer text + CLIP images)
requirements.txt         dependencies for the lightweight build
requirements-full.txt    extra dependencies for the full build
data/                    dataset + generated files (gitignored)
```

## Setup

Requires Python 3.12.

```bash
pip install -r requirements.txt
```

Place the dataset in the `data/` folder:

```
data/marketing_sample_for_amazon_com-amazon_fashion_products__20200201_20200430__30k_data.ldjson
```

## Running it

All commands are run from the repository root.

1. Clean the raw data into a parquet file:

```bash
python -m product_similarity.clean --raw "data/marketing_sample_for_amazon_com-amazon_fashion_products__20200201_20200430__30k_data.ldjson"
```

2. Build the search index:

```bash
python -m product_similarity.ann
```

This also runs a small demo query and a benchmark comparing the fast (ANN) results against
exact search.

By default the text backend is TF-IDF. The image backend is controlled by the
`IMAGE_BACKEND` environment variable. For a quick, lightweight run without downloading
images or installing torch, set it to `none`:

```bash
# Windows
set IMAGE_BACKEND=none
# macOS / Linux
export IMAGE_BACKEND=none
```

## API

Start the service:

```bash
uvicorn api.app:app --host 0.0.0.0 --port 8000
```

Then query it:

```
GET /find_similar_products?product_id=<id>&num_similar=5
```

It returns a list of similar product IDs. A health check is available at `/health`.

## Docker and Kubernetes

Lightweight image (TF-IDF, no torch):

```bash
docker build -t product-similarity .
docker run -p 8000:8000 product-similarity
```

The full multimodal image (transformer text + CLIP images) is built from `Dockerfile.full`.
It is much larger and expects the image embeddings to be generated locally first, so it
does not download thousands of images during the build.

Kubernetes manifests are in `k8s/`. Deploying to a live cluster was optional in the brief;
the manifests are provided and can be applied with:

```bash
kubectl apply -f k8s/
```

## Configuration

Everything is controlled from `config.py`, and can be overridden with environment variables:

- `TEXT_BACKEND` — `tfidf` (default) or `embedding`
- `IMAGE_BACKEND` — `none`, `resnet`, or `clip` (default)
- `NUMERIC_WEIGHT`, `IMAGE_WEIGHT` — how much each part counts in the combined score

Weights can also be passed per request to `find_similar_products`, so the balance can be
tuned without rebuilding anything.

## Design decisions and trade-offs

Late fusion over early fusion. The scores from text, numeric, and image are combined at
query time rather than baking the modalities into one vector up front. This makes the
weights tunable per request and makes the missing-image fallback simple — a missing modality
is just dropped from the weighted average. Early fusion is still used to build the single
vector the HNSW index needs.

HNSW for the nearest-neighbour bonus. I used HNSW (`hnswlib`), following Malkov & Yashunin,
2016. It gives a strong recall/latency trade-off at this scale, supports incremental inserts,
and is a small, torch-free dependency. FAISS is aimed at much larger / GPU workloads, and
Annoy needs a full rebuild to add items. The index is used in a retrieve-then-rerank pattern:
HNSW returns candidates quickly, then they are re-scored with the full weighted score.

Text and category are the strongest signal. The dataset's text fields (name, keywords, brand,
category) are almost always present, so the system is text-and-category-first, with numeric
and image as supporting signals.

Weight is retrieved but not used in the similarity. About 79% of the `weight` values are a
`999999999` placeholder, and weight is a weak similarity signal for clothing compared to
category and description. It is still retrieved with the product's attributes, but left out of
the score. The brief allows choosing which attributes to use, so this is a deliberate,
data-driven choice.

Tie-breaking. When two products have the same similarity score, the tie is broken by rating,
then by sales price.

Caching. Images are cached on disk by URL, and the computed image embeddings are cached too,
so rebuilds don't re-download or re-embed unless the data or backend changes.

## Notes on the data

A few things in the data shaped the design:

- There is no `price` column; `sales_price` is used as the price.
- The category field is a dictionary; the first key is the broad category and the last key is
  the most specific one, so both are parsed out.
- The image field can contain several URLs separated by `|`; only the first is used.
- Missing values are kept as missing during cleaning rather than filled in, so each feature
  can decide how to handle its own gaps.

## What maps to each part of the brief

- Part 1 (similarity function): `find_similar_products` in `search.py` (exact) and `ann.py`
  (indexed). It retrieves the product's attributes, scores it against the others with cosine
  similarity, sorts, and returns the top `N`.
- Part 2 (microservice): `api/app.py` (FastAPI, `GET /find_similar_products`) with the
  `Dockerfile` and `k8s/` manifests.
- Part 3 (bonus, vector search): HNSW index in `ann.py`.
- Optional multimodal: text + image features combined by score with configurable weights and a
  fallback, in `features.py`, `image_features.py`, `search.py`, and `ann.py`.
