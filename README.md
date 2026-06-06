# Product Similarity Search

A service that finds similar products in the Amazon India fashion dataset (~30,000 items).
Given a product ID, it returns the N most similar product IDs.

It combines three kinds of signal; text, numeric, and image into a single similarity
score with configurable weights, and falls back gracefully when a signal is missing. It
also includes an approximate nearest-neighbour index (HNSW) so search stays fast on the
full dataset.

## How it works

Each product is turned into vectors:

- text: product name, meta keywords, brand, category, and colour, combined into one
  string and turned into a vector with TF-IDF + SVD (or, optionally, transformer embeddings).
- numeric: log(sales_price) and rating, standardized.
- image: features from the product image using CLIP or ResNet (optional).

## Setup

Requires Python 3.12.

```bash
pip install -r requirements.txt
```

Place the dataset in the data/ folder:

```
data/marketing_sample_for_amazon_com-amazon_fashion_products__20200201_20200430__30k_data.ldjson
```

## Running it

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
IMAGE_BACKEND environment variable. For a quick, lightweight run without downloading
images or installing torch, set it to none:

```bash
export IMAGE_BACKEND=none
```

## API

Start the service:

```bash
uvicorn api.app:app --host 0.0.0.0 --port 8000
GET /find_similar_products?product_id=<id>&num_similar=5
```

It returns a list of similar product IDs.

## Docker

Lightweight image (TF-IDF, no torch):

```bash
docker build -t product-similarity .
docker run -p 8000:8000 product-similarity
```

## Configuration

Everything is controlled from `config.py`, and can be overridden with environment variables:

- TEXT_BACKEND, tfidf (default) or embedding
- IMAGE_BACKEND, none (default), resnet, or clip
- NUMERIC_WEIGHT, IMAGE_WEIGHT how much each part counts in the combined score

