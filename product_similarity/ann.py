import os

import hnswlib
import numpy as np
import pandas as pd

from . import config
from . import features


index = fused = mats = present = ids = pos = rating = price = None


def build_index(matrix, M=16, ef_construction=200, ef=128):
    idx = hnswlib.Index(space="cosine", dim=matrix.shape[1])
    idx.init_index(max_elements=matrix.shape[0], ef_construction=ef_construction, M=M)
    idx.add_items(matrix, np.arange(matrix.shape[0]))
    idx.set_ef(ef)
    return idx


def init(parquet=None, artifacts_dir=None, rebuild=False, ef=128, **build_kwargs):
    global index, fused, mats, present, ids, pos, rating, price
    if parquet is None:
        parquet = config.PARQUET
    if artifacts_dir is None:
        artifacts_dir = config.DATA_DIR

    df = pd.read_parquet(parquet)
    if rebuild:
        kw = {"text_backend": config.TEXT_BACKEND, "model_name": config.TEXT_MODEL,
              "image_backend": config.IMAGE_BACKEND}
        kw.update(build_kwargs)
        mats, present, ids, art = features.build_modality_matrices(df, **kw)
        features.save(mats, present, ids, art, artifacts_dir)
    else:
        mats, present, ids, art = features.load(artifacts_dir)

    fused = features.fuse(mats, config.WEIGHTS) 

    index_path = os.path.join(artifacts_dir, "hnsw.bin")
    if rebuild or not os.path.exists(index_path):
        index = build_index(fused, ef=ef)
        index.save_index(index_path)
    else:
        index = hnswlib.Index(space="cosine", dim=fused.shape[1])
        index.load_index(index_path, max_elements=fused.shape[0])
        index.set_ef(ef)

    pos = {}
    for i in range(len(ids)):
        pos[ids[i]] = i

    meta = df.set_index("uniq_id").reindex(ids)
    rating = meta["rating"].to_numpy(dtype="float64")
    price = meta["sales_price"].to_numpy(dtype="float64")
    if rebuild:
        print("built HNSW index", fused.shape, config.summary())


def rerank(i, candidates, weights):
    candidates = np.asarray(candidates)
    num = np.zeros(len(candidates))
    den = np.zeros(len(candidates))

    num = num + weights["text"] * (mats["text"][candidates] @ mats["text"][i])
    den = den + weights["text"]

    num = num + weights["numeric"] * (mats["numeric"][candidates] @ mats["numeric"][i])
    den = den + weights["numeric"]

    if mats["image"] is not None and present[i] and weights.get("image", 0) > 0:
        w = weights["image"] * present[candidates]
        num = num + w * (mats["image"][candidates] @ mats["image"][i])
        den = den + w

    return num / np.maximum(den, 1e-9)


def find_similar_products(product_id, num_similar, weights=None, candidate_factor=20):
    if index is None:
        init()
    if product_id not in pos:
        raise KeyError(product_id)
    if weights is None:
        weights = config.WEIGHTS

    i = pos[product_id]

    k = max(num_similar * candidate_factor, 200)
    if k > len(ids):
        k = len(ids)
    labels, distances = index.knn_query(fused[i], k=k)
    candidates = labels[0]

    scores = rerank(i, candidates, weights)
    r = np.nan_to_num(rating[candidates], nan=-1.0)
    p = np.nan_to_num(price[candidates], nan=-1.0)
    order = np.lexsort((-p, -r, -scores))

    result = []
    for j in order:
        if candidates[j] != i:
            result.append(ids[candidates[j]])
        if len(result) >= num_similar:
            break
    return result


def benchmark(n_queries=300, k=10):
    import time
    rng = np.random.RandomState(0)
    queries = rng.choice(len(ids), n_queries, replace=False)

    def exact(i):
        s = fused @ fused[i]
        return set(np.argsort(-s)[1:k + 1])

    hits = 0
    total = 0
    for i in queries:
        ann_result = set(index.knn_query(fused[i], k=k + 1)[0][0])
        ann_result.discard(i)
        hits += len(ann_result & exact(i))
        total += k
    recall = hits / total

    start = time.time()
    for i in queries:
        index.knn_query(fused[i], k=k + 1)
    ann_ms = (time.time() - start) / n_queries * 1000

    start = time.time()
    for i in queries:
        np.argsort(-(fused @ fused[i]))[:k]
    exact_ms = (time.time() - start) / n_queries * 1000

    print("recall@" + str(k) + ":", round(recall, 3),
          "| ANN", round(ann_ms, 2), "ms vs exact", round(exact_ms, 2), "ms")


if __name__ == "__main__":
    init(rebuild=True)
    df = pd.read_parquet(config.PARQUET).set_index("uniq_id")
    q = ids[519]
    print()
    print("QUERY:", df.loc[q, "product_name"][:58], "|", df.loc[q, "category_leaf"])
    print("-" * 80)
    for pid in find_similar_products(q, 5):
        print(df.loc[pid, "category_leaf"], "-", df.loc[pid, "product_name"][:55])
    print()
    benchmark()