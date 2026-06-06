import numpy as np
import pandas as pd

from . import config
from . import features

mats = None
present = None
ids = None
pos = None
rating = None
price = None


def init(parquet=None, artifacts_dir=None, rebuild=False, **build_kwargs):
    global mats, present, ids, pos, rating, price
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
        print("built", config.summary(), " images:", present.sum(), "/", len(present))
    else:
        mats, present, ids, art = features.load(artifacts_dir)

    pos = {}
    for i in range(len(ids)):
        pos[ids[i]] = i

    meta = df.set_index("uniq_id").reindex(ids)
    rating = meta["rating"].to_numpy(dtype="float64")
    price = meta["sales_price"].to_numpy(dtype="float64")


def combined_scores(i, weights):
    n = mats["text"].shape[0]
    num = np.zeros(n)
    den = np.zeros(n)

    text_score = mats["text"] @ mats["text"][i]
    num = num + weights["text"] * text_score
    den = den + weights["text"]

    numeric_score = mats["numeric"] @ mats["numeric"][i]
    num = num + weights["numeric"] * numeric_score
    den = den + weights["numeric"]

    if mats["image"] is not None and present[i] and weights.get("image", 0) > 0:
        image_score = mats["image"] @ mats["image"][i]
        w = weights["image"] * present
        num = num + w * image_score
        den = den + w

    return num / np.maximum(den, 1e-9)


def find_similar_products(product_id, num_similar, weights=None):
    if mats is None:
        init()
    if product_id not in pos:
        raise KeyError(product_id)
    if weights is None:
        weights = config.WEIGHTS

    i = pos[product_id]
    scores = combined_scores(i, weights)

    r = np.nan_to_num(rating, nan=-1.0)
    p = np.nan_to_num(price, nan=-1.0)
    order = np.lexsort((-p, -r, -scores))

    result = []
    for j in order:
        if j != i:
            result.append(ids[j])
        if len(result) >= num_similar:
            break
    return result


if __name__ == "__main__":
    init(rebuild=True)
    df = pd.read_parquet(config.PARQUET).set_index("uniq_id")
    q = ids[519]
    print()
    print("QUERY:", df.loc[q, "product_name"][:60], "|", df.loc[q, "category_leaf"], df.loc[q, "sales_price"])
    print("-" * 80)
    for pid in find_similar_products(q, 5):
        print(df.loc[pid, "category_leaf"], "-", df.loc[pid, "product_name"][:56], df.loc[pid, "sales_price"])