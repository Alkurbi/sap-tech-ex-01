import os
import pickle

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler, normalize

from . import image_features

TEXT_FIELDS = ["product_name", "meta_keywords", "brand",
               "category_parent", "category_leaf", "colour"]
NUMERIC_FIELDS = ["sales_price", "rating"]


def build_text_blob(df):
    def make_row(r):
        parts = []
        for field in TEXT_FIELDS:
            if pd.notna(r[field]):
                parts.append(str(r[field]).replace("|", " "))
        return " ".join(parts).lower()
    return df.apply(make_row, axis=1)


def text_vectors_tfidf(blob, n_components, random_state):
    tfidf = TfidfVectorizer(max_features=20000, ngram_range=(1, 2),
                            min_df=2, stop_words="english")
    X = tfidf.fit_transform(blob)
    n_comp = min(n_components, X.shape[1] - 1)
    svd = TruncatedSVD(n_components=n_comp, random_state=random_state)
    vecs = svd.fit_transform(X)
    return vecs, {"backend": "tfidf", "tfidf": tfidf, "svd": svd}


def text_vectors_embedding(blob, model_name):
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_name)
    vecs = model.encode(blob.tolist(), normalize_embeddings=True,
                        batch_size=256, show_progress_bar=False)
    return np.asarray(vecs, dtype="float32"), {"backend": "embedding", "model_name": model_name}


def numeric_vectors(df):
    num = df[NUMERIC_FIELDS].copy()
    num["sales_price"] = np.log1p(num["sales_price"])
    num = num.fillna(num.median())
    scaler = StandardScaler()
    return scaler.fit_transform(num).astype("float32"), scaler


def build_modality_matrices(df, text_backend="tfidf", model_name="all-MiniLM-L6-v2",
                            image_backend="none", n_components=200, random_state=0):
    blob = build_text_blob(df)
    if text_backend == "embedding":
        text_vecs, text_art = text_vectors_embedding(blob, model_name)
    else:
        text_vecs, text_art = text_vectors_tfidf(blob, n_components, random_state)

    num_vecs, scaler = numeric_vectors(df)

    mats = {
        "text": normalize(text_vecs).astype("float32"),
        "numeric": normalize(num_vecs).astype("float32"),
        "image": None,
    }
    image_present = np.ones(len(df), dtype=bool)

    art = {"scaler": scaler}
    art.update(text_art)

    if image_backend != "none":
        img = image_features.build_image_vectors(df, backend=image_backend)
        image_present = ~np.isnan(img).any(axis=1)   # False for rows with no image
        img = normalize(np.nan_to_num(img, nan=0.0))
        mats["image"] = img.astype("float32")
        art["image_backend"] = image_backend
        art["image_missing"] = int((~image_present).sum())

    return mats, image_present, df["uniq_id"].tolist(), art


def fuse(mats, weights):
    blocks = [weights["text"] * mats["text"], weights["numeric"] * mats["numeric"]]
    if mats.get("image") is not None:
        blocks.append(weights["image"] * mats["image"])
    return normalize(np.hstack(blocks).astype("float32"))


def save(mats, image_present, ids, artifacts, out_dir="artifacts"):
    os.makedirs(out_dir, exist_ok=True)
    if mats["image"] is not None:
        image = mats["image"]
    else:
        image = np.empty((0, 0))
    np.savez(os.path.join(out_dir, "mats.npz"),
             text=mats["text"], numeric=mats["numeric"],
             image=image, image_present=image_present)
    meta = {"ids": ids, "artifacts": artifacts, "has_image": mats["image"] is not None}
    with open(os.path.join(out_dir, "index.pkl"), "wb") as f:
        pickle.dump(meta, f)


def load(out_dir="artifacts"):
    z = np.load(os.path.join(out_dir, "mats.npz"))
    with open(os.path.join(out_dir, "index.pkl"), "rb") as f:
        meta = pickle.load(f)
    if meta["has_image"]:
        image = z["image"]
    else:
        image = None
    mats = {"text": z["text"], "numeric": z["numeric"], "image": image}
    return mats, z["image_present"], meta["ids"], meta["artifacts"]