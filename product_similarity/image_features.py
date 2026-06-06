import hashlib
import os
from urllib.request import Request, urlopen

import numpy as np
from . import config


def cache_path(url, cache_dir):
    name = hashlib.md5(url.encode()).hexdigest()
    return os.path.join(cache_dir, name + ".jpg")


def download_image(url, cache_dir):
    from PIL import Image
    if not isinstance(url, str) or not url.startswith("http"):
        return None
    os.makedirs(cache_dir, exist_ok=True)
    path = cache_path(url, cache_dir)
    try:
        if not os.path.exists(path):
            print("downloading", url)
            req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urlopen(req, timeout=10) as r:
                data = r.read()
            with open(path, "wb") as f:
                f.write(data)
        return Image.open(path).convert("RGB")
    except Exception:
        return None


def embed_resnet(images, present_idx, n_rows):
    import torch
    from torchvision import models
    weights = models.ResNet50_Weights.IMAGENET1K_V2
    net = models.resnet50(weights=weights)
    net.fc = torch.nn.Identity() 
    net.eval()
    transform = weights.transforms()
    out = np.full((n_rows, 2048), np.nan, dtype="float32")
    with torch.no_grad():
        for start in range(0, len(images), 64):
            chunk = images[start:start + 64]
            idx = present_idx[start:start + 64]
            x = torch.stack([transform(im) for im in chunk])
            feats = net(x).cpu().numpy()
            out[idx] = feats
    return out


def embed_clip(images, present_idx, n_rows, model_name):
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_name)
    vecs = model.encode(images, batch_size=64, show_progress_bar=False)
    out = np.full((n_rows, vecs.shape[1]), np.nan, dtype="float32")
    out[present_idx] = vecs.astype("float32")
    return out


def make_fingerprint(urls, backend, model_name):
    joined = "|".join([str(u) for u in urls])
    return hashlib.md5((joined + "::" + backend + "::" + model_name).encode()).hexdigest()


def build_image_vectors(df, backend="clip", cache_dir=None, model_name=None,
                        url_col="image_url", emb_dir=None):
    if cache_dir is None:
        cache_dir = config.IMAGE_CACHE
    if emb_dir is None:
        emb_dir = config.DATA_DIR
    if model_name is None:
        model_name = config.CLIP_MODEL

    urls = df[url_col].tolist()
    fingerprint = make_fingerprint(urls, backend, model_name)
    emb_path = os.path.join(emb_dir, "img_emb_" + backend + ".npz")

    if os.path.exists(emb_path):
        try:
            z = np.load(emb_path)
            if z["fingerprint"].item() == fingerprint and z["emb"].shape[0] == len(df):
                print("loaded cached", backend, "embeddings", z["emb"].shape)
                return z["emb"]
        except Exception:
            pass

    images = []
    present_idx = []
    for i in range(len(urls)):
        im = download_image(urls[i], cache_dir)
        if im is not None:
            images.append(im)
            present_idx.append(i)
    present_idx = np.array(present_idx, dtype=int)
    n = len(df)

    if len(images) == 0:
        raise RuntimeError("no images downloaded")

    if backend == "resnet":
        out = embed_resnet(images, present_idx, n)
    elif backend == "clip":
        out = embed_clip(images, present_idx, n, model_name)
    else:
        raise ValueError("unknown image backend: " + backend)

    os.makedirs(emb_dir, exist_ok=True)
    np.savez(emb_path, emb=out, fingerprint=np.array(fingerprint))
    print("computed and cached", backend, "embeddings", out.shape)
    return out