import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# text backend: "tfidf" or "embedding"
TEXT_BACKEND = os.environ.get("TEXT_BACKEND", "tfidf")
TEXT_MODEL = os.environ.get("TEXT_MODEL", "all-MiniLM-L6-v2")

# image backend: "none", "resnet" or "clip"
IMAGE_BACKEND = os.environ.get("IMAGE_BACKEND", "none")
CLIP_MODEL = os.environ.get("CLIP_MODEL", "clip-ViT-B-32")


NUMERIC_WEIGHT = float(os.environ.get("NUMERIC_WEIGHT", "0.3"))
IMAGE_WEIGHT = float(os.environ.get("IMAGE_WEIGHT", "0.5"))
WEIGHTS = {"text": 1.0, "numeric": NUMERIC_WEIGHT, "image": IMAGE_WEIGHT}

DATA_DIR = os.environ.get("DATA_DIR", os.path.join(ROOT, "data"))
PARQUET = os.path.join(DATA_DIR, "products_clean.parquet")
EMBED_DIR = DATA_DIR
IMAGE_CACHE = os.path.join(DATA_DIR, "img_cache")


def summary():
    return f"text={TEXT_BACKEND} image={IMAGE_BACKEND} numeric_w={NUMERIC_WEIGHT} image_w={IMAGE_WEIGHT}"