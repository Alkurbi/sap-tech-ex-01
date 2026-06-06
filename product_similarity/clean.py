import argparse
import os
import re

import numpy as np
import pandas as pd

from . import config

DROP_COLUMNS = [
    "name_of_author_for_books", "formats___editions", "no__of_offers",
    "no__of_sellers", "technical_details__k_v_pairs", "left_in_stock",
    "no__of_reviews", "seller_name", "seller_id", "other_items_customers_buy",
    "delivery_type", "crawl_timestamp", "product_details__k_v_pairs",
]


def parse_weight(raw):
    if pd.isna(raw):
        return np.nan
    m = re.match(r"([\d.]+)\s*g\b", str(raw).strip())
    if not m:
        return np.nan
    val = float(m.group(1))
    if 0 < val < 10000:
        return val
    return np.nan

def humanize(token):
    # "WomensKurtasKurtis" -> "Womens Kurtas Kurtis"
    token = token.replace("_", " ")
    spaced = re.sub(r"(?<!^)(?=[A-Z])", " ", token)
    return spaced.replace("  ", " ").strip()


def parse_category(d):
    if not isinstance(d, dict) or len(d) == 0:
        return (None, None)
    keys = list(d.keys())
    return (humanize(keys[0]), humanize(keys[-1]))


def parse_colours(raw):
    # "Black|Blue|Grey" -> "black|blue|grey", remove duplicates
    if pd.isna(raw):
        return None
    seen = set()
    out = []
    for part in str(raw).split("|"):
        base = part.split("(")[0].strip().lower()
        if base and base not in seen:
            seen.add(base)
            out.append(base)
    if out:
        return "|".join(out)
    return None


def clean_text(raw):
    # collapse extra whitespace, empty -> None
    if pd.isna(raw):
        return None
    s = re.sub(r"\s+", " ", str(raw)).strip()
    if s:
        return s
    return None


def first_image_url(raw):
    # the image field can hold several urls separated by "|", just keep the first one
    if pd.isna(raw):
        return None
    first = str(raw).split("|")[0].strip()
    if first:
        return first
    return None


def clean(df):
    df = df.drop_duplicates(subset="uniq_id").copy()
    cat = df["parent___child_category__all"].apply(parse_category)

    out = pd.DataFrame({
        "uniq_id": df["uniq_id"].astype(str),
        "product_name": df["product_name"].apply(clean_text),
        "meta_keywords": df["meta_keywords"].apply(clean_text),
        "brand": df["brand"].apply(clean_text),
        "category_parent": cat.str[0],
        "category_leaf": cat.str[1],
        "colour": df["colour"].apply(parse_colours),
        "sales_price": pd.to_numeric(df["sales_price"], errors="coerce"),
        "rating": pd.to_numeric(df["rating"], errors="coerce"),
        "weight_g": df["weight"].apply(parse_weight),
        "image_url": df["medium"].apply(first_image_url),
        "product_url": df["product_url"].apply(clean_text),
    })
    return out.reset_index(drop=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True)
    ap.add_argument("--out", default=config.PARQUET)
    args = ap.parse_args()

    raw = pd.read_json(args.raw, lines=True)
    out = clean(raw)

    folder = os.path.dirname(args.out)
    if folder:
        os.makedirs(folder, exist_ok=True)
    out.to_parquet(args.out, index=False)

    print("rows:", len(out), " cols:", out.shape[1], " ->", args.out)
    print("coverage (fraction not missing):")
    print(out.notna().mean().sort_values(ascending=False).round(3).to_string())


if __name__ == "__main__":
    main()