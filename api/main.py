import os
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel

from product_similarity import ann, config


def _ready():
    has_index = os.path.exists(os.path.join(config.EMBED_DIR, "hnsw.bin"))
    ann.init(rebuild=not has_index)


@asynccontextmanager
async def lifespan(app: FastAPI):
    _ready()
    yield


app = FastAPI(title="Product Similarity Search")


class SimilarProductsResponse(BaseModel):
    product_id: str
    num_similar: int
    similar_products: list[str]


class HealthResponse(BaseModel):
    status: str
    num_products: int


@app.get("/health")
def health() -> HealthResponse:
    n = len(ann._ids) if ann._ids is not None else 0
    return HealthResponse(status="ok" if n else "loading", num_products=n)


@app.get("/find_similar_products")
def get_similar_products(
    product_id: Annotated[str, Query(description="uniq_id of the query product")],
    num_similar: Annotated[int, Query(ge=1, le=100, description="how many to return")] = 10
) -> SimilarProductsResponse:

    try:
        result = ann.find_similar_products(product_id, num_similar)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"product_id '{product_id}' not found")

    return SimilarProductsResponse(
        product_id=product_id, num_similar=num_similar, similar_products=result
    )