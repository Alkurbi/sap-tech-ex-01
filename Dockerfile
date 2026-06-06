FROM python:3.12-slim

WORKDIR /app
ENV PYTHONPATH=/app

# Torch-free image: TF-IDF + HNSW. Builds cleaned data + index at build time, you could change to text embedding and clip accordingly. Uncomment required libraries in requeriments.txt.
ENV TEXT_BACKEND=tfidf
ENV IMAGE_BACKEND=none

# needed for hnswlib wheel
RUN apt-get update && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY product_similarity/ ./product_similarity/
COPY api/ ./api/

COPY data/marketing_sample_for_amazon_com-amazon_fashion_products__20200201_20200430__30k_data.ldjson ./data/

RUN python -m product_similarity.clean --raw "data/marketing_sample_for_amazon_com-amazon_fashion_products__20200201_20200430__30k_data.ldjson" \
 && python -c "from product_similarity import ann; ann.init(rebuild=True)"

EXPOSE 8000
CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]