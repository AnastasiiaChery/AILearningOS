# FastAPI Dependency Injection

FastAPI uses a powerful dependency injection system based on the `Depends` callable.

## How Depends Works

When you declare a parameter with `Depends(get_db)`, FastAPI resolves the dependency
before calling your endpoint. Dependencies can be cached within a request scope.

### Yield Dependencies

A dependency that uses `yield` runs cleanup code after the response is sent. This is the
idiomatic way to manage database sessions: acquire before, release after.

## Background Tasks

`BackgroundTasks` lets you schedule work to run after returning the response. It is ideal
for document ingestion pipelines where the upload returns 202 immediately and processing
continues asynchronously.

# Qdrant Vector Search

Qdrant supports both dense and sparse vectors in a single collection.

## Hybrid Search

Hybrid search combines dense semantic vectors with sparse keyword vectors, then fuses the
results using Reciprocal Rank Fusion (RRF). The magic constant in RRF is typically 60.

### Sparse Vectors and BM25

Sparse vectors represent documents as bags of weighted token indices. BM25 weighting uses
term frequency and inverse document frequency across the whole corpus, not a single document.
