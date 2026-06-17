"""Regression tests for the TF-only sparse vector builder (Module 2).

Locks in the two fixes:
  1. token ids are deterministic (stable across processes) — no Python hash salt;
  2. a sparse vector never contains duplicate indices (collisions are aggregated).
IDF itself is delegated to Qdrant (Modifier.IDF) and is not tested here.
"""
from app.services.rag.qdrant_store import (
    _build_sparse_vector,
    _token_id,
    _tf_weight,
)


def test_token_id_is_deterministic():
    # Same string → same id, every call (the old hash() salt broke this).
    assert _token_id("qdrant") == _token_id("qdrant")
    # Known fixed value pins determinism across processes/runs.
    assert _token_id("qdrant") == 277249


def test_sparse_indices_are_unique():
    sv = _build_sparse_vector("rrf rrf fusion fusion fusion qdrant the the the")
    assert len(sv.indices) == len(set(sv.indices))


def test_tf_saturates():
    # BM25 saturation: marginal gain shrinks as count grows.
    gain_1_to_2 = _tf_weight(2) - _tf_weight(1)
    gain_9_to_10 = _tf_weight(10) - _tf_weight(9)
    assert gain_9_to_10 < gain_1_to_2


def test_no_idf_in_values():
    # Two distinct words with the same count must get the SAME stored weight,
    # because IDF (which would differ) is applied by Qdrant, not here.
    a = _build_sparse_vector("qdrant")
    b = _build_sparse_vector("the")
    assert a.values[0] == b.values[0]
