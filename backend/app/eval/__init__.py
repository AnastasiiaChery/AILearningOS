"""Retrieval evaluation harness.

Measures the retriever with numbers (Recall@k / MRR / nDCG@k) over a hand-curated
golden set, so changes to chunking, fusion, reranking, HyDE, late-interaction, etc.
are compared by metric rather than by eyeballing. See README.md.
"""
