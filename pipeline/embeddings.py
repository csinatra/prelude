"""Voyage AI embeddings — the single embedding seam for ingestion and retrieval.

voyage-4-large for both documents and queries so the embedding space is
consistent. Switched from voyage-code-3 (2026-08-03): once the retrieval unit
became the notebook summary, retrieval is entirely natural-language↔NL (summary
prose, competition descriptions, NL queries) with no code embedded, so a
general-purpose model fits better than a code-specialized one — and voyage-4-large
is cheaper ($0.12 vs $0.18/M) with a 200M free grant. 1024 default dims (matches
the prior store). Requires VOYAGE_API_KEY in the env. See docs/DECISIONS.md.
"""

import os
from typing import Literal

import voyageai

EMBED_MODEL = "voyage-4-large"
# Voyage API limits per request: 128 texts AND 120K total tokens. Token count
# is estimated at 2 chars/token: unicode-heavy corpus content (e.g. non-English
# text in notebook cells) measured as dense as ~2.6 chars/token, so the common
# ~4 chars/token heuristic under-counts and trips the API cap. At 2 chars/token
# even 1-char-per-token content stays ≤ ~110K real tokens per batch.
MAX_BATCH_TEXTS = 128
MAX_BATCH_TOKENS = 55_000

_client: voyageai.Client | None = None


def _get_client() -> voyageai.Client:
    global _client
    if _client is None:
        _client = voyageai.Client(api_key=os.environ["VOYAGE_API_KEY"])
    return _client


def _batches(*, texts: list[str]) -> list[list[str]]:
    batches: list[list[str]] = []
    batch: list[str] = []
    batch_tokens = 0
    for text in texts:
        estimated = max(1, len(text) // 2)
        if batch and (len(batch) >= MAX_BATCH_TEXTS or batch_tokens + estimated > MAX_BATCH_TOKENS):
            batches.append(batch)
            batch, batch_tokens = [], 0
        batch.append(text)
        batch_tokens += estimated
    if batch:
        batches.append(batch)
    return batches


def embed(*, texts: list[str], input_type: Literal["document", "query"]) -> list[list[float]]:
    """Embed texts with voyage-4-large. Batches transparently under both API caps."""
    vectors: list[list[float]] = []
    for batch in _batches(texts=texts):
        result = _get_client().embed(texts=batch, model=EMBED_MODEL, input_type=input_type)
        vectors.extend(result.embeddings)
    return vectors
