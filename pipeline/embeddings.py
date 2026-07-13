"""Voyage AI embeddings — the single embedding seam for ingestion and retrieval.

voyage-code-3 for both documents and queries so the embedding space is
consistent (see implementation brief). Requires VOYAGE_API_KEY in the env.
"""

import os
from typing import Literal

import voyageai

EMBED_MODEL = "voyage-code-3"
# Voyage API limits per request: 128 texts / 120K total tokens.
BATCH_SIZE = 128

_client: voyageai.Client | None = None


def _get_client() -> voyageai.Client:
    global _client
    if _client is None:
        _client = voyageai.Client(api_key=os.environ["VOYAGE_API_KEY"])
    return _client


def embed(*, texts: list[str], input_type: Literal["document", "query"]) -> list[list[float]]:
    """Embed texts with voyage-code-3. Batches transparently."""
    vectors: list[list[float]] = []
    for start in range(0, len(texts), BATCH_SIZE):
        batch = texts[start : start + BATCH_SIZE]
        result = _get_client().embed(texts=batch, model=EMBED_MODEL, input_type=input_type)
        vectors.extend(result.embeddings)
    return vectors
