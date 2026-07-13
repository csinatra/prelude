"""Chunk-size control for corpus documents.

Code4ML code blocks are already cell-level chunks; most fit in one embedding.
Oversized blocks are split at blank-line boundaries (logical units in notebook
code) so no chunk exceeds the cap, with a hard character split as fallback.

The cap is in characters (~4 chars/token): 4096 chars ≈ 1024 tokens, the
chunk size agreed for retrieval precision — well under voyage-code-3's limit.
"""

MAX_CHUNK_CHARS = 4096


def split_oversized(*, text: str, max_chars: int = MAX_CHUNK_CHARS) -> list[str]:
    """Return [text] if it fits, else split at blank lines (hard split as last resort)."""
    if len(text) <= max_chars:
        return [text]

    chunks: list[str] = []
    current = ""
    for block in text.split("\n\n"):
        while len(block) > max_chars:  # single block over the cap: hard split
            if current:
                chunks.append(current)
                current = ""
            chunks.append(block[:max_chars])
            block = block[max_chars:]
        candidate = f"{current}\n\n{block}" if current else block
        if len(candidate) > max_chars:
            chunks.append(current)
            current = block
        else:
            current = candidate
    if current:
        chunks.append(current)
    return [c for c in chunks if c.strip()]
