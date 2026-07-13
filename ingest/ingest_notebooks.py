"""Build the practitioner_knowledge collection from Code4ML code blocks.

Code4ML blocks are already notebook-cell-level chunks tagged with the
competition slug (data_sources) and the notebook's Kaggle score. Dev subset:
only blocks from MLE-bench Lite competitions are ingested (the leave-one-out
filter at query time excludes the current competition).

Usage: python -m ingest.ingest_notebooks   (run ingest.download first)
"""

import hashlib

import pandas as pd

from ingest.chunking import split_oversized
from ingest.config import LITE_COMPETITIONS, PRACTITIONER_KNOWLEDGE, RAW_DIR
from ingest.store import add_documents, get_collection

MIN_BLOCK_CHARS = 80  # drop trivial boilerplate cells
CSV_CHUNK_ROWS = 50_000


def _ingest_frame(*, frame: pd.DataFrame, collection, seen_hashes: set[str]) -> int:
    frame = frame[frame["data_sources"].isin(LITE_COMPETITIONS)]
    ids, texts, metadatas = [], [], []
    for _, row in frame.iterrows():
        block = str(row["code_block"])
        if len(block) < MIN_BLOCK_CHARS:
            continue
        digest = hashlib.sha256(block.encode()).hexdigest()[:16]
        if digest in seen_hashes:  # identical boilerplate repeats across notebooks
            continue
        seen_hashes.add(digest)
        metadata = {
            "competition_id": str(row["data_sources"]),
            "source_type": "code_block",
            "kaggle_id": int(row["kaggle_id"]),
        }
        if pd.notna(row["kaggle_score"]):
            metadata["kaggle_score"] = float(row["kaggle_score"])
        for part, chunk in enumerate(split_oversized(text=block)):
            ids.append(f"block_{digest}_{part}")
            texts.append(chunk)
            metadatas.append(metadata)
    if ids:
        add_documents(collection=collection, ids=ids, texts=texts, metadatas=metadatas)
    return len(ids)


def main() -> None:
    collection = get_collection(name=PRACTITIONER_KNOWLEDGE)
    seen_hashes: set[str] = set()
    total = 0
    for filename in ["code_blocks_upto_20.csv", "code_blocks_21.csv"]:
        path = RAW_DIR / "code4ml" / filename
        for frame in pd.read_csv(path, index_col=0, chunksize=CSV_CHUNK_ROWS):
            ingested = _ingest_frame(frame=frame, collection=collection, seen_hashes=seen_hashes)
            total += ingested
            print(f"{filename}: +{ingested} chunks (running total {total})")
    print(f"collection count: {collection.count()}")


if __name__ == "__main__":
    main()
