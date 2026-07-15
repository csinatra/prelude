"""Build the notebook_summaries collection — level one of two-level retrieval.

One record per unique kaggle_id in the Lite-22 code-block slice: gather the
notebook's blocks, generate a short LLM abstract of its approach, embed the
abstract with Voyage, store with {kaggle_id, competition_id, kaggle_score,
summary_model}.

Resumable: kaggle_ids already in the collection are skipped, so the run can
be interrupted and restarted.

Usage: python -m ingest.ingest_summaries [--limit N]   (run ingest.download first)
--limit caps how many pending notebooks this invocation summarizes — for
staged spends; resumability makes successive capped runs additive.
"""

import argparse

from concurrent.futures import ThreadPoolExecutor

import pandas as pd

from ingest.config import LITE_COMPETITIONS, NOTEBOOK_SUMMARIES, RAW_DIR
from ingest.store import add_documents, get_collection
from pipeline.llm_client import call_llm_text

# Pinned deliberately, NOT the MODEL env var: summaries are corpus
# infrastructure shared identically by every condition and every run — they
# are exempt from the Haiku-dev/Sonnet-eval switch, and the resumable ingest
# must never produce a mixed-model collection. Each record carries
# summary_model metadata so an improper insert is auditable and deletable by
# filter.
SUMMARY_MODEL = "claude-haiku-4-5-20251001"

MAX_NOTEBOOK_CHARS = 12_000  # abstract input cap; blocks concatenated in order
LLM_WORKERS = 8
UPSERT_BATCH = 200

SUMMARY_SYSTEM = (
    "You summarize Kaggle notebooks for a retrieval index. Given the code cells of one "
    "notebook, write a 3-5 sentence abstract of its approach: models used, feature "
    "engineering, validation strategy, and anything distinctive. Plain prose, no headers."
)


def _load_notebooks() -> dict[int, dict]:
    """Group Lite-22 code blocks by kaggle_id, preserving block order."""
    notebooks: dict[int, dict] = {}
    for filename in ["code_blocks_upto_20.csv", "code_blocks_21.csv"]:
        path = RAW_DIR / "code4ml" / filename
        for frame in pd.read_csv(path, index_col=0, chunksize=100_000):
            frame = frame[frame["data_sources"].isin(LITE_COMPETITIONS)]
            for _, row in frame.iterrows():
                kaggle_id = int(row["kaggle_id"])
                entry = notebooks.setdefault(
                    kaggle_id,
                    {
                        "competition_id": str(row["data_sources"]),
                        "kaggle_score": float(row["kaggle_score"])
                        if pd.notna(row["kaggle_score"])
                        else None,
                        "blocks": [],
                    },
                )
                entry["blocks"].append(str(row["code_block"]))
    return notebooks


def _summarize(*, kaggle_id: int, entry: dict) -> tuple[int, str]:
    text = "\n\n".join(entry["blocks"])[:MAX_NOTEBOOK_CHARS]
    abstract = call_llm_text(
        system=SUMMARY_SYSTEM,
        user=f"Notebook code cells:\n{text}",
        max_tokens=512,
        model=SUMMARY_MODEL,
    )
    return kaggle_id, abstract


def main(*, limit: int | None = None) -> None:
    collection = get_collection(name=NOTEBOOK_SUMMARIES)
    notebooks = _load_notebooks()
    print(f"unique notebooks in slice: {len(notebooks)}")

    existing = set(collection.get(ids=[f"nb_{kid}" for kid in notebooks])["ids"])
    pending = {kid: entry for kid, entry in notebooks.items() if f"nb_{kid}" not in existing}
    print(f"already summarized: {len(existing)}; pending: {len(pending)}")
    if limit is not None:
        pending = dict(list(pending.items())[:limit])
        print(f"capped this run to {len(pending)} notebooks (--limit {limit})")

    batch_ids: list[str] = []
    batch_texts: list[str] = []
    batch_metadatas: list[dict] = []

    def flush(done: int, total: int) -> None:
        if batch_ids:
            add_documents(
                collection=collection, ids=batch_ids, texts=batch_texts, metadatas=batch_metadatas
            )
            batch_ids.clear()
            batch_texts.clear()
            batch_metadatas.clear()
            print(f"progress: {done}/{total}")

    with ThreadPoolExecutor(max_workers=LLM_WORKERS) as pool:
        futures = [
            pool.submit(_summarize, kaggle_id=kid, entry=entry) for kid, entry in pending.items()
        ]
        for done, future in enumerate(futures, start=1):
            kaggle_id, abstract = future.result()
            entry = notebooks[kaggle_id]
            metadata = {
                "kaggle_id": kaggle_id,
                "competition_id": entry["competition_id"],
                "summary_model": SUMMARY_MODEL,
            }
            if entry["kaggle_score"] is not None:
                metadata["kaggle_score"] = entry["kaggle_score"]
            batch_ids.append(f"nb_{kaggle_id}")
            batch_texts.append(abstract)
            batch_metadatas.append(metadata)
            if len(batch_ids) >= UPSERT_BATCH:
                flush(done, len(futures))
    flush(len(pending), len(pending))
    print(f"collection count: {collection.count()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    main(limit=parser.parse_args().limit)
