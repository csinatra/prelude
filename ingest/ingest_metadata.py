"""Build the competition_metadata collection.

Sources: Code4ML competitions.csv (structured descriptions, eval metrics) and
mle-bench description.md files (authoritative specs for the eval competitions).

Usage: python -m ingest.ingest_metadata   (run ingest.download first)
"""

import pandas as pd

from ingest.chunking import split_oversized
from ingest.config import COMPETITION_METADATA, LITE_COMPETITIONS, RAW_DIR
from ingest.store import add_documents, get_collection


def _code4ml_docs() -> tuple[list[str], list[str], list[dict]]:
    frame = pd.read_csv(RAW_DIR / "code4ml" / "competitions.csv", index_col=0)
    ids, texts, metadatas = [], [], []
    for _, row in frame.iterrows():
        slug = str(row["data_sources"])
        description = row["description"]
        if not slug or pd.isna(description):
            continue
        metadata = {
            "competition_id": slug,
            "source_type": "competition_description",
            "task_category": str(row["datatype"]) if pd.notna(row["datatype"]) else "unknown",
            "evaluation_metric": (
                str(row["EvaluationAlgorithmAbbreviation"])
                if pd.notna(row["EvaluationAlgorithmAbbreviation"])
                else "unknown"
            ),
        }
        for part, chunk in enumerate(split_oversized(text=str(description))):
            ids.append(f"code4ml_{slug}_{part}")
            texts.append(chunk)
            metadatas.append(metadata)
    return ids, texts, metadatas


def _mlebench_docs() -> tuple[list[str], list[str], list[dict]]:
    ids, texts, metadatas = [], [], []
    for slug in LITE_COMPETITIONS:
        path = RAW_DIR / "mlebench_descriptions" / f"{slug}.md"
        if not path.exists():
            print(f"warning: missing description for {slug}")
            continue
        metadata = {"competition_id": slug, "source_type": "competition_description"}
        for part, chunk in enumerate(split_oversized(text=path.read_text())):
            ids.append(f"mlebench_{slug}_{part}")
            texts.append(chunk)
            metadatas.append(metadata)
    return ids, texts, metadatas


def main() -> None:
    collection = get_collection(name=COMPETITION_METADATA)
    for label, (ids, texts, metadatas) in {
        "code4ml": _code4ml_docs(),
        "mlebench": _mlebench_docs(),
    }.items():
        add_documents(collection=collection, ids=ids, texts=texts, metadatas=metadatas)
        print(f"{label}: {len(ids)} chunks ingested")
    print(f"collection count: {collection.count()}")


if __name__ == "__main__":
    main()
