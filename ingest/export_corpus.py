"""Export the practitioner corpus as a flat, citable artifact plus a fingerprint.

Why this exists: the corpus is NOT reproducible from code and sources alone.
Summarization is an LLM step, so re-running `ingest_summaries` against the same
Code4ML CSVs with the same pinned model yields different text. Published results
therefore have no verifiable link back to their inputs unless the derived corpus
is preserved as an artifact — the case where publishing derived data is expected
practice rather than optional.

Writes two files:

  notebook_summaries.jsonl.gz  one record per notebook (id, competition,
                               kaggle_score, summary_model, text). ~19 MB.
  manifest.json                counts, model identifiers, and the SHA-256 of the
                               export. Small and version-controlled, so results
                               can cite a hash while the bytes live wherever
                               they are eventually deposited (Git LFS, Zenodo).

Embeddings are deliberately excluded: they are ~105 MB, derive deterministically
from the text under a named model, and the Chroma index rebuilds from them. The
vector store itself is never an artifact — it is a derived binary index.

Records are sorted by id and gzip is written with mtime=0, so an unchanged corpus
exports to byte-identical output and the SHA-256 is stable.

Usage: python -m ingest.export_corpus     (read-only; no API calls, no spend)
"""

import gzip
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import chromadb

from ingest.config import CHROMA_PATH, NOTEBOOK_SUMMARIES
from pipeline.embeddings import EMBED_MODEL

OUT_DIR = Path("results/corpus_export")
JSONL_PATH = OUT_DIR / "notebook_summaries.jsonl.gz"
MANIFEST_PATH = OUT_DIR / "manifest.json"


def _records() -> list[dict]:
    collection = chromadb.PersistentClient(path=str(CHROMA_PATH)).get_collection(
        name=NOTEBOOK_SUMMARIES
    )
    got = collection.get(include=["documents", "metadatas"])
    records = [
        {"id": doc_id, **metadata, "text": document}
        for doc_id, document, metadata in zip(got["ids"], got["documents"], got["metadatas"])
    ]
    records.sort(key=lambda record: record["id"])
    return records


def _write_jsonl(*, records: list[dict]) -> str:
    """Write deterministically and return the SHA-256 of the file."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    payload = "".join(json.dumps(record, sort_keys=True) + "\n" for record in records).encode()
    with open(JSONL_PATH, "wb") as raw:
        # mtime=0: gzip otherwise stamps the current time into the header, so an
        # unchanged corpus would hash differently on every export.
        with gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as handle:
            handle.write(payload)
    return hashlib.sha256(JSONL_PATH.read_bytes()).hexdigest()


def export() -> dict:
    records = _records()
    digest = _write_jsonl(records=records)
    scores = [record["kaggle_score"] for record in records if "kaggle_score" in record]
    manifest = {
        "collection": NOTEBOOK_SUMMARIES,
        "documents": len(records),
        "competitions": len({record["competition_id"] for record in records}),
        "summary_models": sorted({record["summary_model"] for record in records}),
        "embedding_model": EMBED_MODEL,
        # Observed rather than declared: the selection filters are not stored per
        # record, so the manifest reports what the data shows and lets a reader
        # confirm the filter rather than take an assertion on trust.
        "documents_with_score": len(scores),
        "min_kaggle_score": min(scores) if scores else None,
        "export_sha256": digest,
        "export_bytes": JSONL_PATH.stat().st_size,
        "exported_at": datetime.now(tz=timezone.utc).isoformat(),
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def fingerprint() -> dict:
    """The subset of the manifest recorded on every run, linking results to inputs.

    Returns nulls when no export exists, so a run is never blocked — but an eval
    run whose provenance carries a null corpus hash cannot be tied to a specific
    corpus afterwards, and the export should be made before eval runs begin.
    """
    if not MANIFEST_PATH.exists():
        return {"corpus_sha256": None, "corpus_documents": None, "corpus_competitions": None}
    manifest = json.loads(MANIFEST_PATH.read_text())
    return {
        "corpus_sha256": manifest["export_sha256"],
        "corpus_documents": manifest["documents"],
        "corpus_competitions": manifest["competitions"],
    }


if __name__ == "__main__":
    for key, value in export().items():
        print(f"{key}: {value}")
