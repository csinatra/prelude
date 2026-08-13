"""Build the notebook_summaries collection — the practitioner-knowledge retrieval unit.

One record per unique kaggle_id in the Lite-22 code-block slice: gather the
notebook's blocks, generate an LLM abstract of its approach, embed the abstract
with Voyage, store with {kaggle_id, competition_id, kaggle_score, summary_model}.

Two generation backends:
  - default (synchronous, ThreadPoolExecutor) — fine for the dev slice.
  - --batch (Anthropic Message Batches API, 50% discount) — for the full-corpus
    scale-up; submits all pending as async batches, persists batch ids so a long
    run resumes after interruption, then embeds + upserts the results.

Resumable either way: kaggle_ids already in the collection are skipped.

Usage: python -m ingest.ingest_summaries [--limit N] [--rebuild] [--batch]
--limit caps how many pending notebooks this invocation summarizes (staged spends).
--rebuild drops the collection first — REQUIRED to regenerate after a summary-
  prompt or embedding-model change, since skip-existing would otherwise leave the
  old summaries untouched.
--batch uses the Message Batches API. A re-run resumes an in-flight batch (and
  ignores --rebuild while one is pending — delete the state file to force a
  fresh rebuild).
"""

import argparse
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import anthropic
import pandas as pd

from ingest.config import CHROMA_PATH, LITE_COMPETITIONS, NOTEBOOK_SUMMARIES, RAW_DIR
from ingest.store import add_documents, drop_collection, get_collection
from pipeline.llm_client import call_llm_text

# Pinned deliberately, NOT the MODEL env var: summaries are corpus
# infrastructure shared identically by every condition and every run — they
# are exempt from the Haiku-dev/Sonnet-eval switch, and the resumable ingest
# must never produce a mixed-model collection. Each record carries
# summary_model metadata so an improper insert is auditable and deletable by
# filter.
SUMMARY_MODEL = "claude-haiku-4-5-20251001"

# Abstract input cap; blocks concatenated in order. Raised from 12k (which fully
# covered only ~79% of notebooks) so the summarizer sees the whole notebook —
# modeling/validation cells cluster at the END, so a tight front-biased cap
# starved the abstract of its most transferable content. 60k covers ~99% of
# Code4ML whole while still capping the pathological multi-hundred-KB outliers;
# the input-cost delta over 40k is ~$4 across the full corpus (the richer prompt
# below, not the cap, drives the Batch re-cost).
MAX_NOTEBOOK_CHARS = 60_000
MAX_SUMMARY_TOKENS = 1024
LLM_WORKERS = 8
UPSERT_BATCH = 200

# Message Batches API caps: 100k requests AND 256MB per batch — chunk under both.
MAX_BATCH_REQUESTS = 100_000
MAX_BATCH_BYTES = 200_000_000
POLL_SECONDS = 30
# In-flight batch ids, so a long --batch run resumes collection after interruption
# instead of re-submitting (double spend). Sibling of the Chroma store; gitignored.
BATCH_STATE_PATH = Path(CHROMA_PATH).parent / "summaries_batch_state.json"

SUMMARY_SYSTEM = (
    "You are an experienced ML engineer distilling a solution notebook into a knowledge-base "
    "summary, so that another engineer facing a DIFFERENT but related problem can learn from it.\n\n"
    "<task>\n"
    "Write a compact technical abstract of the notebook's approach, emphasizing what transfers "
    "across problems of this class over dataset-specific detail.\n"
    "</task>\n\n"
    "<cover>\n"
    "Where present, weave in: the modeling approach (specific estimators or architectures and "
    "notable hyperparameter/configuration choices); feature engineering and data transformations "
    "(name the concrete derived features or representations); the validation strategy (resampling "
    "scheme and target metric); key preprocessing; and any distinctive or failure-mode-avoiding "
    "techniques. Prefer a few reusable specifics over exhaustive coverage.\n"
    "</cover>\n\n"
    "<format>\n"
    "Write in plain, flowing prose — complete sentences in a few short paragraphs. Do not use "
    "markdown, section headers, bold or italic, or bulleted or numbered lists. Do not mention the "
    "competition, leaderboard, or Kaggle. Let length track how much genuinely transferable insight "
    "the notebook holds, not its raw size: prioritize distinctive, reusable specifics and compress "
    "routine steps. Most summaries should land around 250-350 words; a genuinely information-dense "
    "notebook may run longer, but never pad to reach a length.\n"
    "</format>"
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


def _notebook_text(entry: dict) -> str:
    return "\n\n".join(entry["blocks"])[:MAX_NOTEBOOK_CHARS]


_HEADER_RE = re.compile(r"^\s{0,3}#{1,6}\s+\S")


def _clean_summary(text: str) -> str:
    """Drop a stray leading markdown header the format rules can't fully suppress
    (Haiku occasionally prepends a '# Title' line ~a third of the time).

    Conservative by construction: removes only leading blank and header lines and
    stops at the first prose line — since prose never starts with '# ', it cannot
    remove a content block, only a redundant title. The body is left untouched.
    """
    lines = text.split("\n")
    start = 0
    while start < len(lines) and (not lines[start].strip() or _HEADER_RE.match(lines[start])):
        start += 1
    return "\n".join(lines[start:]).strip()


def _metadata_for(*, kaggle_id: int, entry: dict) -> dict:
    metadata = {
        "kaggle_id": kaggle_id,
        "competition_id": entry["competition_id"],
        "summary_model": SUMMARY_MODEL,
    }
    if entry["kaggle_score"] is not None:
        metadata["kaggle_score"] = entry["kaggle_score"]
    return metadata


def _pending(*, notebooks: dict[int, dict], collection, limit: int | None) -> dict[int, dict]:
    existing = set(collection.get(ids=[f"nb_{kid}" for kid in notebooks])["ids"])
    pending = {kid: entry for kid, entry in notebooks.items() if f"nb_{kid}" not in existing}
    print(f"already summarized: {len(existing)}; pending: {len(pending)}")
    if limit is not None:
        pending = dict(list(pending.items())[:limit])
        print(f"capped this run to {len(pending)} notebooks (--limit {limit})")
    return pending


# ── synchronous backend ──────────────────────────────────────────────────

def _summarize(*, kaggle_id: int, entry: dict) -> tuple[int, str]:
    abstract = call_llm_text(
        system=SUMMARY_SYSTEM,
        user=f"Notebook code cells:\n{_notebook_text(entry)}",
        max_tokens=MAX_SUMMARY_TOKENS,
        model=SUMMARY_MODEL,
    )
    return kaggle_id, _clean_summary(abstract)


def _run_sync(*, pending: dict[int, dict], collection) -> None:
    ids: list[str] = []
    texts: list[str] = []
    metadatas: list[dict] = []

    def flush(done: int, total: int) -> None:
        if ids:
            add_documents(collection=collection, ids=ids, texts=texts, metadatas=metadatas)
            ids.clear()
            texts.clear()
            metadatas.clear()
            print(f"progress: {done}/{total}")

    with ThreadPoolExecutor(max_workers=LLM_WORKERS) as pool:
        futures = [
            pool.submit(_summarize, kaggle_id=kid, entry=entry) for kid, entry in pending.items()
        ]
        for done, future in enumerate(futures, start=1):
            kaggle_id, abstract = future.result()
            ids.append(f"nb_{kaggle_id}")
            texts.append(abstract)
            metadatas.append(_metadata_for(kaggle_id=kaggle_id, entry=pending[kaggle_id]))
            if len(ids) >= UPSERT_BATCH:
                flush(done, len(futures))
    flush(len(pending), len(pending))


# ── batch backend (Message Batches API) ──────────────────────────────────

def _request_for(*, kaggle_id: int, entry: dict) -> dict:
    return {
        "custom_id": f"nb_{kaggle_id}",
        "params": {
            "model": SUMMARY_MODEL,
            "max_tokens": MAX_SUMMARY_TOKENS,
            "system": SUMMARY_SYSTEM,
            "messages": [{"role": "user", "content": f"Notebook code cells:\n{_notebook_text(entry)}"}],
        },
    }


def _chunk_requests(requests: list[dict]) -> list[list[dict]]:
    """Split into batches under the per-batch request-count and byte caps."""
    chunks: list[list[dict]] = []
    current: list[dict] = []
    current_bytes = 0
    for request in requests:
        size = len(request["params"]["messages"][0]["content"].encode()) + len(SUMMARY_SYSTEM)
        if current and (len(current) >= MAX_BATCH_REQUESTS or current_bytes + size > MAX_BATCH_BYTES):
            chunks.append(current)
            current, current_bytes = [], 0
        current.append(request)
        current_bytes += size
    if current:
        chunks.append(current)
    return chunks


def _submit(*, pending: dict[int, dict], client: anthropic.Anthropic) -> list[str]:
    requests = [_request_for(kaggle_id=kid, entry=entry) for kid, entry in pending.items()]
    batch_ids: list[str] = []
    for chunk in _chunk_requests(requests):
        batch = client.messages.batches.create(requests=chunk)
        batch_ids.append(batch.id)
        print(f"submitted batch {batch.id} ({len(chunk)} requests)")
    return batch_ids


def _collect(*, batch_ids: list[str], client: anthropic.Anthropic, notebooks: dict[int, dict], collection) -> None:
    collected = 0
    for batch_id in batch_ids:
        while True:
            status = client.messages.batches.retrieve(batch_id).processing_status
            if status == "ended":
                break
            print(f"batch {batch_id}: {status} — waiting {POLL_SECONDS}s")
            time.sleep(POLL_SECONDS)
        ids: list[str] = []
        texts: list[str] = []
        metadatas: list[dict] = []
        errored = 0
        for result in client.messages.batches.results(batch_id):
            if result.result.type != "succeeded":
                errored += 1
                continue
            kaggle_id = int(result.custom_id.removeprefix("nb_"))
            entry = notebooks.get(kaggle_id)
            if entry is None:
                continue
            ids.append(result.custom_id)
            texts.append(_clean_summary(result.result.message.content[0].text))
            metadatas.append(_metadata_for(kaggle_id=kaggle_id, entry=entry))
        if ids:
            add_documents(collection=collection, ids=ids, texts=texts, metadatas=metadatas)
        collected += len(ids)
        print(f"batch {batch_id}: {len(ids)} ok, {errored} errored/expired")
    print(f"collected {collected} summaries")


def _check_resume_matches(*, state: dict, notebooks: dict[int, dict]) -> None:
    """Refuse to collect a batch that was generated against a different corpus.

    The state file guards against double submission, but not against the corpus
    changing underneath an in-flight batch. If the notebook set was rebuilt or
    re-sliced between submit and resume (a --rebuild, an ingest config change),
    the returned summaries describe notebooks that may no longer exist, and
    upserting them would silently mix generations. Stop instead.
    """
    submitted = set(state.get("custom_ids", []))
    if not submitted:
        return  # state written before this check existed; nothing to verify against
    available = {f"nb_{kaggle_id}" for kaggle_id in notebooks}
    missing = submitted - available
    if missing:
        raise SystemExit(
            f"batch state mismatch: {len(missing)} of {len(submitted)} submitted notebooks "
            f"are no longer in the loaded slice (e.g. {sorted(missing)[:3]}). The corpus "
            "changed after submission, so collecting would mix generations. Resolve by "
            f"deleting {BATCH_STATE_PATH} to abandon the in-flight batch, then re-run."
        )


def _run_batch(*, notebooks: dict[int, dict], collection, limit: int | None) -> None:
    client = anthropic.Anthropic()
    if BATCH_STATE_PATH.exists():
        state = json.loads(BATCH_STATE_PATH.read_text())
        _check_resume_matches(state=state, notebooks=notebooks)
        batch_ids = state["batch_ids"]
        print(f"resuming: collecting {len(batch_ids)} in-flight batch(es)")
    else:
        pending = _pending(notebooks=notebooks, collection=collection, limit=limit)
        if not pending:
            print("nothing pending")
            return
        batch_ids = _submit(pending=pending, client=client)
        BATCH_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
        BATCH_STATE_PATH.write_text(
            json.dumps(
                {
                    "batch_ids": batch_ids,
                    "custom_ids": [f"nb_{kaggle_id}" for kaggle_id in pending],
                }
            )
        )
    _collect(batch_ids=batch_ids, client=client, notebooks=notebooks, collection=collection)
    BATCH_STATE_PATH.unlink(missing_ok=True)


def main(*, limit: int | None = None, rebuild: bool = False, batch: bool = False) -> None:
    # Bulk corpus infrastructure: thousands of LLM calls per run. Tracing each
    # one burns the LangSmith monthly trace quota (it did, 2026-07-15) and adds
    # nothing — the resumable collection + summary_model metadata are the audit
    # trail here. Experiment-time pipeline calls stay traced.
    os.environ["LANGSMITH_TRACING"] = "false"
    os.environ["LANGSMITH_TRACING_V2"] = "false"
    resuming = batch and BATCH_STATE_PATH.exists()
    if rebuild and not resuming:
        drop_collection(name=NOTEBOOK_SUMMARIES)
        BATCH_STATE_PATH.unlink(missing_ok=True)
        print(f"dropped {NOTEBOOK_SUMMARIES} for rebuild")
    elif rebuild and resuming:
        print("warning: --rebuild ignored — an in-flight batch is pending collection")
    collection = get_collection(name=NOTEBOOK_SUMMARIES)
    notebooks = _load_notebooks()
    print(f"unique notebooks in slice: {len(notebooks)}")

    if batch:
        _run_batch(notebooks=notebooks, collection=collection, limit=limit)
    else:
        pending = _pending(notebooks=notebooks, collection=collection, limit=limit)
        _run_sync(pending=pending, collection=collection)
    print(f"collection count: {collection.count()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--rebuild", action="store_true", help="drop the collection first (prompt/model change)"
    )
    parser.add_argument(
        "--batch", action="store_true", help="use the Message Batches API (50%% discount, async)"
    )
    args = parser.parse_args()
    main(limit=args.limit, rebuild=args.rebuild, batch=args.batch)
