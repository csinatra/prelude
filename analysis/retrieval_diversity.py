"""What is a retrieved set actually matching on, and how concentrated is it?

Uniformly-structured summaries could in principle compress into a narrow region
of embedding space, so that similarity measures "is an ML notebook summary" more
than "solves a problem like this one". The 2026-08-11 audit found the 24
summaries retrieved for one competition spanning only ~0.019 of cosine
similarity, which raised exactly that question.

The measures here, and how to read each one:

  - *internal similarity vs a random sample.* A control for embedding
    degeneracy, NOT a diversity target. Retrieved sets SHOULD be more
    self-similar than random, because that is what similarity ranking does. The
    number is only alarming if it approaches 1.0 (near-duplicate documents) or
    if it fails to exceed random at all (the embedding is not discriminating).
  - *source competitions and their task types.* The load-bearing signal. If
    retrieval matches on TASK STRUCTURE, the sources will share a task type
    while spanning unrelated subject matter (a horror-fiction authorship problem
    pulling toxic-comment moderation notebooks). If it matches on surface TOPIC,
    sources will share subject matter instead. Task-structure matching is what
    the design wants.
  - *concentration.* Documents drawn from a single competition indicate corpus
    coverage limits for that task type, not a retrieval fault. This is the
    number that argues for corpus expansion.
  - *distinct techniques.* Descriptive only. Do not compare against the random
    baseline: a random draw spans unrelated task types and will show more
    technique variety by construction, which is not the kind of diversity this
    retrieval is meant to supply.

Results are corpus-state-dependent, so the output records the embedding model
and collection size alongside the numbers. Re-run after any corpus expansion,
summary-prompt change, or embedding-model change.

No LLM calls. One query embedding per competition; everything else is read from
the existing store.

Run: PYTHONPATH=. .venv/bin/python -m analysis.retrieval_diversity \
        --competitions spooky-author-identification,nomad2018-predict-transparent-conductors
"""

import argparse
import json
import random
from datetime import datetime, timezone
from pathlib import Path

import chromadb
import numpy as np

from pipeline.config import BASELINE_N_NOTEBOOKS, CHROMA_PATH, NOTEBOOK_SUMMARIES
from pipeline.embeddings import EMBED_MODEL
from pipeline.retriever import retrieve

DESCRIPTIONS_DIR = Path("data/raw/mlebench_descriptions")
DEFAULT_OUT = Path("results/retrieval_diversity.json")
RANDOM_SEED = 0

# Content-level check, independent of the embedding: a redundant set collapses
# onto a couple of these terms, a diverse one spreads across many.
TECHNIQUES = [
    "tf-idf", "tfidf", "count vector", "naive bayes", "logistic regression",
    "svm", "random forest", "xgboost", "lightgbm", "gradient boosting",
    "lstm", "gru", "cnn", "convolutional", "transformer", "bert", "embedding",
    "word2vec", "glove", "fasttext", "doc2vec", "svd", "pca", "ensemble",
    "stacking", "cross-validation", "stratified", "k-fold", "augmentation",
    "transfer learning", "fine-tun", "resnet", "densenet", "efficientnet",
]


def _mean_pairwise_similarity(*, vectors: np.ndarray) -> float:
    """Mean cosine similarity among all pairs in a set."""
    normed = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    sims = normed @ normed.T
    return float(sims[np.triu_indices(len(vectors), k=1)].mean())


def _task_types() -> dict[str, str]:
    """competition slug -> Code4ML task category, for the task-vs-topic read."""
    import pandas as pd

    from ingest.config import RAW_DIR

    frame = pd.read_csv(RAW_DIR / "code4ml" / "competitions.csv", index_col=0)
    return {
        str(row["data_sources"]): str(row["datatype"])
        for _, row in frame.iterrows()
        if pd.notna(row.get("datatype"))
    }


def _technique_count(*, texts: list[str]) -> tuple[int, dict[str, int]]:
    counts: dict[str, int] = {}
    for text in texts:
        lowered = text.lower()
        for term in TECHNIQUES:
            if term in lowered:
                counts[term] = counts.get(term, 0) + 1
    return len(counts), counts


def assess(*, competition_id: str, k: int = BASELINE_N_NOTEBOOKS) -> dict:
    """Compare a retrieved set's internal similarity against a random baseline."""
    raw_problem = (DESCRIPTIONS_DIR / f"{competition_id}.md").read_text()
    retrieved = retrieve(
        query=raw_problem,
        collection=NOTEBOOK_SUMMARIES,
        exclude_competition=competition_id,
        k=k,
    )
    ids = [doc.doc_id for doc in retrieved]
    if len(ids) < 2:
        return {"competition_id": competition_id, "error": "fewer than 2 docs retrieved"}

    collection = chromadb.PersistentClient(path=CHROMA_PATH).get_collection(
        name=NOTEBOOK_SUMMARIES
    )
    got = collection.get(ids=ids, include=["embeddings", "documents", "metadatas"])

    everything = collection.get(include=["metadatas"])
    taken = set(ids)
    pool = [
        doc_id
        for doc_id, meta in zip(everything["ids"], everything["metadatas"])
        if meta["competition_id"] != competition_id and doc_id not in taken
    ]
    sampled = random.Random(RANDOM_SEED).sample(pool, min(len(ids), len(pool)))
    baseline = collection.get(ids=sampled, include=["embeddings", "documents"])

    retrieved_internal = _mean_pairwise_similarity(vectors=np.asarray(got["embeddings"]))
    random_internal = _mean_pairwise_similarity(vectors=np.asarray(baseline["embeddings"]))
    n_retrieved_techniques, technique_counts = _technique_count(texts=got["documents"])
    n_random_techniques, _ = _technique_count(texts=baseline["documents"])

    competitions: dict[str, int] = {}
    for meta in got["metadatas"]:
        competitions[meta["competition_id"]] = competitions.get(meta["competition_id"], 0) + 1

    task_types = _task_types()
    source_task_types: dict[str, int] = {}
    for source, count in competitions.items():
        label = task_types.get(source, "unlabeled")
        source_task_types[label] = source_task_types.get(label, 0) + count
    top_source_share = max(competitions.values()) / len(ids) if competitions else 0.0

    return {
        "competition_id": competition_id,
        "query_task_type": task_types.get(competition_id, "unlabeled"),
        "source_task_types": source_task_types,
        "top_source_share": round(top_source_share, 3),
        "n_docs": len(ids),
        "query_similarity_top": round(retrieved[0].similarity, 4),
        "query_similarity_bottom": round(retrieved[-1].similarity, 4),
        "query_similarity_spread": round(retrieved[0].similarity - retrieved[-1].similarity, 4),
        "retrieved_internal_similarity": round(retrieved_internal, 4),
        "random_internal_similarity": round(random_internal, 4),
        "excess_redundancy": round(retrieved_internal - random_internal, 4),
        "distinct_techniques_retrieved": n_retrieved_techniques,
        "distinct_techniques_random": n_random_techniques,
        "top_techniques": sorted(technique_counts.items(), key=lambda item: -item[1])[:8],
        "source_competitions": competitions,
    }


def assess_staged(*, competition_id: str) -> dict:
    """The same measures over Condition C's staged union, for comparison with B.

    Directed per-stage queries can reach sources a single flat query does not, so
    this is a direct read on one of C's hypothesized mechanisms: if C's union is
    less concentrated than B's flat set, breadth of practitioner knowledge is
    part of what staged retrieval buys, independent of the synthesis structure.

    Costs one parse call, since C's directed queries are built from parse's
    extracted fields and approximating them would measure queries the pipeline
    never issues.
    """
    from pipeline.config import STAGE_N_NOTEBOOKS
    from pipeline.nodes import advise_query, flag_query, parse_problem, surface_query
    from pipeline.retriever import retrieve_with_topup

    raw_problem = (DESCRIPTIONS_DIR / f"{competition_id}.md").read_text()
    parsed = parse_problem(
        {"raw_problem": raw_problem, "competition_id": competition_id, "stage_trace": []}
    )
    fields = {
        "task_type": parsed["task_type"],
        "evaluation_metric": parsed["evaluation_metric"],
        "goal": parsed["parsed_goal"],
    }
    seen: set[str] = set()
    for builder in (surface_query, flag_query, advise_query):
        retrieve_with_topup(
            query=builder(**fields),
            collection=NOTEBOOK_SUMMARIES,
            exclude_competition=competition_id,
            k=STAGE_N_NOTEBOOKS,
            seen=seen,
        )

    collection = chromadb.PersistentClient(path=CHROMA_PATH).get_collection(
        name=NOTEBOOK_SUMMARIES
    )
    got = collection.get(ids=sorted(seen), include=["embeddings", "documents", "metadatas"])
    competitions: dict[str, int] = {}
    for meta in got["metadatas"]:
        competitions[meta["competition_id"]] = competitions.get(meta["competition_id"], 0) + 1
    task_types = _task_types()
    source_task_types: dict[str, int] = {}
    for source, count in competitions.items():
        label = task_types.get(source, "unlabeled")
        source_task_types[label] = source_task_types.get(label, 0) + count
    n_techniques, _ = _technique_count(texts=got["documents"])
    return {
        "n_distinct_docs": len(seen),
        "internal_similarity": round(
            _mean_pairwise_similarity(vectors=np.asarray(got["embeddings"])), 4
        ),
        "distinct_techniques": n_techniques,
        "top_source_share": round(max(competitions.values()) / len(seen), 3) if competitions else 0.0,
        "source_task_types": source_task_types,
        "source_competitions": competitions,
    }


def main(*, competitions: list[str], out_path: Path | None, staged: bool = True) -> None:
    collection = chromadb.PersistentClient(path=CHROMA_PATH).get_collection(
        name=NOTEBOOK_SUMMARIES
    )
    report = {
        # The measurement only means anything relative to a corpus state.
        "provenance": {
            "embed_model": EMBED_MODEL,
            "collection": NOTEBOOK_SUMMARIES,
            "collection_count": collection.count(),
            "k": BASELINE_N_NOTEBOOKS,
            "random_seed": RANDOM_SEED,
            "measured_at": datetime.now(tz=timezone.utc).isoformat(),
        },
        "staged_comparison": staged,
        "results": [],
    }
    for slug in competitions:
        entry = assess(competition_id=slug)
        if staged and "error" not in entry:
            entry["staged_condition_c"] = assess_staged(competition_id=slug)
        report["results"].append(entry)

    for result in report["results"]:
        print(f"=== {result['competition_id']} ===")
        if "error" in result:
            print(f"  {result['error']}\n")
            continue
        print(f"  query task type:         {result['query_task_type']}")
        print(f"  source task types:       {result['source_task_types']}   <- task-vs-topic read")
        print(f"  concentration:           {result['top_source_share']:.0%} from one competition")
        print(f"  docs retrieved:          {result['n_docs']}")
        print(f"  query-similarity spread: {result['query_similarity_spread']} "
              f"({result['query_similarity_bottom']}–{result['query_similarity_top']})")
        print(f"  internal similarity:     retrieved={result['retrieved_internal_similarity']} "
              f"(random control {result['random_internal_similarity']})")
        print(f"  distinct techniques:     {result['distinct_techniques_retrieved']} (descriptive)")
        print(f"  source competitions:     {result['source_competitions']}")
        staged_result = result.get("staged_condition_c")
        if staged_result:
            print(f"  -- Condition C staged union ({staged_result['n_distinct_docs']} distinct) --")
            print(f"     concentration:        {staged_result['top_source_share']:.0%} "
                  f"(B: {result['top_source_share']:.0%})")
            print(f"     internal similarity:  {staged_result['internal_similarity']} "
                  f"(B: {result['retrieved_internal_similarity']})")
            print(f"     distinct techniques:  {staged_result['distinct_techniques']} "
                  f"(B: {result['distinct_techniques_retrieved']})")
            print(f"     source competitions:  {staged_result['source_competitions']}")
        print()

    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2))
        print(f"wrote {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--competitions", required=True, help="comma-separated competition slugs")
    parser.add_argument("--out", default=str(DEFAULT_OUT), help="'-' to skip writing")
    args = parser.parse_args()
    main(
        competitions=[s.strip() for s in args.competitions.split(",") if s.strip()],
        out_path=None if args.out == "-" else Path(args.out),
    )
