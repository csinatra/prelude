# CLAUDE.md — Prelude (spec-pipeline)

> Read this before writing code in this repo. It overrides any default
> behavior an agent might bring from training.

## What this is

A research POC for an Anthropic Fellows Program application. The deliverable
is **evidence for a research claim**, not a production system. Optimize for
clarity, traceable experiments, and reproducibility over polish.

**Claim under test:** directed retrieval and structured reasoning over
retrieved **practitioner knowledge** improves ML problem specification, and the
downstream agent performance it drives on MLE-bench, beyond what unstructured
knowledge retrieval (AssistedDS) achieves. The corpus is public practitioner
notebooks, not organizational knowledge; institutional knowledge is the
direction this generalizes toward, not what the POC retrieves from.

See [README.md](README.md) for the three-condition eval design, the four
pipeline stages, and prior art.

## Core constraints

1. **No LLM API calls from inside the MLE-bench execution environment.**
   The spec pipeline runs as a pre-processing step *outside* the MLE-bench
   container. Output is serialized and injected into AIDE as initial
   context. Any code path that would make an Anthropic/OpenAI/etc. call from
   inside the eval container is a research-integrity bug — flag it.

2. **Model selection by purpose.**
   - **Development iteration** — Claude Haiku (`claude-haiku-4-5-20251001`).
     Fast, cheap, good enough to validate pipeline shape.
   - **Evaluation runs** — the pinned `EVAL_MODEL` in `pipeline/config.py`
     (`claude-sonnet-5`), chosen pre-run and recorded in
     docs/RESEARCH_DESIGN.md alongside the other pre-registered decisions.
     Only for the runs whose results go in the writeup.
   - Keep the model name in one place per module (a constant) so swapping
     is one edit.
   - **Local Ollama backend** — `pipeline/llm_client.py` also supports
     `LLM_PROVIDER=ollama` for free, offline smoke-testing of pipeline
     *wiring* (does the graph run end-to-end?). Never for eval runs, and
     not a substitute for judging output quality — local open-source
     models are meaningfully weaker at this task than Haiku/Sonnet.

3. **Every pipeline stage produces an explicit, inspectable intermediate
   output.** That's the whole point — Condition C's advantage over
   Condition B comes from forced structure. Don't collapse stages or let
   one node "do everything."

4. **API keys via `os.environ` only.** `.env` is gitignored. Never write a
   key literal in any file. If you see one in code, replace it with
   `os.environ["..."]`.

5. **Leave-one-out retrieval is a research-integrity invariant.** No
   retrieval may return artifacts from the competition currently being
   specified. The filter (`competition_id != current`) is enforced inside
   `pipeline/retriever.py::retrieve()` — every node goes through that seam;
   never query ChromaDB directly from a node. A code path that bypasses the
   filter is solution leakage — flag it.

6. **Retrieval informs, never bounds — in C2 only.** C2's stage prompts
   carry the shared `RETRIEVAL_STANCE` (see `pipeline/nodes.py`): retrieved
   excerpts are evidence to be weighed critically, not a limit on
   reasoning. B2 and C1's freeform synthesis is deliberately stance-free —
   uncritical adoption is the AssistedDS failure mode those conditions must
   be free to exhibit. Don't "fix" their prompts by adding the stance, and
   don't restrict any condition's answers to corpus content.

7. **The judge rubric is frozen.** `docs/JUDGE_RUBRIC.md` was written before
   any evaluation run and must not be revised after seeing results —
   amendments require a dated entry and re-judging of prior runs. A SHA-256
   pin test (deferred 2026-07-19) will mechanically enforce this
   immediately before eval runs begin.

8. **Document-budget parity is a research-validity invariant.** The
   retrieval unit is the notebook **summary** (one notebook = one document).
   Condition B and the staged conditions must reason over the same number of
   **distinct** notebook-summary documents, plus equal metadata (`METADATA_K`
   ~ parse). B draws `BASELINE_N_NOTEBOOKS` in one flat pass; C's 3 directed
   stages each contribute `STAGE_N_NOTEBOOKS` *new* distinct docs via cross-stage
   top-up (`retriever.retrieve_with_topup` — repeats across stages are retained
   as an importance signal and backfilled with the next-best unseen doc), so
   `BASELINE_N_NOTEBOOKS = 3 × STAGE_N_NOTEBOOKS`. Changing any budget in
   `pipeline/config.py` requires re-checking this parity; a mismatch invalidates
   cross-condition comparison. Parity is on *distinct documents*, not injected
   tokens — the token asymmetry from retained repeats is logged per run
   (block/synthesis split) and reported, not equalized.

## Commands

```bash
pytest tests/ -q                       # run all tests (LLM + retrieval mocked)
python -m pipeline.toy                 # toy two-stage smoke pipeline
python -m harness.runner --competition spooky-author-identification \
    --condition C2 --seed 0            # one condition on one competition
python -m analysis.calibration        # threshold sweep — RETIRED, see RESEARCH_DESIGN
                                      #   (still gated if ever run: LLM calls)
python -m ingest.download             # corpus downloads (APPROVAL REQUIRED)
python -m ingest.ingest_metadata      # Chroma writes  (APPROVAL REQUIRED)
python -m ingest.ingest_notebooks     # Chroma writes  (APPROVAL REQUIRED)
python -m ingest.ingest_summaries     # LLM spend      (APPROVAL REQUIRED)
```

Required env (see `.env.example` for the authoritative list): `ANTHROPIC_API_KEY`,
`VOYAGE_API_KEY`, `MODEL`, `LANGSMITH_API_KEY`, `LANGSMITH_TRACING_V2`,
`LANGSMITH_PROJECT`; optional: `LLM_PROVIDER` (default `anthropic`),
`OLLAMA_HOST`/`OLLAMA_MODEL` (Ollama backend only), `SIMILARITY_THRESHOLD`,
`CHROMA_PATH`.

## Approval required — never run unprompted

Build and modify these code paths freely, but never *execute* them without
an explicit operator instruction in the current session:

- Any eval run using the pinned `EVAL_MODEL` (Sonnet)
- Any Anthropic Batch API job (e.g. the ~$200 corpus summarization)
- Corpus ingest or re-ingest (any Chroma write: `ingest.*` modules)
- Calibration sweeps (`analysis.calibration` — LLM calls per competition)
- Anything that consumes LangSmith trace quota in bulk

## Keeping the decision log

`docs/DECISIONS.md` is the dated audit trail for the writeup. When a change
affects the experimental design, the corpus, the run matrix, or anything a
reviewer would want justified, append an entry the same day.

Record a **decision** ("we chose X over Y because Z"), not a **milestone** ("we
did X and it worked") and not documentation housekeeping. Milestones belong in
`docs/PROGRESS.md`; reorganizing a doc needs no entry at all. A useful test: if
the entry has no alternative that was rejected and no consequence a reviewer
would question, it is not a decision.

- Keep an entry short where a line will do, with the full rationale at the
  pointer. Extend it when the reasoning *is* the decision and would otherwise
  live nowhere.
- Append only. Never rewrite or delete a past entry, even a wrong one.
- A reversed decision gets a **new dated entry recording the reversal and why**,
  leaving the original in place. The change of mind is part of the record.
- Pre-registration discipline: anything touching hypotheses, metrics, or the
  analysis plan must be logged *before* eval runs, not after.

## Execution environments

Two machines, split by responsibility; `results/runs.jsonl` is the merge
point between them (append-only, merge = concatenate):

- **Local MacBook** — pipeline dev, spec-side harness, dev-corpus work,
  analysis. All spec-pipeline LLM calls happen here.
- **Cloud box (GPU)** — MLE-bench containers, AIDE injection, the
  MLEModernizer tarball, `agent_run`/`graded` registry stages.

When a task belongs to the other machine, build the code here but do not
attempt to execute it. See [docs/RUNBOOK.md](docs/RUNBOOK.md) for the
operational workflow.

## Behavioral guidelines (apply to every task)

Adapted from [multica-ai/andrej-karpathy-skills](https://github.com/multica-ai/andrej-karpathy-skills/blob/main/CLAUDE.md). Bias toward caution over speed; use judgment on trivial tasks.

**1. Think before coding.** State assumptions explicitly. If multiple interpretations exist, surface them — don't pick silently. If a simpler approach exists, say so. If something is unclear, stop and ask. Don't hide confusion.

**2. Simplicity first.** Minimum code that solves the problem. No speculative features, no abstractions for single-use code, no configurability that wasn't requested, no error handling for impossible scenarios. If you wrote 200 lines and it could be 50, rewrite it. Would a senior engineer call this overcomplicated? If yes, simplify.

**3. Surgical changes.** Touch only what the task requires. Don't "improve" adjacent code, comments, or formatting. Don't refactor things that aren't broken. Match existing style even if you'd do it differently. If you notice unrelated dead code, *mention* it — don't delete it. Every changed line should trace directly to the request. Clean up orphans your changes create (now-unused imports/vars); leave pre-existing dead code alone unless asked.

**4. Goal-driven execution.** Translate vague tasks into verifiable goals before starting:
- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Tests pass before and after"

For multi-step work, state a brief plan with a verification step per item:
```
1. [step] → verify: [check]
2. [step] → verify: [check]
```

Strong success criteria let you loop independently. "Make it work" doesn't.

**Research-POC corollary:** in *this* repo specifically, "done" often means "produces a trace I can inspect in LangSmith and a result I can put in a writeup," not "passes a green test suite." When success criteria are ambiguous, ask which one applies.

## Coding conventions

- **Python 3.12.** Type hints on every public function.
- **Keyword arguments over positional** — *everywhere*, even for one-arg
  calls when the API supports it. This is an explicit user preference.
  `func(x=value)` not `func(value)`.
- **`TypedDict` for LangGraph state**, `total=False` so stages populate
  incrementally. Nodes take state, return a *partial dict* of updates —
  never mutate in place.
- **Pydantic models for stage outputs** once we move past the toy. Schema
  is the contract between stages and the documentation of the pipeline.
- Logging: `print()` is fine for the toy; use `logging` once we have the
  real graph. No `loguru` needed at this scale.

## Layout

```
pipeline/
├── __init__.py
├── config.py          # central tunables: budgets, collections, threshold.
│                       #   PARITY INVARIANT: B's flat budget == staged total
├── toy.py              # two-stage smoke pipeline (understand → advise), Anthropic-only
├── state.py             # PipelineState TypedDict for the full four-stage chain
├── nodes.py              # four C2 stage nodes + shared query builders (surface_query etc.)
├── schemas.py             # Pydantic stage contracts incl. SpecificationFlag (categorized,
│                           #   evidence-cited, confidence-rated) and Recommendation
│                           #   (approach/tradeoff/failure_mode linked to flags by flag_id)
├── llm_client.py           # call_llm() schema-constrained + call_llm_text() freeform
├── retriever.py             # retrieve() flat top-k + retrieve_with_topup() staged
│                             #   (cross-stage distinct-doc top-up); leave-one-out in BOTH
├── embeddings.py             # embed() — voyage-4-large for documents and queries, batched
├── condition_b.py             # Condition B: run_b1() raw block, run_b2() freeform pass
├── condition_c1.py             # Condition C1: staged retrieval + B2 freeform synthesis
├── condition_c2.py              # run_c2() — Condition C2 entry point over the graph
└── graph.py                      # build_graph() that wires the four nodes

ingest/                # offline corpus build — never imported by eval-time pipeline code
├── config.py          # sources, paths, collection names, Lite-22 list
├── download.py        # Code4ML CSVs + mle-bench descriptions → data/raw/
├── chunking.py        # cell-level chunks, 4096-char cap, blank-line splitting
├── store.py           # Chroma write helper (cosine, explicit embeddings)
├── ingest_metadata.py # → competition_metadata collection
├── ingest_notebooks.py# → practitioner_knowledge collection
└── ingest_summaries.py# → notebook_summaries (one LLM abstract/notebook; resumable)

analysis/              # post-run mechanistic analysis (scaffold until runs exist)
├── judge.py           # per-flag judging vs docs/JUDGE_RUBRIC.md + per-category aggregation
└── artifacts.py       # results/{comp}_{condition}_{seed}/ preservation layout

docs/                  # RESEARCH_DESIGN.md, JUDGE_RUBRIC.md (FROZEN), COST_ESTIMATE.md,
                       #   RUNBOOK.md, PROGRESS.md (milestone history), DECISIONS.md (audit trail)
tests/                 # pytest; no real API calls — call_llm + both retrieval seams mocked
notebooks/             # exploration, eval analysis
data/                  # raw downloads + ChromaDB store — gitignored
results/               # run artifacts — gitignored during dev, committed for final evals
```

Corpus specifics (sources, collections, coverage caveats) live in
[README.md](README.md#corpus-and-retrieval). Two operational notes for agents:
the Code4ML `сode_blocks_*.csv` filenames on Zenodo start with a **Cyrillic
"с"** (U+0441) — copy URLs from `ingest/config.py`, don't retype them; and the
brief's MLEModernizer source is a single 107 GB tar.gz, deferred to the cloud
box — don't try to download it locally.

## LangGraph patterns used in this repo

These are the canonical shapes — match them when adding nodes.

```python
from typing import TypedDict
from langgraph.graph import END, START, StateGraph

class SpecState(TypedDict, total=False):
    problem_statement: str
    understanding: dict       # → swap to pydantic model
    # ...

def my_node(state: SpecState) -> dict:
    # read from state, return partial update
    return {"understanding": {...}}

graph = StateGraph(state_schema=SpecState)
graph.add_node(node="understand", action=my_node)
graph.add_edge(start_key=START, end_key="understand")
graph.add_edge(start_key="understand", end_key=END)
app = graph.compile()
result = app.invoke(input={"problem_statement": "..."})
```

## Current state

Four-stage C2 pipeline plus B1/B2/C1 conditions built and unit-tested.
Retrieval unit is the notebook **summary** (code-chunk retrieval retired after a
2026-08-03 probe); staged conditions use directed per-stage retrieval with
cross-stage top-up for distinct-document parity with B. Prompts generalized
beyond Kaggle framing; summaries are richer whole-notebook abstracts (pinned
Haiku). Embedding model switched to `voyage-4-large` (general-purpose fits the
now NL↔NL retrieval); batch-mode summary ingest built (`ingest_summaries
--batch`, Message Batches API).

Spec-side harness working: per-condition spec.md renderer, append-only registry,
per-run artifacts with provenance + usage ledger. Runs are isolated by
experiment stage (`PRELUDE_REGISTRY_STAGE`, default `dev`) — see
[docs/DATA.md](docs/DATA.md) for the publish boundary. Cloud-box verified
end-to-end (2026-07-24): box provisioned, smoke run GREEN (Haiku → valid graded
submission), B/C spec injection confirmed via the PRELUDE_SPEC_PATH mount.

**Corpus rebuilt 2026-08-17** at eval scale: score-filtered Code4ML, 25,633
summaries over 580 competitions, new prompt, `voyage-4-large`, zero batch errors.
Exported with a SHA-256 fingerprint (`ingest.export_corpus`) that every run
manifest records. Retrieval re-characterized on it; `SIMILARITY_THRESHOLD` stays
`None` — per-competition medians span ~0.47–0.59, so no global cutoff is
meaningful and the threshold sweep is retired rather than re-run.
History: [docs/PROGRESS.md](docs/PROGRESS.md).

## Next
1. Re-verify the pipeline end-to-end against the rebuilt corpus — the conditions
   have not been run since the ingest.
2. Cloud-box harness: spec injection and a single-run smoke are verified
   (2026-07-24). Remaining: exercise the automated batch drain end-to-end, since
   the grade-via-JSONL and journal-metric seams are still unexercised on a real
   multi-run. Spec generation stays local (DECISIONS.md 2026-08-11 reverses the
   earlier all-cloud plan), so specs are built and inspected in one batch, then
   synced to the box with the registry rows before draining. Both machines must
   share `PRELUDE_REGISTRY_STAGE`.
3. AIDE budget calibration: confirm a step budget generous enough to show a
   convergence curve rather than merely a valid submission. Load-bearing for H3,
   and for H1 floor effects.
4. Pre-register the eval subset and seed list in DECISIONS.md on stated
   criteria, noting that retrieval properties were already known when pinning.
5. Eval runs per RESEARCH_DESIGN.md (B1/B2/C1-pilot/C2 on the POC-scope Lite
   subset × 3 seeds, Sonnet) + mechanistic judging via analysis/judge.py against
   the frozen rubric, with the blinded human anchor (docs/JUDGE_VALIDATION.md)
   run before the mechanistic writeup.

## Out of scope (don't propose these unprompted)

- Multi-tenancy, auth, deployment infra beyond the three containers
  described in README.
- Streaming responses — eval runs are batch.
- A web UI. LangSmith is our observability.
- Fine-tuning. The whole POC is about prompt-level structure.
