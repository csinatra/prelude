# CLAUDE.md — Prelude (spec-pipeline)

> Read this before writing code in this repo. It overrides any default
> behavior an agent might bring from training.
>
> **Note:** there is an unrelated `CLAUDE.md` in the parent
> `~/Documents/ClaudeWorkspace/` directory for a trading project (Quorum).
> It does **not** apply here. Ignore it.

## What this is

A research POC for an Anthropic Fellows Program application. The deliverable
is **evidence for a research claim**, not a production system. Optimize for
clarity, traceable experiments, and reproducibility over polish.

**Claim under test:** structured reasoning over retrieved organizational
knowledge improves ML problem specification — and downstream agent
performance on MLE-bench — *beyond* what unstructured knowledge provision
(AssistedDS) achieves.

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
   - **Evaluation runs** — Claude Sonnet (latest). Only for the runs whose
     results go in the writeup.
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
   amendments require a dated entry and re-judging of prior runs.

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
├── retriever.py             # retrieve() flat + retrieve_two_level() notebook-then-chunk;
│                             #   leave-one-out enforced in BOTH
├── embeddings.py             # embed() — voyage-code-3 for documents and queries, batched
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

docs/                  # RESEARCH_DESIGN.md, JUDGE_RUBRIC.md (FROZEN), COST_ESTIMATE.md
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

## What "done" looks like for the current phase

- ✅ Toy two-stage pipeline runs end-to-end against the real Anthropic API
  and produces a LangSmith trace.
- ✅ Real four-stage chain (`parse_problem → surface_signals →
  flag_assumptions → advise_approach`) built and wired in `pipeline/graph.py`,
  runs end-to-end via `pipeline/condition_c2.py`.
- ✅ Backend is swappable (`LLM_PROVIDER=anthropic|ollama`) via
  `pipeline/llm_client.py`, verified against a local Ollama model.
- ✅ Unit tests (`tests/test_pipeline.py`) cover each node plus a full run,
  with `call_llm` mocked — no real API calls.
- ✅ Stage outputs are Pydantic models (`pipeline/schemas.py`), enforced via
  native structured outputs — `messages.parse(output_format=...)` on
  Anthropic, JSON-schema `format` on Ollama. Verified end-to-end against
  both backends.
- ✅ RAG layer built: `ingest/` package (Code4ML + mle-bench descriptions →
  two ChromaDB collections), `pipeline/retriever.py` seam with leave-one-out
  filter, one directed retrieval per stage wired into `pipeline/nodes.py`,
  schemas/prompts refit for MLE-bench competition descriptions. Unit-tested
  (retrieval + LLM mocked); voyage-code-3 verified live.

- ✅ Dev-subset corpus ingested and live-verified: 1,005 metadata chunks,
  62,379 practitioner chunks; four-stage run on spooky-author-identification
  (Haiku, 54s) with leave-one-out held in all four retrievals.
- ✅ Design-review refactor: conditions renamed/extended to B1/B2/C1/C2
  (see docs/RESEARCH_DESIGN.md); SpecificationFlag/Recommendation schemas
  with evidence citation and flag linkage; two-level (notebook→chunk)
  retrieval as the shared practitioner-knowledge unit; analysis scaffold
  (frozen judge rubric, artifact preservation); pipeline/config.py central
  tunables.
- ✅ Full dev corpus: notebook_summaries complete (5,937 abstracts, pinned
  Haiku, `summary_model`-stamped). Observability: prompts, resolved model,
  and token usage traced at the `llm_client` seam; bulk ingest and tests
  deliberately untraced (LangSmith quota). Provenance: manifest.json per
  run artifact (git commit, model, condition coordinates).
- ✅ Spec-side harness (`harness/`): per-condition spec.md renderer
  (additive composition — each grid step changes exactly one thing),
  append-only runs.jsonl registry (status lifecycle spec_built →
  agent_run → graded, mergeable across machines), CLI runner with
  block/synthesis token split. Live-verified: all four conditions on
  spooky-author-identification, leave-one-out held, zero dangling flag
  references.
- ✅ Threshold calibration: 10-competition sweep (`analysis/calibration.py`,
  re-runnable). Decision, recorded pre-run in RESEARCH_DESIGN.md:
  `SIMILARITY_THRESHOLD=None` for v1 — stage-directed queries score ~0.06
  lower than flat against the same chunks, so any global cutoff breaks
  knowledge parity; top-k rank ordering is the quality control.

**Next:**
1. Cloud-box half of the harness: inject spec.md into AIDE inside the
   MLE-bench container (no LLM calls in there — core constraint 1),
   advance runs through agent_run → graded in runs.jsonl, wire MLE-bench
   grading + secondary metrics into the registry.
2. Corpus expansion before production runs: remaining Code4ML summaries
   via the Anthropic Batch API (~$200 decision recorded in
   COST_ESTIMATE.md; needs a batch-mode ingest variant), then size
   MLEModernizer after opening the tarball on the cloud box (unit count +
   native-abstract check first). Re-run the calibration sweep after any
   expansion — thresholds are corpus-relative.
3. Eval runs per RESEARCH_DESIGN.md (B1/B2/C1-pilot/C2, ~10 Lite
   competitions × 3 seeds, Sonnet) + mechanistic judging via
   analysis/judge.py against the frozen rubric.

## Out of scope (don't propose these unprompted)

- Multi-tenancy, auth, deployment infra beyond the three containers
  described in README.
- Streaming responses — eval runs are batch.
- A web UI. LangSmith is our observability.
- Fine-tuning. The whole POC is about prompt-level structure.
