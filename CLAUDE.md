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

6. **Retrieval informs, never bounds.** Stage prompts carry the shared
   `RETRIEVAL_STANCE` (see `pipeline/nodes.py`): retrieved excerpts are
   evidence of past practice to be weighed critically, not a limit on the
   model's reasoning. Don't write prompts that restrict answers to corpus
   content — uncritical adoption of provided knowledge is the AssistedDS
   failure mode Condition C exists to beat.

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
├── toy.py             # two-stage smoke pipeline (understand → advise), Anthropic-only
├── state.py           # PipelineState TypedDict for the full four-stage chain
├── nodes.py           # parse_problem, surface_signals, flag_assumptions, advise_approach
│                       #   — one retrieve() call per stage + RETRIEVAL_STANCE
├── schemas.py          # ParsedProblem, SurfacedSignals, AssumptionFlags, Advice — Pydantic
│                        #   stage-output contracts, enforced via native structured outputs
├── llm_client.py        # call_llm() — swappable Anthropic / Ollama backend, schema-constrained
├── retriever.py          # retrieve() — single retrieval seam; leave-one-out enforced here
├── embeddings.py          # embed() — voyage-code-3 for documents and queries, batched
├── runner.py               # run(raw_problem, competition_id) — entry point over the graph
└── graph.py                 # build_graph() that wires the four nodes

ingest/                # offline corpus build — never imported by eval-time pipeline code
├── config.py          # sources, paths, collection names, Lite-22 list
├── download.py        # Code4ML CSVs + mle-bench descriptions → data/raw/
├── chunking.py        # cell-level chunks, 4096-char cap, blank-line splitting
├── store.py           # Chroma write helper (cosine, explicit embeddings)
├── ingest_metadata.py # → competition_metadata collection
└── ingest_notebooks.py# → practitioner_knowledge collection

tests/                 # pytest; no real API calls — call_llm + retrieve mocked
notebooks/             # exploration, eval analysis
data/                  # raw downloads + ChromaDB store — gitignored
results/               # eval outputs — gitignored
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
  runs end-to-end via `pipeline/runner.py`.
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

**Next:**
1. Run the full dev-subset ingest (`ingest.download` → `ingest_metadata` →
   `ingest_notebooks`), then a live four-stage run on a real Lite
   competition — verify the LangSmith trace shows four filtered retrievals.
2. Calibrate `SIMILARITY_THRESHOLD` on 5–10 dev competitions (Haiku),
   documented in a notebook.
3. Draft the Condition B (unstructured/AssistedDS-style) baseline —
   single flat k=20 retrieval over the same corpus, one unstructured call.
4. Build the MLE-bench eval harness that scores A/B/C and produces the
   writeup numbers.

## Out of scope (don't propose these unprompted)

- Multi-tenancy, auth, deployment infra beyond the three containers
  described in README.
- Streaming responses — eval runs are batch.
- A web UI. LangSmith is our observability.
- Fine-tuning. The whole POC is about prompt-level structure.
