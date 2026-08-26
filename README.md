# Prelude

A research POC on how structured specification, built from retrieved practitioner knowledge, affects agentic execution in ML engineering.

> **Status (2026-08-17):** pipeline architecture and experimental design
> complete (B1 / B2 / C1 / C2, summary-level staged retrieval, mechanistic-analysis
> scaffold). AIDE injection path confirmed end-to-end via smoke run. Corpus
> rebuilt at eval scale (25,633 practitioner summaries over 580 competitions),
> and retrieval characterized against it.
> **Evaluation runs not yet executed.**

## Contents

- [Motivation](#motivation)
- [Research question](#research-question)
- [Conditions](#conditions)
- [Pipeline](#pipeline--four-stages-with-explicit-intermediate-outputs)
- [Corpus and retrieval](#corpus-and-retrieval)
- [Related work](#related-work)
- [Stack](#stack)
- [Architecture](#architecture)
- [Setup](#setup)
- [Run the toy pipeline](#run-the-toy-pipeline)
- [Run the full pipeline](#run-the-full-pipeline)
- [Layout](#layout)
- [Documentation](#documentation)
- [Tests](#tests)

## Motivation

Before an agent writes a line of code, a human expert does the part that's hardest to automate: they frame the problem. As agentic systems take on more of the work between a stated goal and a finished result, more of what determines success sits upstream of execution. When agents fall short of human experts on long research tasks the gap often isn't raw execution skill. RE-Bench found its agents lost ground by mismanaging accumulated context, holding onto "stubborn and incorrect assumptions" and failing to notice information that contradicted them. Errors around problem framing made early in the reasoning chain are exactly the kind that compound. Translating an ambiguous goal into something verifiable is a critical first step to building any solution, starting with the basic question of what the core problem actually is and what would count as a good answer.

Problem framing draws on things that don't show up cleanly in a task description. Things like institutional precedent, awareness of failure modes and implicit assumptions, judgment about what a metric actually needs to capture, and constraints that live outside the immediate problem statement. People with domain expertise supply this instinctively. Before ever committing to a direction, practitioners test an idea through rounds of conversation with collaborators and senior colleagues who have seen adjacent problems. An agentic system has significant knowledge baked into its parameters, but applying that knowledge to a specific problem still takes direction. Something has to point it at the right context, the right failure modes, the right constraints. What a practitioner gets from those conversations has to be retrieved from the record other practitioners left behind. Left on its own, a system may land on an idea that's sound in principle but just doesn't fit the problem as it actually is, given the deployment environment, the data limitations, the infrastructure already in place. That's the gap between a research implementation and one that survives production.

Upstream of that sits another gap, between stating a goal and having a specification complete enough to reach the intended result. That is the space this project explores.

Any LLM-based system working through a complex task needs the same thing, a specification that captures the full scope of the desired outcome. Producing one means decomposing an ambiguous problem into its core elements, matching what knowledge is needed against what's available, and reasoning critically across those dimensions. This project starts with a narrower focus, testing whether structure helps in one verifiable domain, on the premise that, in the context of ML engineering, the gap is concrete and easier to measure.

## Research question

Does directed retrieval and structured reasoning over practitioner knowledge
improve ML problem specification quality and downstream agent performance beyond
what unstructured knowledge retrieval achieves? Specification quality is judged
against a frozen rubric while downstream performance is measured against
MLE-bench baselines.

## Conditions

A 2 (retrieval: flat vs staged) × 3 (synthesis: none / freeform /
structured-critical) design — full rationale in
[docs/RESEARCH_DESIGN.md](docs/RESEARCH_DESIGN.md):

| Condition | Retrieval | Synthesis | Module |
|-----------|-----------|-----------|--------|
| A | none | none | published MLE-bench baseline, cited not run |
| B1 | flat, single query | none — raw context block | `pipeline/condition_b.py` |
| B2 | flat, single query | freeform, stance-free | `pipeline/condition_b.py` |
| C1 | staged, 4 directed queries | freeform, stance-free (pilot condition) | `pipeline/condition_c1.py` |
| C2 | staged, 4 directed queries | structured schemas + critical stance | `pipeline/condition_c2.py` |

Concretely, on `random-acts-of-pizza` (Haiku dev model — illustrative, not an
eval result): **B2** and **C1** inject freeform advice prose the agent may adopt
uncritically —

```
## Advisor notes / ### Feature Engineering Strategy
Linguistic features (high ROI): word/character counts, punctuation patterns ...
```

— while **C2** injects typed, confidence-rated assumption flags, each linked to
a mitigation:

```
[F0] outcome_measurement_gap  (confidence: high)
  The ground-truth label (successful pizza receipt) is not observable in the
  available signals ...
→ recommendation (addresses F0): calibration-focused model, post-hoc
  Platt/isotonic scaling.
```

Full four-way contrast (B1→B2→C1→C2) in
[RESEARCH_DESIGN.md, Illustrative output](docs/RESEARCH_DESIGN.md#illustrative-output).

## Pipeline — four stages with explicit intermediate outputs

1. **Understand** — actual goal, causal vs predictive framing, and constraints (data, compute, time, submission format)
2. **Surface** — available data signals, desired signals, relevant prior work
3. **Flag** — assumption violations: IID, exposure bias, outcome measurement gaps, attribution ambiguity, sequential dependencies, resource constraints
4. **Advise** — modeling approaches, tradeoffs, known failure modes

Pipeline output is injected as initial context to the AIDE scaffold before the
MLE-bench competition begins. Pipeline runs *outside* the MLE-bench container
to comply with the rule against external LLM API calls during competition
execution.

## Corpus and retrieval

All corpus access goes through two seams in `pipeline/retriever.py`, both
enforcing leave-one-out, over a local ChromaDB store (`data/chroma/`,
gitignored) embedded with Voyage `voyage-4-large`:

- `retrieve()` for flat top-k, used by the parse stage and by Condition B's
  single flat pass.
- `retrieve_with_topup()` for the staged conditions, where each directed stage
  contributes `STAGE_N` *distinct* summaries. Repeats across stages are retained
  as an importance signal and each one is backfilled with the next-best unseen
  summary.

The retrieval unit is the notebook **summary**, held constant across conditions
so the flat-vs-staged comparison isolates query structure rather than retrieval
granularity. Budgets are parity-matched in `pipeline/config.py`, so B and C
reason over the same number of distinct documents.

| Collection | Contents | Queried by |
|---|---|---|
| `competition_metadata` | Code4ML `competitions.csv` plus mle-bench `description.md` (Lite-22), 1,005 chunks across 947 competitions | parse (C1/C2), B flat |
| `notebook_summaries` | one LLM abstract per unique notebook, the retrieval unit for all practitioner-knowledge access | B flat pass; C1/C2 directed stages |

Every retrieval carries `{"competition_id": {"$ne": current_competition_id}}`,
so the pipeline can never see the evaluated competition's own artifacts. The
full treatment of corpus construction, the 18-of-22 coverage asymmetry, and the
open similarity-threshold decision is in
[RESEARCH_DESIGN.md](docs/RESEARCH_DESIGN.md#corpus-construction).

Build the corpus (dev subset, Lite-22 only):

```bash
python -m ingest.download           # Code4ML CSVs (~1.4 GB) + mle-bench descriptions
python -m ingest.ingest_metadata    # → competition_metadata collection
python -m ingest.ingest_summaries   # → notebook_summaries (one LLM abstract per notebook; resumable)
```

The eval corpus widens the notebook slice past Lite-22 and keeps only notebooks
carrying a positive `kaggle_score` — the available evidence that a notebook
produced a real submission:

```bash
python -m ingest.ingest_summaries --scope full --scored-only --batch --rebuild
```

`--rebuild` drops and rebuilds a collection, which is required after an
embedding-model or summary-prompt change since skip-existing resumability would
otherwise leave stale records in place. `--batch` uses the Anthropic Message
Batches API (50% discount, async, resumes after interruption), which the
full-scope run needs at its size. Sizing and cost for both slices are in
[COST_ESTIMATE.md](docs/COST_ESTIMATE.md).

## Related work

- **MLE-bench** (OpenAI, ICLR 2025) — evaluation infrastructure. <https://github.com/openai/mle-bench>
- **AssistedDS** (EMNLP 2025) — baseline condition; finding: LLMs uncritically adopt unstructured knowledge.
- **CatDB** (VLDB 2025) — closest existing analog; assumes populated data catalog, doesn't address unknown signals.
- **DS-Agent** (Guo et al., ICML 2024) — closest prior art; CBR over retrieved Kaggle solutions, iteratively revised against execution feedback (Prelude builds its spec once, upfront).
- **MLE-Dojo** (Qiang et al., 2025) — Gym-style benchmark/training environment over 200+ Kaggle competitions; scope contrast, not competing (doesn't study the agent's starting specification).
- **RE-Bench** (METR, 2025) — frontier AI R&D evaluation against human experts; source of the long-horizon failure mode this project targets (agents entrench incorrect assumptions), and independent corroboration that automated grading structurally limits how representative any such benchmark can be. <https://arxiv.org/abs/2411.15114>
- **Yang et al. 2023**, "LLMs as Optimizers" — conceptual foundation.
- **Co-Scientist** (Gottweis et al., *Nature*, 2026; Google DeepMind) — independent convergence in an adjacent domain, not prior art; a shared premise that problem formation deserves a structured phase, not a shared architecture.

Full positioning, including what each work does and does not address, is in
[RESEARCH_DESIGN.md](docs/RESEARCH_DESIGN.md#related-work-positioning).

## Stack

- LangGraph — pipeline orchestration
- LangSmith — observability and evaluation
- Anthropic API — Sonnet for evaluation runs, Haiku for development iteration
- ChromaDB — local persistent vector store (two queried collections, cosine)
- Voyage AI — `voyage-4-large` embeddings for ingestion and queries
- Ollama — optional local backend for free, offline wiring smoke tests (never eval runs)
- AIDE — MLE-bench execution layer
- Python 3.12 via `uv`

## Architecture

Three networked containers in cloud evaluation:

```
┌────────────────┐    spec    ┌─────────────────┐
│ spec pipeline  │ ─────────► │  AIDE scaffold  │
│ (this repo)    │  output    │                 │
└────────────────┘            └────────┬────────┘
                                       │
                                       ▼
                              ┌─────────────────┐
                              │ mle-bench env   │
                              │ (no LLM access) │
                              └─────────────────┘
```

- **Local dev** — M4 MacBook Air 32GB. Pipeline development only, no training runs.
- **Cloud eval** — Lambda Cloud A10 (24GB), ~\$1.29/hr.
- **Analysis** — local (M4 MacBook Air); results pulled from the cloud box.

## Setup

```bash
brew install uv
uv sync --extra dev     # exact versions from uv.lock
source .venv/bin/activate
cp .env.example .env    # then fill in: ANTHROPIC_API_KEY, VOYAGE_API_KEY,
                        # LANGSMITH_API_KEY, LANGSMITH_TRACING_V2, LANGSMITH_PROJECT
```

`.env` is loaded on import (`pipeline/env.py`) — no shell sourcing needed. A
variable already exported in the environment wins over the file.

Dependencies are declared in `pyproject.toml` and pinned in `uv.lock`. The
lockfile is committed so a corpus build or an evaluation run can be reproduced
against the exact package versions that produced it.

For cloud-box provisioning and end-to-end experiment execution (spec
builds → AIDE runs → grading → analysis), see
[docs/RUNBOOK.md](docs/RUNBOOK.md).

## Run the toy pipeline

Two-stage smoke pipeline (`understand -> advise`) used to validate the
LangGraph + Anthropic SDK + LangSmith wiring. Always runs against Anthropic
Haiku — not backend-configurable.

```bash
source .venv/bin/activate
python -m pipeline.toy
```

Traces appear in LangSmith under project `spec-pipeline-dev`.

## Run the full pipeline

The real four-stage chain (`parse_problem -> surface_signals ->
flag_assumptions -> advise_approach`, see [Pipeline](#pipeline--four-stages-with-explicit-intermediate-outputs)
above). Output is tailored for an ML-literate reader — an experienced
AI/ML engineer, not a general audience.

Requires the corpus to be built first (see [Corpus and retrieval](#corpus-and-retrieval)).
Pass `competition_id` (Kaggle slug) so leave-one-out retrieval excludes the
competition's own artifacts.

```bash
source .venv/bin/activate
python -c "
from pipeline.condition_c2 import run_c2
import json
print(json.dumps(run_c2(raw_problem='...', competition_id='spooky-author-identification'), indent=2))
"
```

Other conditions: `pipeline.condition_b.run_b1` / `run_b2` and
`pipeline.condition_c1.run_c1` take the same keyword arguments.

### Configurable backends

Backend is selected by `LLM_PROVIDER`, read by `pipeline/llm_client.py`.
Every node calls through this one module — swapping backends requires no
code changes.

| `LLM_PROVIDER` | Model source | Env vars | Use for |
|---|---|---|---|
| `anthropic` (default) | Anthropic API | `MODEL` (Haiku for dev, Sonnet for eval) | development iteration and evaluation runs |
| `ollama` | Local `ollama serve` | `OLLAMA_MODEL`, `OLLAMA_HOST` (default `http://localhost:11434`) | free, offline smoke-testing of pipeline wiring only — never eval runs |

```bash
# Anthropic (default) — requires ANTHROPIC_API_KEY, MODEL in .env; see snippet above

# Local Ollama — requires `ollama serve` running and the model pulled
ollama pull llama3.1:8b-instruct-q4_K_M
LLM_PROVIDER=ollama OLLAMA_MODEL=llama3.1:8b-instruct-q4_K_M python -c "
from pipeline.condition_c2 import run_c2 as run
import json
print(json.dumps(run(raw_problem='...'), indent=2))
"
```

Note: local open-source models are meaningfully weaker at this task than
Haiku/Sonnet (e.g. misclassifying causal framings as predictive) — use
Ollama to validate the graph runs end-to-end, not to judge output quality.

## Layout

```
pipeline/
├── config.py      # central tunables: budgets, collections, threshold (parity invariant)
├── state.py        # PipelineState TypedDict — the contract between stages
├── schemas.py       # ParsedProblem, SurfacedSignals, SpecificationFlag/AssumptionFlags,
│                     #   Recommendation/Advice — Pydantic stage contracts
├── nodes.py           # four C2 stage nodes + shared query builders
├── graph.py            # build_graph() — wires the four nodes
├── condition_c2.py      # run_c2(raw_problem, competition_id) — Condition C2 entry point
├── condition_c1.py       # run_c1() — staged retrieval + freeform synthesis (both artifacts)
├── condition_b.py          # run_b1()/run_b2() — flat retrieval conditions
├── llm_client.py           # call_llm() schema-constrained + call_llm_text() freeform
├── retriever.py             # retrieve() + retrieve_with_topup() — leave-one-out enforced here
├── embeddings.py             # embed() — voyage-4-large, single embedding seam
└── toy.py                     # two-stage smoke pipeline, Anthropic-only

ingest/      # offline corpus build: download, chunking, ingestion, corpus export
analysis/    # post-run: flag judge (frozen rubric) + blinded human anchor, paired stats,
             #   retrieval characterization
docs/        # design, decisions, rubric, data handling, runbook — see below
tests/       # pytest — unit + smoke; no real API calls (LLM + retrieval mocked)
notebooks/   # exploratory + analysis
data/        # raw downloads + ChromaDB store (gitignored)
results/     # per experiment stage: runs_{stage}.jsonl + {stage}/{run_key}/ (see docs/DATA.md)
```

## Documentation

- [RESEARCH_DESIGN.md](docs/RESEARCH_DESIGN.md) — the experiment
- [DECISIONS.md](docs/DECISIONS.md) — dated audit trail, append-only
- [JUDGE_RUBRIC.md](docs/JUDGE_RUBRIC.md) — per-flag judging criteria, **frozen, do not edit**
- [JUDGE_VALIDATION.md](docs/JUDGE_VALIDATION.md) — blinded human anchor on the judge
- [CORPUS_SAMPLES.md](docs/CORPUS_SAMPLES.md) — a sample of each document class
- [DATA.md](docs/DATA.md) — what is published, and the corpus fingerprint
- [RUNBOOK.md](docs/RUNBOOK.md) — two-machine operational workflow
- [COST_ESTIMATE.md](docs/COST_ESTIMATE.md) — corpus and eval spend, by slice
- [PROGRESS.md](docs/PROGRESS.md) — milestone history
- [CLAUDE.md](CLAUDE.md) — operating brief for working in the repo

## Tests

```bash
pytest tests/ -v
```

`tests/test_pipeline.py` covers the full pipeline: each node in isolation
plus an end-to-end run through the graph, asserting `stage_trace` order and
field population at each stage. `pipeline.nodes.call_llm` and
`pipeline.nodes.retrieve` are monkeypatched — no network calls, so this
passes with either backend configured. `tests/test_retriever.py` exercises
the leave-one-out filter, similarity threshold, and k against a temp
ChromaDB with fake embeddings; `tests/test_chunking.py` covers oversized
code-block splitting. `tests/test_api.py` is the one exception — it makes a
real Anthropic API call to confirm connectivity.

## License and acknowledgments

MIT — see [LICENSE](LICENSE). This work builds on external tools and datasets
(MLE-bench, AIDE / aideml, Code4ML, Voyage AI, and others), credited in
[ACKNOWLEDGMENTS.md](ACKNOWLEDGMENTS.md).
