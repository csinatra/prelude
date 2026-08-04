# Prelude

Structured problem specification before agentic execution begins — a research
POC on LLM-assisted problem framing for ML engineering.

> **Status (2026-08-03):** pipeline architecture and experimental design
> complete (B1 / B2 / C1 / C2, summary-level staged retrieval, mechanistic-analysis
> scaffold). AIDE injection path confirmed end-to-end via smoke run.
> **Evaluation runs not yet executed.**

## Motivation

Before an agent writes a line of code, a human expert does the part that's hardest to automate: they frame the problem. As agentic systems take on more of the work between a stated goal and a finished result, more of what determines success sits upstream of execution. Translating an ambiguous goal into something verifiable is a critical first step to building any solution, starting with the basic question of what the core problem actually is and what would count as a good answer.

Problem framing draws on things that don't show up cleanly in a task description. Things like institutional precedent, awareness of failure modes and implicit assumptions, judgment about what a metric actually needs to capture, and constraints that live outside the immediate problem statement. People with domain expertise supply this instinctively. Often that means rounds of conversation with collaborators and senior colleagues, testing an idea against people who have seen adjacent problems, before ever committing to a direction. An agentic system has significant knowledge baked into its parameters, but applying that knowledge to a specific problem still takes direction. Something has to point it at the right context, the right failure modes, the right constraints. Left on its own, a system may land on an idea that's sound in principle but just doesn't fit the actual constraints of the problem, the deployment environment, the data limitations, the infrastructure already in place. That's the gap between a research implementation and one that survives production.

That gap, between stating a goal and understanding a problem well enough to act on it, is the space this project explores. The underlying capabilities apply more broadly to any LLM-based system working through a complex task. That means decomposing an ambiguous problem into its core elements, understanding what knowledge is actually needed against what's available, and reasoning critically across all of that to arrive at a solution. This project starts with a narrower focus, testing whether structure helps in one measurable domain, on the premise that in the context of ML engineering, the gap is concrete and easier to see.

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
gitignored) embedded with Voyage `voyage-code-3`:

- `retrieve()` — flat top-k retrieval. Used for `competition_metadata` (parse
  stage) and, over `notebook_summaries`, for Condition B's single flat pass.
- `retrieve_with_topup()` — the staged practitioner-knowledge unit for C1/C2:
  each directed stage retrieves top-`STAGE_N` notebook summaries, retaining any
  a prior stage already surfaced (re-selection is an importance signal) and
  backfilling each such repeat with the next-best unseen summary, so every stage
  contributes `STAGE_N` *distinct* documents. The retrieval unit — the notebook
  **summary** — is held constant across conditions, so the flat-vs-staged
  comparison isolates query structure, not retrieval granularity.

Budgets are parity-matched knobs in `pipeline/config.py`: B and C reason over the
same number of **distinct** notebook summaries (`BASELINE_N_NOTEBOOKS` =
3 × `STAGE_N_NOTEBOOKS`).

Two collections:

| Collection | Contents | Queried by |
|---|---|---|
| `competition_metadata` | Code4ML `competitions.csv` + mle-bench `description.md` (Lite-22) — 1,005 chunks across 947 competitions | parse (C1/C2), B flat |
| `notebook_summaries` | one LLM abstract per unique notebook — the retrieval unit for all practitioner-knowledge access | B flat pass; C1/C2 directed stages |

**Leave-one-out:** every retrieval carries
`{"competition_id": {"$ne": current_competition_id}}`, enforced inside
`pipeline/retriever.py` — the pipeline can never see the evaluated
competition's own artifacts. Code4ML covers 18 of the Lite-22; the other 4
have descriptions only (from mle-bench). Under leave-one-out this doesn't
change eval validity — every competition retrieves only from *other*
competitions — but the 18 covered ones are the test cases where the filter
does real work.

**Note on sources:** the implementation brief's Source 1 (MLEModernizer,
zenodo 15022707) ships as a single 107 GB tar.gz and is deferred to the cloud
box. Code4ML's code-block CSVs (~1.4 GB) fill the practitioner-knowledge role
for the dev corpus.

Build the corpus (dev subset — Lite-22 code blocks only):

```bash
python -m ingest.download           # Code4ML CSVs (~1.4 GB) + mle-bench descriptions
python -m ingest.ingest_metadata    # → competition_metadata collection
python -m ingest.ingest_notebooks   # → practitioner_knowledge collection
python -m ingest.ingest_summaries   # → notebook_summaries (one LLM abstract per notebook; resumable)
```

The similarity threshold (`SIMILARITY_THRESHOLD` env var) is deliberately
unset — to be calibrated against the real corpus on 5–10 dev competitions
before eval runs.

## Related work

- **MLE-bench** (OpenAI, ICLR 2025) — evaluation infrastructure. <https://github.com/openai/mle-bench>
- **AssistedDS** (EMNLP 2025) — baseline condition; finding: LLMs uncritically adopt unstructured knowledge.
- **CatDB** (VLDB 2025) — closest existing analog; assumes populated data catalog, doesn't address unknown signals.
- **DS-Agent** (Guo et al., ICML 2024) — closest prior art; CBR over retrieved Kaggle solutions, iteratively revised against execution feedback (Prelude builds its spec once, upfront).
- **MLE-Dojo** (Qiang et al., 2025) — Gym-style benchmark/training environment over 200+ Kaggle competitions; scope contrast, not competing (doesn't study the agent's starting specification).
- **Yang et al. 2023**, "LLMs as Optimizers" — conceptual foundation.

- **Co-Scientist** (Gottweis et al., *Nature*, 2026; Google DeepMind) — *independent convergence, adjacent domain (not prior art).* A multi-agent system for scientific hypothesis generation, validated in wet-lab work, that independently reflects Prelude's core premise: problem understanding and hypothesis formation deserve a structured phase before the solution phase. A shared premise, not a shared architecture.

## Stack

- LangGraph — pipeline orchestration
- LangSmith — observability and evaluation
- Anthropic API — Sonnet for evaluation runs, Haiku for development iteration
- ChromaDB — local persistent vector store (three collections, cosine)
- Voyage AI — `voyage-code-3` embeddings for ingestion and queries
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
uv venv --python 3.12
source .venv/bin/activate
uv pip install langgraph langsmith anthropic python-dotenv pytest requests chromadb voyageai pandas
cp .env.example .env   # then fill in: ANTHROPIC_API_KEY, VOYAGE_API_KEY,
                       # LANGSMITH_API_KEY, LANGSMITH_TRACING_V2, LANGSMITH_PROJECT
```

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
from dotenv import load_dotenv; load_dotenv()
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
├── embeddings.py             # embed() — voyage-code-3, single embedding seam
└── toy.py                     # two-stage smoke pipeline, Anthropic-only

ingest/      # offline corpus build: download, chunking, three ingestion scripts
analysis/    # post-run scaffold: flag judge (frozen rubric), artifact preservation
docs/        # RESEARCH_DESIGN.md, JUDGE_RUBRIC.md (frozen), COST_ESTIMATE.md
tests/       # pytest — unit + smoke; no real API calls (LLM + retrieval mocked)
notebooks/   # exploratory + analysis
data/        # raw downloads + ChromaDB store (gitignored)
results/     # run artifacts keyed {competition}_{condition}_{seed} (gitignored during dev)
```

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
