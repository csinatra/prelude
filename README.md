# spec-pipeline

Structured LLM reasoning pipeline for ML problem specification. Research POC
for an Anthropic Fellows Program application.

## Research question

Does structured reasoning over retrieved organizational knowledge improve ML
problem specification quality and downstream agent performance — measured
against MLE-bench baselines — beyond what unstructured knowledge provision
achieves?

## Three-condition evaluation

| Condition | Description | Source |
|-----------|-------------|--------|
| A | MLE-bench baseline, no specification assistance | Published, cited not run |
| B | Unstructured knowledge provision | Replicates AssistedDS approach |
| C | Structured specification reasoning pipeline | Primary contribution |

## Pipeline — four stages with explicit intermediate outputs

1. **Understand** — actual goal, constraints, causal vs predictive framing, resource constraints
2. **Surface** — available data signals, desired signals, relevant prior work
3. **Flag** — assumption violations: IID, exposure bias, outcome measurement gaps, attribution ambiguity, sequential dependencies, resource constraints
4. **Advise** — modeling approaches, tradeoffs, known failure modes

Pipeline output is injected as initial context to the AIDE scaffold before the
MLE-bench competition begins. Pipeline runs *outside* the MLE-bench container
to comply with the rule against external LLM API calls during competition
execution.

## Prior art

- **MLE-bench** (OpenAI, ICLR 2025) — evaluation infrastructure. <https://github.com/openai/mle-bench>
- **AssistedDS** (EMNLP 2025) — baseline condition; finding: LLMs uncritically adopt unstructured knowledge.
- **CatDB** (VLDB 2025) — closest existing analog; assumes populated data catalog, doesn't address unknown signals.
- **Yang et al. 2023**, "LLMs as Optimizers" — conceptual foundation.

## Stack

- LangGraph — pipeline orchestration
- LangSmith — observability and evaluation
- Anthropic API — Sonnet for evaluation runs, Haiku for development iteration
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
- **Cloud eval** — AWS g5.xlarge (A10G), ~$1.50/hr.
- **Analysis** — Colab free tier.

## Setup

```bash
brew install uv
uv venv --python 3.12
source .venv/bin/activate
uv pip install langgraph langsmith anthropic python-dotenv pytest
# create .env with ANTHROPIC_API_KEY, LANGSMITH_API_KEY, LANGSMITH_TRACING_V2=true,
# LANGSMITH_PROJECT=spec-pipeline-dev
```

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

```bash
source .venv/bin/activate
python -c "
from dotenv import load_dotenv; load_dotenv()
from pipeline.runner import run
import json
print(json.dumps(run(raw_problem='...'), indent=2))
"
```

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
from pipeline.runner import run
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
├── state.py       # PipelineState TypedDict — the contract between stages
├── schemas.py      # ParsedProblem, SurfacedSignals, AssumptionFlags, Advice — Pydantic
│                    #   stage-output models, enforced via native structured outputs
├── nodes.py         # parse_problem, surface_signals, flag_assumptions, advise_approach
├── graph.py          # build_graph() — wires the four nodes
├── runner.py          # run(raw_problem) — thin entry point over the compiled graph
├── llm_client.py       # call_llm() — swappable Anthropic / Ollama backend, schema-constrained
└── toy.py               # two-stage smoke pipeline (understand -> advise), Anthropic-only

tests/       # pytest — unit + smoke; no real API calls (LLM calls are mocked)
notebooks/   # exploratory + analysis
results/     # eval outputs (gitignored)
```

## Tests

```bash
pytest tests/ -v
```

`tests/test_pipeline.py` covers the full pipeline: each node in isolation
plus an end-to-end run through the graph, asserting `stage_trace` order and
field population at each stage. `pipeline.nodes.call_llm` is monkeypatched
with canned JSON responses — no network calls, so this passes with either
backend configured. `tests/test_api.py` is the one exception — it makes a
real Anthropic API call to confirm connectivity.
