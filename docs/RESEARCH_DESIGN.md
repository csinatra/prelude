# Prelude — Research Design

*v1 scope. Drafted July 2026, before any evaluation run. Amendments after
runs begin must be dated and listed at the bottom.*

## Research question and hypothesis

**Question:** Does structured reasoning over retrieved prior work improve ML
agent performance on benchmark tasks beyond what unstructured knowledge
provision achieves?

**Falsifiable hypothesis (H1):** On MLE-bench Lite competitions, an AIDE
agent given Condition C2's structured specification will achieve a higher
Any-Medal rate than the same agent given Condition B's unstructured context.
H1 is falsified if C2 fails to separate from B beyond seed noise, or if
unstructured provision alone matches structured specification. *(Amended
2026-07-16, pre-run: H1 originally referenced the published no-assistance
baseline (A); that comparison is invalid because the published AIDE runs
used gpt-4o-2024-08-06, not our agent model — see the Condition A note
under the experimental design.)*

**Secondary hypothesis (H2, mechanistic):** C2's advantage, if any, is
mediated by specification flags being acted on — flag categories with higher
action rates (per the frozen judge rubric) contribute disproportionately to
outcome differences.

*Corroborating prior evidence (suggestive, not validating):* DS-Agent's
development-stage ablation found retrieval-augmented CBR beat its
no-retrieval variant, and iterative case revision beat one-shot CBR — best
rank 2.08 (CBR with feedback revision) vs 2.58 (CBR without) vs 3.41 (no
retrieval), across 12 tasks, their Table 2. This is consistent with H1's
basic premise that retrieved practitioner knowledge helps. It is only
suggestive, not validation: DS-Agent works by a different mechanism
(iterative case revision against execution feedback, vs Prelude's one-shot
upfront spec) and on a much smaller, easier task set than MLE-bench Lite.

## Related work positioning

- **MLE-bench** (OpenAI, ICLR 2025): evaluation infrastructure and the
  Condition A baseline. Measures agent capability; does not study what
  knowledge or specification the agent starts with.
- **AssistedDS** (EMNLP 2025): showed LLMs adopt provided unstructured
  knowledge uncritically. Defines Condition B's design; does not test
  structured alternatives.
- **CatDB** (VLDB 2025): closest analog — catalog-grounded data-science
  automation. Assumes a populated catalog; does not address reasoning about
  unknown/missing signals, and provides no staged specification process.
- **DS-Agent** (Guo et al., ICML 2024): case-based reasoning framework that
  retrieves whole solved Kaggle cases and iteratively revises them against
  execution feedback. Closer prior art than AssistedDS/CatDB — directly
  retrieves practitioner Kaggle solutions, similar to Prelude's corpus. Key
  difference: DS-Agent's retrieval is revised iteratively against execution
  feedback in a CBR loop; Prelude's spec is built once, upfront, before the
  agent's own search begins.
- **MLE-Dojo** (Qiang et al., May 2025): interactive Gym-style benchmark
  environment built on 200+ Kaggle competitions, supporting SFT/RL agent
  training. Different axis from Prelude — an alternative/broader execution
  environment and training framework, not a study of what specification or
  knowledge an agent starts with. Cite for scope contrast, not as directly
  competing work.
- **Yang et al. 2023, "LLMs as Optimizers":** conceptual grounding for
  prompt-level structured reasoning; not applied to ML problem
  specification.
- **Co-Scientist** (Gottweis, Natarajan, et al., *Nature*, 2026; Google
  DeepMind): *independent convergence in an adjacent domain, not prior art.*
  A multi-agent system that generates and refines scientific hypotheses for
  researchers to test, validated in wet-lab collaborations. It reflects the
  same core conviction as Prelude, reached independently: that problem
  understanding and hypothesis formation deserve a structured phase before the
  solution phase. A shared premise, not a shared architecture — cited as a
  signal the premise is an active frontier question, not as transferable
  technique.

## System overview

```mermaid
flowchart LR
    subgraph corpus["Offline corpus build (ingest/)"]
        c4ml["Code4ML CSVs"] --> pk[("practitioner_knowledge<br/>62k code chunks")]
        c4ml --> ns[("notebook_summaries<br/>5,937 pinned-Haiku abstracts")]
        descs["mle-bench description.md"] --> cm[("competition_metadata")]
    end

    subgraph specbuild["Spec build (dev machine, Anthropic API)"]
        desc["competition description"] --> cond["condition pipeline<br/>B1 / B2 / C1 / C2"]
        pk -.->|"leave-one-out<br/>retrieval"| cond
        ns -.-> cond
        cm -.-> cond
        cond --> spec["spec.md + run artifacts<br/>runs.jsonl: spec_built"]
    end

    subgraph cloudbox["Cloud box (GPU; no spec-pipeline LLM calls)"]
        spec -->|"mounted at<br/>/home/spec/spec.md"| aide["aide-prelude agent<br/>(MLE-bench container)"]
        aide --> sub["submission.csv +<br/>AIDE journal"]
        sub --> grade["mlebench grade"]
        grade --> reg["runs.jsonl:<br/>agent_run → graded"]
    end

    reg --> anal["analysis/:<br/>judging, trajectories, plots"]
```

The spec pipeline (left, laptop) and agent execution (right, cloud box)
never share a runtime: specs are serialized text by the time they reach the
container (core constraint 1). The append-only registry is the join —
spec-time entries from the dev machine, agent/grading entries from the box,
merged per run_key.

## Experimental design

Five conditions, structured as a 2 (retrieval: flat vs staged) × 3
(synthesis: none vs freeform vs structured-critical) grid with three cells
built plus one grid-external anchor:

| Condition | Retrieval | Synthesis | Isolates (vs) |
|---|---|---|---|
| A | none | none | no-assistance anchor — matched agent, contingent arm (see note) |
| B1 | flat, single query | none — raw context block | knowledge provision per se (vs A) |
| B2 | flat, single query | freeform, stance-free | LLM preprocessing (vs B1) |
| C1 | staged, 4 directed queries | freeform, stance-free (B2's path) | staged retrieval (vs B2) |
| C2 | staged, 4 directed queries | structured schemas + critical-integration stance | synthesis structure (vs C1) |

Per-condition workflow — every arrow into a spec is one grid step's single
change:

```mermaid
flowchart TD
    D["competition description"] --> FQ["two-level retrieval, flat<br/>(single query = full description)"]
    FQ --> CB["context block<br/>(notebook cards + metadata)"]
    CB --> B1["B1 spec:<br/>context block"]
    CB --> FS1["freeform synthesis<br/>(stance-free)"]
    FS1 --> B2["B2 spec:<br/>block + advice"]

    D --> P["parse stage<br/>(structured extraction)"]
    P --> Q["3 directed queries<br/>(surface / flag / advise)"]
    Q --> SR["two-level retrieval, staged<br/>(same notebook→chunk unit and<br/>total budget as flat; directed queries)"]
    SR --> SCB["staged context block<br/>(deduped across stages)"]
    SCB --> FS2["freeform synthesis<br/>(B2's path, stance-free)"]
    FS2 --> C1["C1 spec:<br/>staged block + advice"]
    SR --> ST["surface → flag → advise<br/>(structured stages, RETRIEVAL_STANCE)"]
    ST --> C2["C2 spec:<br/>staged block + framing,<br/>signals, flags, recommendations"]

    A["Condition A (contingent):<br/>no spec mounted = stock AIDE"]
```

### Staged pipeline — queries, prompts, schemas

C decomposes spec-building into four stages. **parse** extracts the problem's ML
structure (`task_type`, `evaluation_metric`, `goal`) from the raw description —
this runs in **both** C1 and C2, since the directed queries must be built from
something. **surface**, **flag**, and **advise** then each retrieve with a query
built from those extracted fields (Table 1). C1 and C2 run this identical staged
retrieval; they differ only in synthesis (Table 2): **C2** keeps every stage's
output as a typed schema, while **C1** keeps none of it — it pools the four
retrieved blocks into one freeform pass (B2's style: no schema, no stance), over
staged rather than flat retrieval. B retrieves everything with one flat
`raw_problem` query and, in B2, one freeform pass.

**Table 1 — staged retrieval** (run identically by C1 and C2). Query text is the
verbatim string embedded for similarity search — `flag` and `advise` prepend a
fixed phrase to the `{task_type} {evaluation_metric} {goal}` fields `parse`
extracted; `parse` queries with the raw description. Sample hit is the top
retrieval for `random-acts-of-pizza` (other competitions only, leave-one-out;
cosine similarity in parens) — the directed queries pull different, on-target code.

| Stage | Query text | Query target | Sample hit |
|---|---|---|---|
| **parse** | the raw competition description | descriptions of similar past competitions | `ml1819-whats-cooking?` (0.65) — a competition description |
| **surface** | `{task_type} {evaluation_metric} {goal}` | data signals and prior work practitioners used | `jigsaw-toxic-comment` (0.55) — `roc_curve` / `roc_auc_score` evaluation code |
| **flag** | `validation leakage overfitting pitfalls {task_type} {evaluation_metric} {goal}` | validation / leakage / overfitting patterns | `siim-isic-melanoma` (0.58) — leaderboard-probing (`known positive set to 1, rest 0`) |
| **advise** | `model architecture training approach {task_type} {evaluation_metric} {goal}` | modeling approaches for the metric | `jigsaw-toxic-comment` (0.64) — a Keras text-model pipeline |

**Table 2 — C2 structured synthesis** (C2 only). Each stage is a
schema-constrained LLM call whose output is preserved in the spec (sample output
from the random-acts run). B2 and C1 produce **none** of this per-stage
structure — they collapse the retrieval into a single freeform prose pass (the
format contrast is visible under Illustrative output); C1 uses `parse`'s
extraction only to build the Table 1 queries.

| Stage | Prompt produces | Schema | Sample output |
|---|---|---|---|
| **parse** | goal, task type, evaluation metric, target variable, framing (causal / predictive / descriptive / ambiguous), constraints | `ParsedProblem` | task = binary classification; metric = ROC-AUC; target = `requester_received_pizza`; framing = predictive |
| **surface** | available signals; signals that would help but are missing; relevant prior work | `SurfacedSignals` | available: `request_text_content`, `request_text_length`; missing: `user_previous_success_rate` |
| **flag** | assumption violations, each a typed `category` + `confidence` + `evidence_doc_ids` | `AssumptionFlags` (list of `SpecificationFlag`) | `[F0] outcome_measurement_gap` (high) — success label not observable in the available signals |
| **advise** | concrete approaches, each a `tradeoff` + `failure_mode` + `addresses_flags` | `Advice` (list of `Recommendation`) | calibration-focused Bi-GRU/XGBoost + post-hoc calibration — addresses F0 |

Two prompt details worth making explicit:

- **Shared critical stance, C2 only.** Every C2 stage prompt appends
  `RETRIEVAL_STANCE`: retrieved excerpts are *"evidence of past practice, not a
  boundary on your reasoning … explicitly disregard excerpts that are
  irrelevant, outdated, or low quality."* B2 and C1 synthesis is deliberately
  stance-free — uncritical adoption is the AssistedDS failure mode those
  conditions must be free to exhibit (CLAUDE.md constraint 6).
- **Honest grounding, not forced citation.** The flag prompt instructs: *"cite a
  document only if it genuinely shaped the flag; a flag drawn from your general
  knowledge should honestly report an empty `evidence_doc_ids` list — that is a
  valid and expected answer, not a failure."* So an empty `evidence_doc_ids` is
  a truthful "this came from model priors," not a defect — read the
  retrieval-grounded fraction with that in mind.

### Illustrative output

The four injected artifacts on one competition, showing the additive design —
each step changes exactly one variable. *Built with the Haiku **dev** model on
`random-acts-of-pizza` (the off-eval smoke competition); illustrative only, not
an eval result — eval runs use the pinned Sonnet `EVAL_MODEL`. Excerpts are
truncated/reflowed for readability. Document budget (count of retrieved docs) is
parity-matched across all four; what varies is how they are retrieved and what
synthesis sits on top.*

**B1** — the raw retrieved block, verbatim, nothing on top:

```
[code4ml_homework-for-students_0 | competition_description]
'Feature Engineering  Build Models  Submission  Notebooks  csv  Kernel ...'
  ... top-k practitioner chunks + competition descriptions, flat single query
```

**B2** ( = B1 + synthesis) — same flat block plus a stance-free advice pass.
Prose the agent may adopt uncritically — the AssistedDS failure mode B exists to
exhibit:

```
## Advisor notes
### Feature Engineering Strategy
Linguistic features (high ROI): word/character counts, punctuation patterns,
capitalization stats, sentiment indicators, readability metrics ...
```

**C1** ( = B2 with staged retrieval) — same freeform advice format, but the
block is now pulled by four *directed* queries instead of one flat query.
Isolates the effect of retrieval structure:

```
## Advisor notes
## 1. Problem Understanding
This is a binary classification problem ... The key insight: this isn't purely
NLP — you have rich metadata that correlates with altruistic behavior.
```

**C2** ( = C1 with structured synthesis) — staged retrieval, but the freeform
prose is replaced by typed, confidence-rated assumption flags, each linked to a
concrete recommendation. Isolates the effect of output structure:

```
[F0] outcome_measurement_gap   (confidence: high)
  The ground-truth label (successful pizza receipt) is not observable in the
  available signals — only request metadata and user history exist.

→ recommendation (addresses F0): calibration-focused model — Bi-GRU/XGBoost,
  then post-hoc Platt/isotonic scaling to output calibrated probabilities.
```

The B2→C1 step changes retrieval (flat → directed); the C1→C2 step changes
synthesis (freeform prose → structured flags + linked mitigations). C2's flags
are categorized and confidence-rated, and recommendations cite the `flag_id`s
they address — the structure whose downstream effect this POC measures.

Design notes:

- **Condition A (decided 2026-07-16, pre-run):** the published MLE-bench
  AIDE baseline used `gpt-4o-2024-08-06` as the code model and is not
  model-matched to our runs — it is cited as context only and never
  compared statistically. A *matched* A exists for free in the harness:
  `aide-prelude` with no spec mounted is byte-identical to stock AIDE
  (same agent model, hardware, time/step budgets, mle-bench version).
  It is pre-registered as a **contingent arm**, not a primary condition:
  the full A arm (~30 runs, est. +\$200–250) — unmounted `aide-prelude` on
  the eval competitions at the same model/hardware/budget as B/C — triggers
  only if C2 fails to separate from B, or any condition lands below the
  plausible no-assistance range, the outcomes under which "is retrieval
  beneficial at all, or detrimental?" becomes load-bearing for
  interpretation. Infrastructure smoke runs (RUNBOOK step 4) are decoupled
  from this: they use the 8-step `dev` variant on a held-out off-eval
  competition and are throwaway integration checks, **not** matched-A data
  points — the dev budget is too short to be a valid A run. The core
  research question (structured decomposition + directed retrieval vs naive
  provision) is carried by the B/C contrasts and does not require A.
- **Retrieval unit held constant.** All conditions receive practitioner
  knowledge as "notebook cards" (top-M chunks from each of N notebooks
  surfaced via a summaries index) plus flat competition-metadata chunks.
  B selects notebooks with one flat query; C1/C2 with four stage-directed
  queries. This confines the flat-vs-staged contrast to query structure
  rather than retrieval granularity.
- **Document budgets parity-matched** via `pipeline/config.py` (B's flat
  budget = staged conditions' total). Exact values are calibration
  parameters, not design constants.
- **C1 holds synthesis at B2's level.** Its staged retrieval feeds a single
  freeform, stance-free pass (no schema, no `RETRIEVAL_STANCE`), so the only
  change from B2 is flat → staged retrieval — see the staged-pipeline tables
  above.
- **C1 is a pilot condition:** 3–4 competitions, single seed, not the full
  matrix. Built to run at full scale; the harness defaults to the pilot
  subset.
- AIDE scaffold, agent model, and MLE-bench grading are held constant across
  all run conditions (A/B/C). This constancy is what makes the matched-A
  baseline valid and the B-vs-C contrast clean: the only manipulated variable
  across the grid is the injected spec. **Condition A guardrail:** A must use
  this same pinned agent (the shared `aide-prelude` image, no spec mounted) —
  never re-matched to a published external baseline, which is the model
  mismatch the H1 amendment corrected.
- **Two model roles, pinned separately (a deliberate advisor/executor
  split).** The *spec pipeline* — C's staged retrieval, decomposition, and
  opinionated recommendations — runs on `EVAL_MODEL = claude-sonnet-5`
  (`pipeline/config.py`, pinned 2026-07-19), a strong reasoner; its output is
  identical whether the downstream agent is weak or strong. The *AIDE agent*
  that consumes the injected spec runs on `claude-haiku-4-5-20251001`
  (resolved 2026-07-22). Using different models for "author the specification"
  and "implement against it" is a standard, realistic architecture, not a
  compromise — and both models are held constant across A/B/C, so the split
  cannot confound the B-vs-C comparison. The agent-model choice therefore
  bears **only on the secondary (downstream MLE-bench) signal**, never on the
  primary mechanistic spec-quality judging, which scores the specs directly
  and is agent-independent.
- **Why Haiku for the agent (2026-07-22).** The pinned agent is
  `thesofakillers/aideml@v6.3.3` (mle-bench's designated fork, 2024), which
  cannot drive the current top models unmodified: the 5-family and Opus
  4.7/4.8 removed the `temperature` parameter aideml always sends (HTTP 400),
  and Sonnet 5 defaults adaptive thinking ON when `thinking` is omitted, so
  its responses carry a thinking block that trips aideml's single-text-block
  assertion. Haiku 4.5 (prior generation) accepts `temperature` and returns a
  single text block, so aideml's code path runs unmodified. Two non-algorithmic
  fixes are applied (verified end-to-end on-box 2026-07-23): an httpx<0.28
  environment pin, and a completion of aideml's Anthropic backend — upstream
  left the function-calling (`func_spec`) path as `NotImplementedError` (which
  is why mle-bench's own aide/claude pairs an Anthropic code model with a gpt-4o
  feedback model), so the AIDE review step is implemented via Anthropic tool use
  to reach parity with aideml's OpenAI backend, keeping the run single-provider
  on Claude. Haiku keeps the strongest fidelity story and the lowest per-run cost.
- **Agent capability × structure is a generalizability caveat, not a
  confound.** The downstream C-vs-B gap is measured at Haiku's capability
  point; whether that gap widens or narrows with a stronger agent is
  theoretically ambiguous (structure-substitution at the strong end vs
  execution-gating at the weak end) and is **future work** — a capability
  sweep (Haiku → Sonnet-5-thinking-off → Sonnet-5-thinking-on) directly probes
  it. The Sonnet-5 path is a bounded one-switch adapter (`thinking: disabled`
  + drop `temperature`), reserved for that sweep, not the default arm.

## Corpus construction

- **Current dev corpus:** Code4ML filtered to MLE-bench Lite-22 competitions
  — 62,379 code chunks (deduped from 132,796 cell-level blocks), 1,005
  competition-metadata chunks (1,156 Code4ML descriptions + 22 mle-bench
  `description.md`), plus a `notebook_summaries` collection (one abstract
  per unique notebook) as level one of two-level retrieval. Summaries are
  corpus infrastructure, generated by a pinned model (Haiku,
  `summary_model` stamped in metadata) shared identically across all
  conditions — exempt from the Haiku-dev/Sonnet-eval switch because summary
  quality affects retrieval uniformly and cannot confound between-condition
  comparisons.
- **Planned before production runs:** expansion to full Code4ML (~2.74M
  blocks, ~1,150 competitions; the Lite slice is ~5%) and MLEModernizer
  (107 GB replication package — cloud-box ingestion, separate swap).
- **Leave-one-out:** every retrieval carries
  `competition_id != current_competition`, enforced inside the retrieval
  seam (`pipeline/retriever.py`); for two-level retrieval the filter applies
  at the notebook level, which subsumes the chunk level. A code path
  bypassing this filter is solution leakage (CLAUDE.md core constraint 5).
- **Two-level rationale:** flat chunk retrieval returns decontextualized
  fragments from arbitrary notebooks; notebook-then-chunk retrieval returns
  coherent excerpt groups from notebooks selected as wholes, and lets
  retrieval reason at the level practitioners work at (a notebook is an
  approach; a chunk is a step).

## Outcome metrics (defined in advance)

**Primary:** Any-Medal rate (MLE-bench grading), mean ± one standard error
across 3 seeds per competition. Acknowledged as low-powered at n≈3–5
competitions (POC subset); treated as directional, not confirmatory — under
the POC framing (see Roadmap) the mechanistic spec-judging carries the
primary evidential weight and the MLE-bench delta is a conservative lower
bound.

**Secondary (higher resolution):**
1. Leaderboard percentile of the final submission
2. Valid-submission rate (fraction of runs producing a gradeable submission)
3. Time-to-first-valid-submission (wall-clock within the AIDE run)

**Cost/efficiency accounting (added 2026-07-17, pre-run):** every run
carries a two-sided ledger linking upfront specification cost to downstream
agent behavior, answering whether C's additional spec-build LLM calls save
agent cycles relative to B:

- *Spec side* (registry + `llm_usage.json` per run artifact): build
  wall-clock, LLM call count, input/output tokens — per call, in stage
  order, so per-stage attribution is free.
- *Agent side* (registry via `harness.advance`, journal preserved as the
  trajectory artifact): run wall-clock, AIDE steps used,
  time-to-first-valid-submission, and per-step score/time curves derived
  from the AIDE journal — enabling trajectory comparison across conditions
  and problem types. Per-step agent *token* usage is not in AIDE's journal
  by default; the aide-prelude Anthropic backend appends each call's usage
  (in/out/cache tokens + timestamps) to `prelude_token_usage.jsonl`, a
  behavior-neutral side-channel identical across all conditions —
  correlated to journal node ctimes offline for per-step attribution.

All agent-side artifacts (submission, journal, final solution code, token
log) are copied off the ephemeral mle-bench run dir onto the persistent
volume (`analysis.artifacts.preserve_agent_outputs`, called by the batch
driver) so `--terminate-on-done` never destroys the mechanistic-judge
inputs — the registry alone keeps only outcome fields.

## Mechanistic evaluation

Per-flag flag→action→outcome judging against the **frozen rubric**
(`docs/JUDGE_RUBRIC.md`; frozen before any run, amendment-controlled):
each `SpecificationFlag` from a C2 run is classified `not_acted_on` /
`acted_on_unclear` / `acted_on_positive` by an LLM judge
(`analysis/judge.py`) that sees the solution artifacts but never the score.
Aggregation per category: detection rate, action rate, outcome
contribution, and retrieval-grounded fraction (non-empty
`evidence_doc_ids`). Requires the artifact preservation layout in
`analysis/artifacts.py`.

## Threats to validity

- **Construct validity (acknowledged, central):** Kaggle competitions are
  *pre-specified by construction* — task, metric, and data are given, which
  limits the available specification-failure signal that Prelude is designed
  to catch in production settings. Mitigation: competition selection favors
  Lite competitions with known data quirks (leakage paths, temporal
  structure, measurement gaps). A synthetic-corruption arm (deliberately
  under-specified task descriptions) is named as future work, not current
  scope. Consequently, any effect measured here is interpreted as a
  conservative lower bound on the expected improvement in the
  ambiguously-specified real-world settings Prelude targets: the framing
  portion of the C2 specification largely restates what Kaggle already
  makes explicit, so the measurable signal is confined to flags and
  recommendations.
- **Budget/attention confounds:** document budgets are parity-matched; token
  counts of injected artifacts are logged per run and reported, split into
  retrieved-block vs synthesis-artifact tokens so "more retrieved knowledge"
  is distinguishable from "structured restatement of the same knowledge."
  *Similarity threshold evaluated and rejected pre-run (2026-07-15):* a
  10-competition calibration sweep (`analysis/calibration.py`,
  `results/calibration/`) showed the stage-directed queries score ~0.06
  lower than the flat query against the identical chunk collection (short
  keyword queries vs full-description queries — query impoverishment, not a
  text/code modality gap), so any global cutoff filters the staged
  conditions roughly twice as hard as flat retrieval and breaks knowledge
  parity. `SIMILARITY_THRESHOLD=None` for v1: budgets bound quantity,
  top-k rank ordering supplies quality control. Revisit triggers: evidence
  of junk retrievals in run inspection (then: tail-relevance judging, and
  per-kind quantile floors derived by a uniform documented rule), or corpus
  expansion (re-run the sweep — thresholds are corpus-relative). External
  corroboration for bounding quantity: DS-Agent's own hyperparameter sweep
  found retrieval performance *declines* past a single retrieved case (their
  Figure 6b) — independent evidence that naive volume increases can actively
  hurt, not merely fail to help. This supports, but is not the sole
  justification for, the existing fixed-k design (the parity and
  query-impoverishment arguments above stand on their own).
- **Judge circularity:** rubric frozen pre-run; judge blinded to outcomes;
  evidence quotes required.
- **Corpus coverage asymmetry:** 18/22 Lite competitions have practitioner
  notebooks; retrieved-doc counts are reported per competition per
  condition as descriptive statistics.
- **Low n:** primary metric is directional; secondary metrics and
  mechanistic analysis carry the evidential weight.

## Roadmap

**v1 (this design, POC scope — 2026-07-22):** B1/B2/C1-pilot/C2 on a small
subset (3–5) of Lite-22 (specific competitions pinned pre-run), 3 seeds (1
for C1 pilot), mechanistic judging, writeup. Scope rationale: as a solo POC
the goal is a rigorous, honest end-to-end run — not a fully-powered result.
The mechanistic spec-quality judging is the primary informative signal (it
measures specification quality directly); the MLE-bench score delta is a
conservative secondary lower bound. Kaggle's pre-specified tasks structurally
disadvantage the mechanism (Threats → construct validity), so MLE-bench is
chosen for objective grading, not representativeness, and a small subset is
adequate. Full Lite-22 with a disjoint wider-mle-bench smoke competition is
the natural v1.5 if an initial result is pursued.

**v2 (explicitly out of scope):** MLE-Dojo / interactive specification
reasoning; iteration depth as an experimental variable; real
organizational-corpus case study; synthetic-corruption arm.

## Amendments

(none)
