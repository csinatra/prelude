# Runbook — cloud box setup and experiment execution

Operational guide for running the eval end-to-end. The research framing
lives in [RESEARCH_DESIGN.md](RESEARCH_DESIGN.md); this is the how-to.
Steps marked **[confirm on box]** were built against mle-bench's source
(pinned `507f92e`) but have not yet been exercised on real infrastructure —
verify them during the first smoke run and update this doc.

## The two-machine split

| | Dev machine (laptop) | Cloud box (GPU) |
|---|---|---|
| Runs | corpus ingest, spec builds, analysis | AIDE agent runs, MLE-bench grading |
| LLM calls | yes (Anthropic API) | only AIDE's own agent calls — never the spec pipeline (core constraint 1) |
| Registry writes | `spec_built` entries | `agent_run`, `graded` entries |

The join is `results/runs.jsonl` (append-only, merge = concatenate) plus
the per-run artifact directories.

## 1. Prerequisites

- **Accounts:** GPU provider (single A10/A100-class instance, Ubuntu 22.04+
  image with NVIDIA drivers + Docker — standard on Lambda/RunPod-style
  images); Kaggle account (join each competition on kaggle.com first —
  downloads fail otherwise); Anthropic API key with credits (AIDE's agent
  calls; budget in [COST_ESTIMATE.md](COST_ESTIMATE.md)).
- **Persistent volume** attached to the box for prepared competition data
  (`MLEBENCH_DATA_DIR`). Rent compute per-run; keep the volume. Never
  re-download data because an instance was released.
- **Dev machine state:** corpus ingested (ChromaDB collections present),
  `.env` configured, tests green (`pytest tests/ -q`).

## 2. Build specs (dev machine)

```bash
# one run = one competition x condition x seed
python -m harness.runner --competition spooky-author-identification \
    --condition C2 --seed 0
```

Produces `results/{comp}_{condition}_{seed}/` (spec.md, retrievals,
pipeline output, llm_usage.json, manifest) and a `spec_built` registry
entry. Build the full grid per RESEARCH_DESIGN.md (B1/B2/C2 x ~10 Lite
competitions x 3 seeds; C1 pilot subset). Spot-check one spec.md per
condition before shipping.

Ship to the box (results/ is gitignored — transfer directly):

```bash
rsync -av results/ <box>:~/work/prelude/results/
```

## 3. Provision the box (once per instance)

```bash
export ANTHROPIC_API_KEY=... KAGGLE_USERNAME=... KAGGLE_KEY=...
export MLEBENCH_DATA_DIR=/mnt/persistent/mlebench-data
git clone https://github.com/csinatra/prelude.git ~/work/prelude
~/work/prelude/scripts/setup_cloudbox.sh
```

The script preflights GPU/Docker, clones mle-bench at the pinned commit,
installs it, and symlinks `cloudbox/agents/aide-prelude` into mle-bench's
agents dir. Then prepare each competition onto the persistent volume
(one-time per competition):

```bash
cd ~/work/mle-bench
.venv/bin/mlebench prepare -c spooky-author-identification --data-dir $MLEBENCH_DATA_DIR
```

## 4. Smoke run = matched Condition A (register and keep)

First agent run: `aide-prelude/dev` (8 steps) with **no spec mounted** —
byte-identical to stock AIDE, so it doubles as a matched Condition A data
point (see the Condition A note in RESEARCH_DESIGN.md). Launch per
mle-bench's README (`run_agent.py` with `--agent-id aide-prelude/dev`).
**[confirm on box]:** exact flags; that the agent image builds cleanly
from our forked dir; and that aideml v6.3.3's Anthropic backend accepts
the configured model id. If `claude-sonnet-5` isn't supported, drop to
the newest Sonnet it accepts — the hard requirement is one pinned model
for every run in the grid, not a particular version (see the note in
`cloudbox/agents/aide-prelude/config.yaml`); record the final choice in
RESEARCH_DESIGN.md before eval runs.

Register it — A-runs use the registry too (they have no spec-build phase,
so they enter via `register` rather than `harness.runner`):

```bash
python -m harness.advance register --competition spooky-author-identification \
    --condition A --seed 0
```

Then advance it through agent-run/graded as in step 6.

## 5. Condition runs

Per run: mount the run's spec at `/home/spec/spec.md` inside the agent
container, full-step variant:

```bash
# [confirm on box]: mount mechanism — run_agent.py extra-mount support or
# docker -v injection; aide-prelude/start.sh appends the file iff present
run: --agent-id aide-prelude   + mount results/{run_key}/spec.md -> /home/spec/spec.md
```

Sanity checks per run, before moving on:
- container log shows the ADVISOR CONTEXT section appended (absent for A)
- one AIDE journal + submission.csv landed in the run's output dir

## 6. Record and grade (box)

```bash
cd ~/work/prelude
# after the agent run: wallclock/steps/time-to-first-valid from the AIDE journal
python -m harness.advance agent-run --run-key spooky-author-identification_C2_0 \
    --submission <path> --trajectory <journal path> \
    --wallclock-secs N --steps N --time-to-first-valid-secs N

# grade with mle-bench, then record the metric subset
cd ~/work/mle-bench && .venv/bin/mlebench grade ...   # produces grading_report.json
cd ~/work/prelude
python -m harness.advance graded --run-key spooky-author-identification_C2_0 \
    --report <grading_report.json>
```

Copy `submission.csv` and the journal into the run's artifact dir
(`results/{run_key}/`) — they feed the frozen-rubric judging and
trajectory analysis.

## 7. Merge back and analyze (dev machine)

```bash
rsync -av <box>:~/work/prelude/results/ results/
# registries merge by concatenation; load_runs() merges fields per run_key
```

Then: `analysis/judge.py` against the frozen rubric ([JUDGE_RUBRIC.md](JUDGE_RUBRIC.md)
— do not edit it), trajectory/efficiency analysis per the cost/efficiency
accounting section of RESEARCH_DESIGN.md.

## Budget discipline

- Release GPU instances between batches; the volume persists.
- The registry is the spend ledger: `spec_llm_*` fields for upfront cost,
  `agent_wallclock_secs` for GPU time. Reconcile against COST_ESTIMATE.md
  as runs accumulate — flag early if per-run actuals exceed estimates.
- The contingent A arm (~30 runs) triggers only per the decision rule in
  RESEARCH_DESIGN.md — not by default.
