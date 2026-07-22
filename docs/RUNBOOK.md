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
git clone https://github.com/csinatra/prelude.git ~/work/prelude
# fill from the template, then source it — box gets only the four keys it
# needs (least privilege; spec-pipeline keys stay on the dev machine)
cp ~/work/prelude/.env.cloudbox.example ~/work/prelude/.env.cloudbox
# edit .env.cloudbox, then:
set -a && . ~/work/prelude/.env.cloudbox && set +a
~/work/prelude/scripts/setup_cloudbox.sh
```

The script preflights GPU/Docker, clones mle-bench at the pinned commit,
installs it, symlinks `cloudbox/agents/aide-prelude` into mle-bench's agents
dir, and symlinks `results/` onto the persistent volume
(`$MLEBENCH_DATA_DIR/prelude-results`) so the registry, artifacts, and failure
logs survive instance termination — without this, `--terminate-on-done` would
destroy the run state it just produced. Then prepare each competition onto the
persistent volume (one-time per competition):

```bash
cd ~/work/mle-bench
.venv/bin/mlebench prepare -c random-acts-of-pizza --data-dir $MLEBENCH_DATA_DIR
```

## 4. Smoke run (integration test — throwaway)

First agent run is a wiring check, not a data point. Use
`random-acts-of-pizza` — reserved as the off-eval integration competition
(excluded from the eval subset; small text task, description already local) —
with the `aide-prelude/dev` 8-step variant and **no spec mounted**
(byte-identical to stock AIDE). Launch per mle-bench's README (`run_agent.py`
with `--agent-id aide-prelude/dev`).
**[confirm on box]:** exact flags; that the agent image builds cleanly
from our forked dir; and that aideml v6.3.3's Anthropic backend accepts
the configured model id. If `claude-sonnet-5` isn't supported, drop to
the newest Sonnet it accepts — the hard requirement is one pinned model
for every run in the grid, not a particular version (see the note in
`cloudbox/agents/aide-prelude/config.yaml`); record the final choice in
RESEARCH_DESIGN.md before eval runs.

Discard the smoke result: the 8-step dev budget is too short to be a valid
Condition A run, and the competition is off-eval by design. The matched-A
anchor is **separate and contingent** (RESEARCH_DESIGN.md Condition A note) —
if triggered, it runs unmounted `aide-prelude` at the full budget on the eval
competitions and enters the registry via `harness.advance register`
(`--condition A`), then advances through agent-run/graded as in step 6.

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

### Automated: drain the queue (recommended once the seams are confirmed)

Steps 4–6 above are the manual, one-run-at-a-time path — use them for the
first smoke run to confirm the box-specific seams. After that, `harness.batch`
runs every unfinished run in the registry back-to-back (agent → grade →
advance), so the GPU never idles between runs or after the last one:

```bash
python -m harness.batch --data-dir $MLEBENCH_DATA_DIR \
    --terminate-on-done --instance-id <lambda-instance-id>
```

It queues by `run_key` (competition × condition × seed), grouped by
competition so a problem's full cross-condition set completes contiguously,
blocks on each AIDE run to completion, and with `--terminate-on-done`
terminates the Lambda box once drained (needs `LAMBDA_API_KEY` +
`--instance-id`).

**Failures are parked, not retried.** A run that raises is marked `abandoned`
in the registry with its `last_error` and skipped — so a deterministic failure
never re-runs on the next invocation (which would burn GPU unbounded and keep
the box from draining to termination). There is no automatic retry: inspect the
parked run's `last_error` and mle-bench's per-run logs, fix the root cause, then
re-queue explicitly:

```bash
python -m harness.batch --data-dir $MLEBENCH_DATA_DIR --retry-abandoned
```

**Reviewing results and the terminate tradeoff.** `results/` is symlinked onto
the persistent filesystem, so terminating never loses run state — but a Lambda
filesystem is only reachable through a *running* instance it's attached to (no
standalone mount; VS Code Remote-SSH needs a live box). Two patterns:

- *Attended (smoke + early grid):* run **without** `--terminate-on-done`. The
  box stays up after draining; review over SSH/VS Code, `rsync` results to the
  laptop (the analysis machine), then terminate manually.
- *Unattended (mature runs):* use `--terminate-on-done` for cost control.
  Results persist on the filesystem; to review, either pull them beforehand or
  re-attach the filesystem to a fresh (cheap) instance. `--retry-abandoned`
  later just needs the same filesystem re-attached — the registry state is
  intact.

**[confirm on box]:** the `_run_agent` / `_locate_outputs` /
`_read_journal_metrics` / `_grade` seams in `harness/batch.py` carry the
guessed mle-bench commands and output layout — verify and fix them during the
smoke run before relying on the automated path.

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
