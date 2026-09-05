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

- **Accounts:** GPU provider — a **real VM** (single A10/A100-class, Ubuntu
  22.04, NVIDIA drivers + Docker + passwordless sudo). Lambda A10 is the
  reference box; it must be a VM, not a container-as-a-service, because
  mle-bench runs agents in Docker under the Sysbox runtime (`setup_cloudbox.sh`
  installs Sysbox and Python 3.11). Kaggle account (**join each competition on
  kaggle.com first**, and use a **legacy-format API token** — Settings → API →
  the classic `kaggle.json`; the pinned mle-bench Kaggle client 401s on
  newer-format tokens). Anthropic API key with credits (AIDE's agent calls;
  budget in [COST_ESTIMATE.md](COST_ESTIMATE.md)).
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
entry. Build the full grid per RESEARCH_DESIGN.md (A/B1/B2/C2 x the eval
competitions x 3 seeds; C1 pilot subset). `--condition A` takes the same
command but builds no spec and makes no LLM call: it only registers the run,
so no run directory appears until the agent's outputs are preserved into it.
Spot-check one spec.md per
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
cp ~/work/prelude/.env.cloudbox.example ~/work/prelude/.env
# edit .env, then export it for the setup script (the `python -m` entry points
# load .env themselves — see pipeline/env.py):
set -a && . ~/work/prelude/.env && set +a
~/work/prelude/scripts/setup_cloudbox.sh
```

The script is idempotent and does the full bring-up: Docker/GPU preflight (with
a clear message if you still need `sudo usermod -aG docker $USER` + re-login),
**Python 3.11** (Lambda Stack ships 3.10; mle-bench needs ≥3.11), the **Sysbox
runtime**, the mle-bench clone at the pinned commit installed in a 3.11 venv,
**copies** `cloudbox/agents/aide-prelude` into mle-bench's agents dir (a
symlink is invisible to mle-bench's glob-based registry on Python <3.13), and
builds the **base `mlebench-env`** and the **`aide-prelude` agent** images.

The base image is built with **heavy dependencies on**, matching mle-bench's own
Dockerfile default. This is not optional: AIDE's environment prompt tells the
agent every package is already installed and names `xgboost`, `torch-geometric`,
`timm` and `statsmodels`, and a lean image carries only `timm`. TensorFlow and
torch coexist on the GPU, verified on an A10. A saved
`$MLEBENCH_DATA_DIR/mlebench-env.tar.gz` is loaded in preference to building, and
is checked for the heavy stack before being trusted — a tarball without it is
discarded and rebuilt, so the box cannot silently disagree with what the script
reports building.

**Set `PRELUDE_RESULTS_DIR` before running anything.** The script creates
`$MLEBENCH_DATA_DIR/prelude-results` and prints the line to add, but does not
write it for you, and nothing fails loudly if it is missing: runs simply write to
the boot disk, which a terminated instance takes with it. Put the box's `.env` at
`~/work/prelude/.env` — not only in the shell used for provisioning, since the
batch driver reads the file — then confirm both settings resolve:

```bash
cd ~/work/prelude && .venv/bin/python -c \
  "from harness import registry; print(registry.active_stage(), registry.results_root())"
```

**Join every competition on kaggle.com first.** `mlebench prepare` downloads
through the Kaggle API, which refuses competition data until the account has
accepted that competition's rules in the browser. There is no API call for it and
no way to batch it, so it is a manual, per-competition prerequisite that gates
the whole pipeline: an unjoined competition fails at prepare, before any run
starts. Do the full set for a pinned eval subset in one sitting rather than
discovering it one competition at a time during a drain. The token must also be
the **legacy** format (kaggle.com > Settings > API); newer tokens 401 against the
pinned client.

Then prepare each competition onto the persistent volume (one-time per
competition):

```bash
cd ~/work/mle-bench
.venv/bin/mlebench prepare -c random-acts-of-pizza --data-dir $MLEBENCH_DATA_DIR
```

## 4. Smoke run (integration test — throwaway)

First agent run is a wiring check, not a data point. Use
`random-acts-of-pizza` — reserved as the off-eval integration competition
(excluded from the eval subset; small text task, description already local) —
with the `aide-prelude/dev` 8-step variant and **no spec mounted**
(byte-identical to stock AIDE):

```bash
cd ~/work/mle-bench
mkdir -p experiments/splits
echo random-acts-of-pizza > experiments/splits/random-acts-of-pizza.txt
.venv/bin/python run_agent.py --agent-id aide-prelude/dev \
    --competition-set experiments/splits/random-acts-of-pizza.txt \
    --data-dir $MLEBENCH_DATA_DIR
```

mle-bench takes a `--competition-set` file (one competition id per line), not a
single `--competition` flag. The **agent-model question is resolved**
(2026-07-22): the pin is `claude-haiku-4-5-20251001`, aideml v6.3.3 pristine —
the 5-family/Opus-4.7+ reject `temperature` and Sonnet 5 defaults thinking on
(breaking aideml's response parsing), while Haiku 4.5 runs it unmodified. See
`cloudbox/agents/aide-prelude/config.yaml` and RESEARCH_DESIGN.md. Inspect the
run's `run.log` under `runs/<group>/<comp>_<uuid>/`: a clean run gets past AIDE
drafting into code execution across its 8 steps and lands a `submission.csv` +
AIDE journal in the run's output dir.

Discard the smoke result. It runs unmounted and so looks like a Condition A run,
but it is not a data point: the dev budget is a fraction of the eval budget and
the competition is off-eval by design. A's runs are prepared with every other
condition in step 2.

## 5. Calibration run (one-time, pins the agent budget)

Run before the eval grid, on the reserved calibration competition
(`uw-madison-gi-tract-image-segmentation`; see RESEARCH_DESIGN.md for why that
one). Its purpose is the convergence curve that fixes the per-run budget, so it
uses the full agent, the pinned `EVAL_MODEL` spec, and its own registry stage:

```bash
PRELUDE_REGISTRY_STAGE=calibration
```

Two checks, neither of which fails loudly on its own.

**After the first spec is built, confirm the token metrics are not null.**

```bash
PRELUDE_REGISTRY_STAGE=calibration python -c "
from harness import registry
run = list(registry.load_runs().values())[-1]
print({k: run.get(k) for k in ('spec_tokens', 'block_tokens', 'synthesis_tokens')})"
```

`harness.runner._count_tokens` catches every exception and returns `None`,
including on a missing `MODEL` env var. A calibration run that silently nulls all
three would produce H3 measurements with nothing behind them and no error
anywhere to show for it.

**After the agent run, read the best node's own analysis against the injected
`spec.md` once.** This is the fastest detector of a broken injection: if the
agent's plan shows no trace of spec content, that has to be known during
calibration rather than at judging time, when every run is already spent. It is
an operational smoke check and not evidence for H2, which the base-rate
counterfactual measures.

## 6. Condition runs

The spec is injected via the `PRELUDE_SPEC_PATH` env var. `setup_cloudbox.sh`
patches mle-bench's `agents/run.py` (`run_agent.py` has no `--extra-mount`) with a
hook: when `PRELUDE_SPEC_PATH` is set, it mounts that file read-only at
`/home/spec/spec.md`, which `aide-prelude/start.sh` appends as ADVISOR CONTEXT.
Condition A leaves it unset (stock aide). The batch driver (`harness.batch`) sets
it per run from `spec_path`; a manual B/C run:

```bash
cd ~/work/mle-bench
echo <competition> > experiments/splits/<run_key>.txt
PRELUDE_SPEC_PATH=~/work/prelude/results/$PRELUDE_REGISTRY_STAGE/<run_key>/spec.md \
  .venv/bin/python run_agent.py --agent-id aide-prelude \
    --competition-set experiments/splits/<run_key>.txt --data-dir $MLEBENCH_DATA_DIR \
    --container-config ~/work/prelude/cloudbox/container_config.json
```

`--container-config` is not optional: mle-bench's default gives the agent 4
vCPUs and no GPU, against the benchmark's own 36-vCPU + A10 baseline. `harness.batch`
passes it automatically; a manual run must pass it too, or the run is not
comparable to the others. See [cloudbox/README.md](../cloudbox/README.md).

**Confirmed on box (2026-07-24):** the B/C spec mount works end-to-end. On a
`random-acts-of-pizza` `/dev` run, `run.py` logged `cat /home/spec/spec.md`
right after the `ADVISOR CONTEXT` banner (absent for A), the C2 structured
content (flags/recommendations) appeared in AIDE's prompt, and a valid
`submission.csv` + journal landed in the run's output dir.

## 7. Record and grade (box)

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

Steps 6–7 above are the manual, one-run-at-a-time path — use them for the
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

**Reviewing results and the terminate tradeoff.** Run outputs go to
`PRELUDE_RESULTS_DIR` on the persistent filesystem, so terminating never loses
run state — but a Lambda filesystem is only reachable through a *running*
instance it's attached to (no standalone mount; VS Code Remote-SSH needs a live
box). Two patterns:

- *Attended (smoke + early grid):* run **without** `--terminate-on-done`. The
  box stays up after draining; review over SSH/VS Code, `rsync` results to the
  laptop (the analysis machine), then terminate manually.
- *Unattended (mature runs):* use `--terminate-on-done` for cost control.
  Results persist on the filesystem; to review, either pull them beforehand or
  re-attach the filesystem to a fresh (cheap) instance. `--retry-abandoned`
  later just needs the same filesystem re-attached — the registry state is
  intact.

**[confirm on box]:** the 2026-07-24 smoke confirmed the output layout these
seams target (via a manual `run_agent` + `grade-sample`), but `harness.batch`'s
own `_grade` (JSONL form) and `_read_journal_metrics` were not exercised —
verify on the first automated batch run before relying on the drained path.

## 8. Merge back and analyze (dev machine)

```bash
rsync -av <box>:~/work/prelude/results/ results/
# registries merge by concatenation; load_runs() merges fields per run_key
# WITHIN one stage. Both machines must be on the same PRELUDE_REGISTRY_STAGE,
# or the box appends to a registry the dev machine never reads (docs/DATA.md).
```

Then: `analysis/judge.py` against the frozen rubric ([JUDGE_RUBRIC.md](JUDGE_RUBRIC.md)
— do not edit it), trajectory/efficiency analysis per the cost/efficiency
accounting section of RESEARCH_DESIGN.md.

## Budget discipline

- Release GPU instances between batches; the volume persists.
- The registry is the spend ledger: `spec_llm_*` fields for upfront cost,
  `agent_llm_*` for the agent side, `agent_wallclock_secs` for GPU time.
  Reconcile against COST_ESTIMATE.md as runs accumulate — flag early if per-run
  actuals exceed estimates.
- **Checking on an unattended drain.** `harness.status` reads the registry, so
  live state means running it on the box. One-shot from the dev machine, no
  session to keep alive:

  ```bash
  ssh <box> 'cd ~/work/prelude && .venv/bin/python -m harness.status --logs'
  ```

  `--logs` names the running container and prints the `docker logs -f` command
  to stream it; `--follow` polls until the queue drains, for a second terminal
  left open. Run on the dev machine it works but reports whatever was last
  synced, and finds no container. A stall is flagged only while work is
  outstanding, so a finished queue's silence is not reported as a problem.
- **Watching a run in progress.** `harness.status` answers "is the queue
  moving"; `harness.dashboard` answers "what is the agent doing", which needs
  the search tree, the host metrics and the logs side by side. Serve it on the
  box and reach it over a tunnel rather than opening a port on a box holding API
  keys:

  ```bash
  ssh <box> 'cd ~/work/prelude && .venv/bin/python -m harness.dashboard --serve'
  ssh -N -L 8000:localhost:8000 <box>   # second terminal, then open localhost:8000
  ```

  The page re-renders every 30s. To watch a log as it is written, use the
  `docker exec ... tail -f` commands the page prints — `aide.log` for what the
  agent is doing, `aide.verbose.log` for why a node came back buggy, since
  `journal.json` serializes `term_out` empty. `--out page.html` writes one
  snapshot instead of serving.
- Serial runtime is a scoping constraint alongside cost: one instance drains the
  queue one run at a time, so the per-run cap sets how long the whole grid takes.
