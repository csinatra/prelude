#!/bin/bash
# Provision a GPU cloud box for MLE-bench agent runs. Idempotent — safe to
# re-run after interruption. Assumes Ubuntu 22.04+ with NVIDIA drivers and
# Docker preinstalled (standard on Lambda/RunPod-style GPU images).
#
# Design decisions (2026-07-16): no custom image for prelude itself — clone
# and provision; the reproducibility that matters is mle-bench's pinned
# commit + our git commit stamped in run manifests. Prepared competition
# data lives on a persistent volume ($MLEBENCH_DATA_DIR) so re-renting
# compute never re-downloads it.
#
# Required env:
#   ANTHROPIC_API_KEY   — agent model calls from inside the container (AIDE)
#   KAGGLE_USERNAME / KAGGLE_KEY — mlebench prepare downloads
#   MLEBENCH_DATA_DIR   — persistent volume mount point for prepared data
set -euo pipefail

MLEBENCH_COMMIT="507f92e1138bb6e40dac5c6ee7a6758e6424bf97" # pinned; aide-prelude forked at this commit
PRELUDE_REPO="https://github.com/csinatra/prelude.git"
WORK_DIR="${WORK_DIR:-$HOME/work}"

for var in ANTHROPIC_API_KEY KAGGLE_USERNAME KAGGLE_KEY MLEBENCH_DATA_DIR; do
  if [ -z "${!var:-}" ]; then
    echo "ERROR: $var is not set" >&2
    exit 1
  fi
done

command -v docker >/dev/null || { echo "ERROR: docker not installed" >&2; exit 1; }
nvidia-smi >/dev/null || { echo "ERROR: no NVIDIA GPU visible" >&2; exit 1; }
docker run --rm --gpus all ubuntu:22.04 true 2>/dev/null \
  || { echo "ERROR: docker cannot access the GPU (install nvidia-container-toolkit)" >&2; exit 1; }

mkdir -p "$WORK_DIR" "$MLEBENCH_DATA_DIR"
cd "$WORK_DIR"

# ── prelude (this repo) ────────────────────────────────────────────────
if [ ! -d prelude ]; then git clone "$PRELUDE_REPO"; fi
cd prelude && git pull --ff-only && cd ..

# Persist results/ (runs.jsonl registry + artifacts + logs) on the mounted
# volume, not the ephemeral boot disk. The queue/resume/abandon model needs the
# registry to survive instance termination, and --terminate-on-done would
# otherwise destroy the very run outputs and failure logs it just produced.
RESULTS_DIR="$MLEBENCH_DATA_DIR/prelude-results"
mkdir -p "$RESULTS_DIR"
if [ -e "$WORK_DIR/prelude/results" ] && [ ! -L "$WORK_DIR/prelude/results" ]; then
  echo "WARNING: $WORK_DIR/prelude/results exists and is not a symlink — leaving as-is" >&2
else
  ln -sfn "$RESULTS_DIR" "$WORK_DIR/prelude/results"
fi

# ── mle-bench, pinned ─────────────────────────────────────────────────
if [ ! -d mle-bench ]; then git clone https://github.com/openai/mle-bench.git; fi
cd mle-bench
git fetch --all --quiet && git checkout --quiet "$MLEBENCH_COMMIT"
python3 -m venv .venv 2>/dev/null || true
.venv/bin/pip install --quiet -e .

# link our agent variant into mle-bench's agents dir
ln -sfn "$WORK_DIR/prelude/cloudbox/agents/aide-prelude" agents/aide-prelude
cd ..

echo "Provisioned (results/ -> $RESULTS_DIR on the persistent volume). Next steps:"
echo "  1. Prepare the smoke competition (persistent volume, one-time):"
echo "     cd mle-bench && .venv/bin/mlebench prepare -c random-acts-of-pizza --data-dir \$MLEBENCH_DATA_DIR"
echo "  2. Smoke run (throwaway wiring test): aide-prelude/dev, no /home/spec mount"
echo "     run agent aide-prelude/dev per mle-bench README — confirm the [confirm on box] seams"
echo "  3. Condition runs: mount the run's spec.md at /home/spec/spec.md"
echo "  4. Drain the queue back-to-back once the seams are confirmed:"
echo "     python -m harness.batch --data-dir \$MLEBENCH_DATA_DIR --terminate-on-done --instance-id <id>"
