#!/bin/bash
# Provision a GPU cloud box for MLE-bench agent runs. Idempotent — safe to
# re-run after interruption. Assumes Ubuntu 22.04 with NVIDIA drivers + Docker
# (standard on Lambda GPU images) and passwordless sudo.
#
# Design decisions (2026-07-16): no custom image for prelude itself — clone
# and provision; the reproducibility that matters is mle-bench's pinned
# commit + our git commit stamped in run manifests. Prepared competition
# data lives on a persistent volume ($MLEBENCH_DATA_DIR) so re-renting
# compute never re-downloads it.
#
# On-box bring-up findings encoded here (2026-07-22, first real provision):
#   - Lambda Stack ships Python 3.10; mle-bench requires >=3.11 → install 3.11.
#   - mle-bench runs every agent under the Sysbox runtime → install sysbox-ce.
#   - The agent dir must be COPIED, not symlinked (mle-bench's registry uses
#     pathlib glob('**/config.yaml'), which doesn't follow dir symlinks <3.13).
#   - Agent images are pre-built (base mlebench-env, then aide-prelude).
#   - Kaggle: the pinned kaggle client only accepts a LEGACY-format API token
#     (kaggle.com > Settings > API); newer tokens 401.
#   - git-lfs: mle-bench's leaderboards are LFS objects; without a pull they're
#     pointer files and grading's medal-ranking fails on every competition.
#   - The base image's heavy-deps stack (tensorflow/torch) is ON, matching
#     mle-bench's own Dockerfile default. It was off here while the build was
#     believed broken; a 2026-09-01 rebuild succeeded with no package failing, so
#     that block is resolved. It is not optional for eval runs: AIDE's own
#     environment prompt tells the agent "all packages are already installed" and
#     names xgboost, torch-geometric, timm and statsmodels, none of which a lean
#     image carries except timm. TensorFlow and torch coexist on the GPU
#     (verified on an A10); pip warns that TF's CUDA 12.3 wheels do not satisfy
#     torch 2.2.0's 12.1 pins, but 12.x is forward compatible and both run.
#   - prelude needs its OWN venv on the box (2026-08-24) — the batch driver runs
#     from this repo, and prelude requires >=3.12 while mle-bench pins 3.11.
#
# Required env:
#   ANTHROPIC_API_KEY   — agent model calls from inside the container (AIDE)
#   KAGGLE_USERNAME / KAGGLE_KEY — mlebench prepare downloads (legacy token)
#   MLEBENCH_DATA_DIR   — persistent volume mount point for prepared data
# Optional env:
#   INSTALL_HEAVY_DEPENDENCIES=false — skip the full tf/torch base env. Only for
#     a throwaway wiring check: it leaves AIDE's prompt claiming packages that
#     are not there (see above), so it is not valid for a run that produces data.
set -euo pipefail

MLEBENCH_COMMIT="507f92e1138bb6e40dac5c6ee7a6758e6424bf97" # pinned; aide-prelude forked at this commit
PRELUDE_REPO="https://github.com/csinatra/prelude.git"
WORK_DIR="${WORK_DIR:-$HOME/work}"
PY="python3.11"
SYSBOX_VERSION="0.7.0"
SYSBOX_SHA256="eeff273671467b8fa351ab3d40709759462dc03d9f7b50a1b207b37982ce40a9"
INSTALL_HEAVY_DEPENDENCIES="${INSTALL_HEAVY_DEPENDENCIES:-true}"

for var in ANTHROPIC_API_KEY KAGGLE_USERNAME KAGGLE_KEY MLEBENCH_DATA_DIR; do
  if [ -z "${!var:-}" ]; then
    echo "ERROR: $var is not set" >&2
    exit 1
  fi
done

# ── Docker + GPU preflight ──────────────────────────────────────────────
command -v docker >/dev/null || { echo "ERROR: docker not installed" >&2; exit 1; }
if ! docker info >/dev/null 2>&1; then
  echo "ERROR: cannot reach the Docker daemon as $(whoami) — usually a group issue." >&2
  echo "  Fix, then reconnect (group change needs a fresh login) and re-run:" >&2
  echo "    sudo usermod -aG docker \$USER && exit" >&2
  exit 1
fi
nvidia-smi >/dev/null || { echo "ERROR: no NVIDIA GPU visible" >&2; exit 1; }
docker run --rm --gpus all ubuntu:22.04 true 2>/dev/null \
  || { echo "ERROR: docker cannot access the GPU (install nvidia-container-toolkit)" >&2; exit 1; }

# ── Python 3.11 (mle-bench requires >=3.11; Lambda Stack ships 3.10) ─────
if ! command -v "$PY" >/dev/null; then
  sudo apt-get update
  sudo apt-get install -y software-properties-common
  sudo add-apt-repository -y ppa:deadsnakes/ppa
  sudo apt-get update
  sudo apt-get install -y python3.11 python3.11-venv python3.11-dev
fi

# ── Sysbox runtime (mle-bench runs agents under sysbox-runc) ─────────────
if ! docker info 2>/dev/null | grep -q sysbox-runc; then
  sudo apt-get update && sudo apt-get install -y jq
  SYSBOX_DEB="/tmp/sysbox-ce_${SYSBOX_VERSION}.linux_amd64.deb"
  wget -qO "$SYSBOX_DEB" \
    "https://github.com/nestybox/sysbox/releases/download/v${SYSBOX_VERSION}/sysbox-ce_${SYSBOX_VERSION}.linux_amd64.deb"
  echo "${SYSBOX_SHA256}  ${SYSBOX_DEB}" | sha256sum -c
  sudo apt-get install -y "$SYSBOX_DEB"   # reconfigures + restarts dockerd
fi

mkdir -p "$WORK_DIR" "$MLEBENCH_DATA_DIR"
cd "$WORK_DIR"

# ── prelude (this repo) ────────────────────────────────────────────────
if [ ! -d prelude ]; then git clone "$PRELUDE_REPO"; fi
cd prelude && git pull --ff-only && cd ..

# prelude's own environment. The box runs the batch driver (harness.batch) out
# of this repo, so its deps have to be installed here too — a separate venv from
# mle-bench's, which is pinned to 3.11 while prelude requires >=3.12. Omitted on
# the first provision, so the first drain died on `ModuleNotFoundError: dotenv`
# after the box was otherwise fully built.
#
# uv installs itself to ~/.local/bin, which a NON-interactive shell (`ssh box
# '...'`) does not have on PATH — the profile that adds it is only read by login
# shells. Spelled out absolutely so this works under both.
UV="$HOME/.local/bin/uv"
if [ ! -x "$UV" ]; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi
# --frozen: install uv.lock exactly, never silently re-resolve. The lock is what
# makes an eval run reproducible (pyproject carries lower bounds only).
(cd prelude && "$UV" sync --frozen)

# Run outputs (registry + artifacts + logs) belong on the mounted volume, not
# the ephemeral boot disk: the queue/resume/abandon model needs the registry to
# survive termination, and --terminate-on-done would otherwise destroy the very
# outputs it just produced.
#
# This used to be a symlink at prelude/results. That could never work: the repo
# tracks results/corpus_export/manifest.json, so the clone creates the directory
# first and the link was skipped with a warning nobody reads. The path is now
# explicit config (PRELUDE_RESULTS_DIR), and harness.batch refuses to terminate
# an instance whose results are still on the boot disk.
RESULTS_DIR="$MLEBENCH_DATA_DIR/prelude-results"
mkdir -p "$RESULTS_DIR"
echo "Run outputs -> $RESULTS_DIR"
echo "  Add to $WORK_DIR/prelude/.env:  PRELUDE_RESULTS_DIR=$RESULTS_DIR"

# git-lfs — mle-bench stores per-competition leaderboards in LFS. Without a pull
# they're pointer files, and grading's medal-ranking asserts ("Leaderboard must
# have a `score` column") on every competition.
command -v git-lfs >/dev/null || { sudo apt-get update && sudo apt-get install -y git-lfs; }

# ── mle-bench, pinned ─────────────────────────────────────────────────
if [ ! -d mle-bench ]; then git clone https://github.com/openai/mle-bench.git; fi
cd mle-bench
git fetch --all --quiet && git checkout --quiet "$MLEBENCH_COMMIT"
git lfs install --local && git lfs pull   # fetch the real leaderboard CSVs

# PRELUDE spec-mount hook (B/C runs): when PRELUDE_SPEC_PATH is set, mount that
# spec.md read-only at /home/spec/spec.md so aide-prelude/start.sh appends it as
# ADVISOR CONTEXT. Orchestration-only, identical across A/B/C. Context-matched
# (line-number tolerant), idempotent, and fails loudly if the pinned run.py drifts.
if ! grep -q PRELUDE_SPEC_PATH agents/run.py; then
  patch -p1 --forward --fuzz=3 \
    < "$WORK_DIR/prelude/cloudbox/agents/aide-prelude/patches/mlebench-run-spec-mount.patch"
fi
grep -q PRELUDE_SPEC_PATH agents/run.py \
  || { echo "ERROR: spec-mount hook not applied to agents/run.py (patch context drift?)" >&2; exit 1; }
# venv on 3.11 (a stale 3.10 venv can't install mlebench — rebuild if wrong)
if [ ! -x .venv/bin/python ] \
   || ! .venv/bin/python -c 'import sys; sys.exit(0 if sys.version_info[:2] >= (3, 11) else 1)'; then
  rm -rf .venv
  "$PY" -m venv .venv
fi
.venv/bin/pip install --quiet -e .

# Our agent variant: COPY, not symlink — mle-bench's registry discovers agents
# with pathlib glob('**/config.yaml'), which does not descend into symlinked
# directories on Python <3.13, so a symlinked aide-prelude is never registered.
rm -rf agents/aide-prelude
cp -r "$WORK_DIR/prelude/cloudbox/agents/aide-prelude" agents/aide-prelude

# ── base image: mlebench-env ──────────────────────────────────────────
# Load a saved tarball if present (skips the ~30-min build), else build.
#
# The saved tarball is checked against the heavy-deps setting rather than
# trusted. A tarball predating the 2026-09-01 switch carries a LEAN image, and
# loading it silently produces a box that disagrees with what this script says
# it built — the failure mode that hid a missing tensorflow/xgboost for weeks.
# Since the image is a frozen input to the eval (RESEARCH_DESIGN.md missing-data
# rule), a stale one has to be discarded, not inherited.
if ! docker image inspect mlebench-env >/dev/null 2>&1; then
  SAVED_ENV="$MLEBENCH_DATA_DIR/mlebench-env.tar.gz"
  if [ -f "$SAVED_ENV" ]; then
    echo "Loading saved mlebench-env from $SAVED_ENV ..."
    gunzip -c "$SAVED_ENV" | docker load
    if [ "$INSTALL_HEAVY_DEPENDENCIES" = "true" ] \
       && ! docker run --rm --entrypoint bash mlebench-env \
              -lc "conda run -n agent python -c 'import tensorflow'" >/dev/null 2>&1; then
      echo "Saved image predates heavy deps — discarding and rebuilding."
      docker image rm -f mlebench-env
    fi
  fi
fi
if ! docker image inspect mlebench-env >/dev/null 2>&1; then
  docker build --platform=linux/amd64 -t mlebench-env -f environment/Dockerfile . \
    --build-arg INSTALL_HEAVY_DEPENDENCIES="$INSTALL_HEAVY_DEPENDENCIES"
fi

# ── agent image: aide-prelude ─────────────────────────────────────────
export SUBMISSION_DIR=/home/submission LOGS_DIR=/home/logs CODE_DIR=/home/code AGENT_DIR=/home/agent
docker build --platform=linux/amd64 -t aide-prelude agents/aide-prelude/ \
  --build-arg SUBMISSION_DIR="$SUBMISSION_DIR" --build-arg LOGS_DIR="$LOGS_DIR" \
  --build-arg CODE_DIR="$CODE_DIR" --build-arg AGENT_DIR="$AGENT_DIR"
cd ..

echo "Provisioned (results/ -> $RESULTS_DIR on the persistent volume). Next steps:"
echo "  1. Prepare the smoke competition (persistent volume, one-time):"
echo "     cd mle-bench && .venv/bin/mlebench prepare -c random-acts-of-pizza --data-dir \$MLEBENCH_DATA_DIR"
echo "     (join the competition on kaggle.com first; use a LEGACY-format Kaggle token)"
echo "  2. Smoke run (throwaway wiring test): aide-prelude/dev, no /home/spec mount:"
echo "     .venv/bin/python run_agent.py --agent-id aide-prelude/dev \\"
echo "       --competition-set experiments/splits/random-acts-of-pizza.txt --data-dir \$MLEBENCH_DATA_DIR"
echo "     (create the split file: echo random-acts-of-pizza > experiments/splits/random-acts-of-pizza.txt)"
echo "  3. Condition runs: mount the run's spec.md at /home/spec/spec.md"
echo "  4. Drain the queue back-to-back once the seams are confirmed (prelude's"
echo "     venv, not mle-bench's — they are different Python versions):"
echo "     cd $WORK_DIR/prelude && .venv/bin/python -m harness.batch \\"
echo "       --data-dir \$MLEBENCH_DATA_DIR --terminate-on-done --instance-id <id>"
