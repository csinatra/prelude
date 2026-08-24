#!/usr/bin/env bash
# Answer the three open cloud-box questions in one pass, before a run depends on
# them. Read-only except for the optional heavy-deps build (--build).
#
#   1. Does a container under sysbox-runc actually see a GPU? mle-bench runs
#      agents under Sysbox, which has not historically coexisted with NVIDIA's
#      container runtime. cloudbox/container_config.json requests a GPU on the
#      assumption that it can.
#   2. Do the host's resources match what container_config.json pins?
#   3. Why does the heavy-deps base image fail to build? Split into steps so the
#      failure names a culprit instead of one opaque layer.
#
# Usage: scripts/check_cloudbox.sh [--build]
set -uo pipefail

BUILD=false
[ "${1:-}" = "--build" ] && BUILD=true
CONFIG="$(cd "$(dirname "$0")/.." && pwd)/cloudbox/container_config.json"

echo "=== 1. host ==="
echo "vCPUs:  $(nproc)"
echo "memory: $(free -g 2>/dev/null | awk '/^Mem:/{print $2" GiB"}')"
echo "disk:   $(df -h /var/lib/docker 2>/dev/null | awk 'NR==2{print $4" free of "$2}')"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "GPU: nvidia-smi unavailable"

echo
echo "=== 2. pinned config vs host ==="
if [ -f "$CONFIG" ]; then
    PINNED=$(python3 -c "import json;print(int(json.load(open('$CONFIG'))['nano_cpus'])//10**9)" 2>/dev/null)
    echo "container_config.json pins ${PINNED} vCPUs; host has $(nproc)"
    [ "${PINNED:-0}" -gt "$(nproc)" ] && echo "  MISMATCH: Docker will reject a cap above host capacity"
else
    echo "missing: $CONFIG"
fi

echo
echo "=== 3. GPU passthrough ==="
echo "-- plain runtime (baseline: does the host's nvidia stack work at all?)"
docker run --rm --gpus all ubuntu:22.04 nvidia-smi -L 2>&1 | head -3

echo "-- sysbox-runc + device_requests (what mle-bench agents actually get)"
docker run --rm --runtime=sysbox-runc --gpus all ubuntu:22.04 nvidia-smi -L 2>&1 | head -5
echo "   ^ if this failed but the plain runtime worked, Sysbox is the blocker:"
echo "     the choice is between Sysbox isolation and GPU training (DECISIONS.md)"

if [ "$BUILD" = false ]; then
    echo
    echo "(re-run with --build to localize the heavy-deps failure)"
    exit 0
fi

echo
echo "=== 4. heavy-deps install, step by step ==="
# Upstream installs requirements + tensorflow[and-cuda] + torch in ONE RUN, so a
# failure is unattributable and nothing caches. Same commands, separately.
MLEBENCH_DIR="${MLEBENCH_DIR:-$HOME/work/mle-bench}"
REQ="$MLEBENCH_DIR/environment/requirements.txt"
[ -f "$REQ" ] || { echo "missing $REQ"; exit 1; }

run_step() {
    echo "-- $1"
    docker run --rm -v "$REQ:/tmp/requirements.txt:ro" python:3.11-slim \
        bash -c "pip install --dry-run $2" 2>&1 | tail -5
}

# --dry-run resolves without downloading: catches version conflicts and yanked
# packages in seconds. It does NOT catch source builds that fail to compile, so
# a clean dry run does not prove the real build succeeds.
#
# Diagnostic only — no framework is favored here. Whether torch and tensorflow
# actually conflict is unconfirmed; if the failure turns out to be torchtext
# (sunset, and nothing needs it) or a legacy source build, both survive and no
# tiebreak is needed. Establish what fails before deciding what to drop.
run_step "full requirements.txt as upstream ships it" "-r /tmp/requirements.txt"
run_step "tensorflow[and-cuda]==2.17 alone" "tensorflow[and-cuda]==2.17"
run_step "torch stack as upstream pins it" \
    "torch==2.2.0 torchaudio==2.2.0 torchtext==0.17.0 torchvision==0.17.0"
run_step "torch stack minus torchtext" "torch==2.2.0 torchaudio==2.2.0 torchvision==0.17.0"
run_step "torchtext==0.17.0 alone (sunset package)" "torchtext==0.17.0"
run_step "torch + tensorflow together" \
    "torch==2.2.0 torchvision==0.17.0 tensorflow[and-cuda]==2.17"
