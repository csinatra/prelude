"""What the run queue is doing right now — one command, no ssh archaeology.

A drain runs unattended for days, and checking on it otherwise means reading raw
registry JSONL, guessing which run is in flight, and hunting for the container's
log path. Nothing is wrong with that except that it is different every time,
which is how a stalled run goes unnoticed overnight.

The registry is the source of truth: every lifecycle transition appends an entry
with a timestamp, so progress, staleness, and results all derive from it. Docker
is consulted only to name the live container, and only when asked.

Usage:
    python -m harness.status                  # queue summary
    python -m harness.status --follow         # refresh until the queue drains
    python -m harness.status --logs           # + the live container and how to tail it
"""

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone

from harness.registry import active_stage, load_runs

# A run legitimately occupies the box for its whole budget, so elapsed time alone
# proves nothing. What is suspicious is elapsed time past the point where even a
# full-budget run should have written *something* — the common failures (wedged
# container, dead driver, box rebooted) all present identically as silence.
STALE_AFTER_SECS = 3600 * 13


def _age_secs(*, timestamp: str | None) -> float | None:
    if not timestamp:
        return None
    try:
        return (datetime.now(tz=timezone.utc) - datetime.fromisoformat(timestamp)).total_seconds()
    except ValueError:
        return None


def _format_age(*, seconds: float | None) -> str:
    if seconds is None:
        return "?"
    hours, minutes = divmod(int(seconds) // 60, 60)
    return f"{hours}h{minutes:02d}m" if hours else f"{minutes}m"


def summarize(*, runs: dict[str, dict] | None = None) -> dict:
    """Queue state derived from the registry alone."""
    runs = load_runs() if runs is None else runs
    by_status: dict[str, int] = {}
    for run in runs.values():
        key = "abandoned" if run.get("abandoned") else run.get("status", "unknown")
        by_status[key] = by_status.get(key, 0) + 1

    def stamp(run: dict) -> str:
        return run.get("updated_at") or run.get("created_at") or ""

    latest = max(runs.values(), key=stamp, default=None)
    idle_secs = _age_secs(timestamp=stamp(latest)) if latest else None
    scored = [
        (run["run_key"], run["score"])
        for run in runs.values()
        if run.get("status") == "graded" and run.get("score") is not None
    ]
    remaining = sum(
        1 for run in runs.values() if run.get("status") != "graded" and not run.get("abandoned")
    )
    return {
        "stage": active_stage(),
        "total": len(runs),
        "by_status": by_status,
        "remaining": remaining,
        "graded": by_status.get("graded", 0),
        "last_activity": latest["run_key"] if latest else None,
        "idle_secs": idle_secs,
        # Silence only means something while work is outstanding. A drained queue
        # is quiet because it is finished, and flagging that trains the reader to
        # ignore the flag that matters.
        "stalled": bool(remaining and idle_secs and idle_secs > STALE_AFTER_SECS),
        "recent_scores": sorted(scored)[-3:],
    }


def live_container() -> str | None:
    """The running agent container, if any.

    mle-bench names each one `competition-<id>-<timestamp>-<uuid>`, so this
    identifies the run in flight directly, where the registry can only infer it
    (the driver writes no "started" marker). Absent off-box, which is fine.
    """
    try:
        names = subprocess.run(
            ["docker", "ps", "--filter", "name=competition-", "--format", "{{.Names}}"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        ).stdout.split()
    except Exception:
        return None
    return names[0] if names else None


def render(*, state: dict, container: str | None = None) -> str:
    lines = [
        f"stage={state['stage']}  graded={state['graded']}/{state['total']}  "
        f"remaining={state['remaining']}",
        "  " + "  ".join(f"{name}={count}" for name, count in sorted(state["by_status"].items())),
    ]
    if state["last_activity"]:
        flag = "   ** STALLED **" if state["stalled"] else ""
        lines.append(
            f"  last activity {_format_age(seconds=state['idle_secs'])} ago"
            f" ({state['last_activity']}){flag}"
        )
    for run_key, score in state["recent_scores"]:
        lines.append(f"  graded {run_key}: {score}")
    if container:
        lines.append(f"  live container: {container}")
        lines.append(f"  tail it:        docker logs -f {container}")
    return "\n".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--follow", action="store_true", help="refresh until the queue drains")
    parser.add_argument("--interval", type=int, default=300, help="--follow seconds (default 300)")
    parser.add_argument("--logs", action="store_true", help="also locate the live container")
    parser.add_argument("--json", action="store_true", help="machine-readable summary")
    args = parser.parse_args()

    # The in-process form of PYTHONUNBUFFERED, which cannot be set anywhere that
    # would reach this process: the interpreter consumes it at startup, before
    # .env is loaded, and `ssh box '...'` is non-interactive so no profile runs.
    # Without it the intended invocation block-buffers stdout and --follow shows
    # nothing for minutes, which is indistinguishable from a hung monitor.
    sys.stdout.reconfigure(line_buffering=True)

    while True:
        state = summarize()
        if args.json:
            print(json.dumps(state, indent=2))
        else:
            print(f"[{datetime.now(tz=timezone.utc):%Y-%m-%d %H:%M:%SZ}]")
            print(render(state=state, container=live_container() if args.logs else None))
        if not args.follow or state["remaining"] == 0:
            break
        time.sleep(args.interval)
