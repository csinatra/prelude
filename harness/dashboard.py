"""One page showing what the box is doing: queue, trajectory, logs.

Complements `harness.status` rather than replacing it. Status answers "is the
queue moving" in one line over ssh; this answers "what is the agent actually
doing right now", which needs the per-node trajectory and the container log side
by side — and reading a search tree in a terminal is how you miss things.

Deliberately minimal: stdlib only, one self-contained page, no framework, no
build step, no persisted state. It renders from sources that already exist (the
registry, AIDE's journal, `docker logs`) and stores nothing of its own, so it
cannot disagree with them and cannot corrupt a run. A web UI is otherwise out of
scope for this repo (CLAUDE.md); this exists because a 12h unattended run is not
observable any other way.

Usage, from the box:
    python -m harness.dashboard --serve          # regenerates on every request
    python -m harness.dashboard --out page.html  # one-shot snapshot

Then from the laptop, tunnel and open http://localhost:8000:
    ssh -N -L 8000:localhost:8000 <box>
"""

import argparse
import html
import json
import subprocess
import sys
from datetime import datetime, timezone
from functools import partial
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

from harness.batch import MLEBENCH_DIR, _node_metric
from harness.registry import load_runs
from harness.status import live_container, summarize

LOG_LINES = 60
VERBOSE_LINES = 120
CODE_CHARS = 4000
# Where aide-prelude/Dockerfile puts the agent's logs inside the container.
LOGS_DIR = "/home/logs"

# A 30s re-render is enough to see a run progressing but not to watch a step
# fail. Rather than build a streaming endpoint for that, hand over the commands
# that already stream — the container name is generated per run, so they resolve
# it by filter instead of asking anyone to copy it.
FOLLOW_COMMANDS = f"""\
# what the agent is doing (from the box)
docker exec $(docker ps -q --filter name=competition-) tail -f {LOGS_DIR}/aide.log

# why a node came back buggy — tracebacks and full model responses
docker exec $(docker ps -q --filter name=competition-) tail -f {LOGS_DIR}/aide.verbose.log

# either of the above from the laptop, without an interactive session
ssh <box> 'docker exec $(docker ps -q --filter name=competition-) tail -f {LOGS_DIR}/aide.log'"""


def _run_group_dirs() -> list[Path]:
    """mle-bench run groups, newest first. Empty off-box."""
    runs_root = MLEBENCH_DIR / "runs"
    if not runs_root.is_dir():
        return []
    groups = [path for path in runs_root.glob("*_run-group_*") if path.is_dir()]
    return sorted(groups, key=lambda path: path.stat().st_mtime, reverse=True)


def _container_file(*, container: str, path: str) -> str | None:
    """Read a file from inside a running container, or None."""
    try:
        result = subprocess.run(
            ["docker", "exec", container, "cat", path],
            capture_output=True, text=True, timeout=20, check=False,
        )
    except Exception:  # noqa: BLE001 - the page must render regardless
        return None
    return result.stdout if result.returncode == 0 and result.stdout else None


def latest_journal(*, container: str | None = None) -> tuple[str | None, dict]:
    """The most current journal available, and where it came from.

    The container is checked FIRST and the host second, because mle-bench copies
    a run's logs out only when the run ENDS. aideml does write journal.json after
    every step, but it writes it inside the container, so during the hours that
    matter the host path holds the *previous* run's journal and nothing else —
    which is worse than empty, since a stale trajectory looks like a live one.

    Returns ({}, None) rather than raising when the agent has not completed its
    first node yet, the normal state early in a run.
    """
    if container:
        raw = _container_file(container=container, path=f"{LOGS_DIR}/journal.json")
        if raw:
            try:
                return f"{container}:{LOGS_DIR}/journal.json", json.loads(raw)
            except json.JSONDecodeError:
                pass  # mid-write; try again on the next refresh
        # A live container with no journal yet means the first step has not
        # finished. Falling back to the host here would render the PREVIOUS
        # run's tree under a heading that reads as current — the exact failure
        # this function exists to avoid. Show nothing instead.
        return None, {}
    for group in _run_group_dirs():
        for journal in sorted(group.glob("**/journal.json"), key=lambda p: p.stat().st_mtime):
            try:
                return str(journal), json.loads(journal.read_text())
            except (OSError, json.JSONDecodeError):
                continue
    return None, {}


def live_progress(*, container: str | None) -> dict:
    """What the agent is doing before its first node exists.

    journal.json appears only after a step completes, and a first step that
    trains a real model runs for tens of minutes. Without this the page is blank
    for exactly the period when someone is most likely to be checking whether
    anything is wrong. aide.log carries the step-by-step narration and the token
    side-channel carries truncation, both written continuously.
    """
    if not container:
        return {}
    log = _container_file(container=container, path=f"{LOGS_DIR}/aide.log") or ""
    # aide.log narrates which node is drafting or executing; the traceback that
    # explains WHY a node came back buggy is only in the verbose log, and
    # journal.json does not carry it either (aideml serializes term_out empty).
    # Without this pane a run of failing nodes is visible but unattributable.
    verbose = _container_file(container=container, path=f"{LOGS_DIR}/aide.verbose.log") or ""
    usage_raw = _container_file(container=container, path=f"{LOGS_DIR}/prelude_token_usage.jsonl")
    calls, truncated = 0, 0
    for line in (usage_raw or "").splitlines():
        try:
            call = json.loads(line)
        except json.JSONDecodeError:
            continue
        calls += 1
        truncated += call.get("stop_reason") == "max_tokens"
    return {
        "aide_log": "\n".join(log.splitlines()[-25:]),
        "verbose_log": "\n".join(verbose.splitlines()[-VERBOSE_LINES:]),
        "calls": calls,
        "truncated": truncated,
    }


def host_metrics() -> dict:
    """GPU, load, memory and disk for the box.

    Surfaced because resource state is how you tell a long step from a stuck one,
    and the two look identical in a log. A step that has been "executing" for
    forty minutes at 0% GPU is either preprocessing on one core or wedged; the
    numbers distinguish those, and nothing else on this page does.

    nvidia-smi is a provisioning prerequisite (setup_cloudbox.sh preflights it),
    so its absence means something is wrong with the box rather than with this.
    """
    metrics: dict[str, str] = {}
    try:
        gpu = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10, check=False,
        ).stdout.strip()
        if gpu:
            used, total, temp = (part.strip() for part in gpu.split(",")[1:])
            metrics["gpu"] = f"{gpu.split(',')[0].strip()}% util · {used}/{total} MiB · {temp}°C"
    except Exception:  # noqa: BLE001 - metrics must never break the page
        metrics["gpu"] = "unavailable"
    try:
        metrics["load"] = Path("/proc/loadavg").read_text().split(" ")[0]
        mem = {
            line.split(":")[0]: int(line.split()[1])
            for line in Path("/proc/meminfo").read_text().splitlines()[:3]
        }
        used_gb = (mem["MemTotal"] - mem["MemAvailable"]) / 1024 / 1024
        metrics["memory"] = f"{used_gb:.0f}/{mem['MemTotal'] / 1024 / 1024:.0f} GiB"
    except Exception:  # noqa: BLE001
        pass
    try:
        disk = subprocess.run(
            ["df", "-h", "--output=pcent,avail", str(Path.home())],
            capture_output=True, text=True, timeout=10, check=False,
        ).stdout.splitlines()
        if len(disk) > 1:
            metrics["disk"] = " ".join(disk[1].split())
    except Exception:  # noqa: BLE001
        pass
    return metrics


def container_logs(*, container: str | None, lines: int = LOG_LINES) -> str:
    if not container:
        return ""
    try:
        result = subprocess.run(
            ["docker", "logs", "--tail", str(lines), container],
            capture_output=True, text=True, timeout=15, check=False,
        )
    except Exception as exc:  # noqa: BLE001 - a log tail must never break the page
        return f"(could not read logs: {exc})"
    return (result.stdout + result.stderr)[-20000:]


def collect() -> dict:
    """Everything the page shows, gathered from sources that already exist."""
    container = live_container()
    journal_path, journal = latest_journal(container=container)
    nodes = journal.get("nodes", []) if isinstance(journal, dict) else []
    return {
        "generated_at": datetime.now(tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ"),
        "queue": summarize(),
        "runs": sorted(load_runs().values(), key=lambda run: run.get("run_key", "")),
        "container": container,
        "journal_path": journal_path,
        "nodes": nodes,
        "live": live_progress(container=container),
        "host": host_metrics(),
        "logs": container_logs(container=container),
    }


def _fmt(value: object, digits: int = 4) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _node_rows(*, nodes: list[dict]) -> str:
    rows = []
    for index, node in enumerate(nodes):
        value, _ = _node_metric(node=node)
        buggy = bool(node.get("is_buggy"))
        code = (node.get("code") or "")[:CODE_CHARS]
        summary = (node.get("analysis") or node.get("plan") or "").strip()
        rows.append(
            f"""<tr class="{'buggy' if buggy else 'ok'}">
    <td>{node.get('step', index)}</td>
    <td>{'buggy' if buggy else 'ok'}</td>
    <td class="num">{_fmt(value)}</td>
    <td class="num">{_fmt(node.get('exec_time'), 1)}s</td>
    <td class="wrap">{html.escape(summary[:400])}</td>
  </tr>
  <tr class="code-row"><td colspan="5"><details><summary>solution</summary>
    <pre>{html.escape(code)}</pre></details></td></tr>"""
        )
    return "\n".join(rows) or '<tr><td colspan="5">No nodes yet.</td></tr>'


def _run_rows(*, runs: list[dict]) -> str:
    rows = []
    for run in runs:
        state = "abandoned" if run.get("abandoned") else run.get("status", "?")
        rows.append(
            f"""<tr class="{state}">
    <td>{html.escape(str(run.get('run_key', '')))}</td>
    <td>{html.escape(str(run.get('condition', '')))}</td>
    <td>{html.escape(state)}</td>
    <td class="num">{_fmt(run.get('score'))}</td>
    <td class="num">{_fmt(run.get('leaderboard_percentile'), 3)}</td>
    <td class="num">{_fmt(run.get('agent_steps'))}</td>
    <td class="wrap">{html.escape(str(run.get('last_error') or ''))[:300]}</td>
  </tr>"""
        )
    return "\n".join(rows) or '<tr><td colspan="7">Registry is empty.</td></tr>'


def _live_block(*, live: dict) -> str:
    """What the agent is doing right now, for the long gap before node 1 exists."""
    if not live:
        return ""
    truncated = live["truncated"]
    # Truncation is called out because it is otherwise invisible: it does not
    # error, it degrades the response into unparseable code several steps later.
    warn = (
        f' <strong class="warn">{truncated} truncated at max_tokens</strong>'
        if truncated
        else ""
    )
    return f"""<h2>Live</h2>
<div class="meta">{live['calls']} agent calls so far{warn}</div>
<pre class="tail">{html.escape(live['aide_log']) or '(no aide.log yet)'}</pre>
<details><summary>verbose log — last {VERBOSE_LINES} lines (tracebacks, model responses)</summary>
<pre class="tail">{html.escape(live['verbose_log']) or '(no aide.verbose.log yet)'}</pre></details>"""


def render(*, state: dict) -> str:
    queue = state["queue"]
    stalled = " — STALLED" if queue["stalled"] else ""
    return f"""<!doctype html>
<meta charset="utf-8"><title>prelude runs</title>
<meta http-equiv="refresh" content="30">
<style>
 body {{ font: 13px/1.5 ui-monospace, SFMono-Regular, Menlo, monospace;
        margin: 1.5rem; background: #fbfbfa; color: #1a1a19; }}
 h1 {{ font-size: 1.1rem; margin: 0 0 .2rem; }}
 h2 {{ font-size: .95rem; margin: 1.6rem 0 .4rem; }}
 .meta {{ color: #6b6b68; margin-bottom: 1rem; }}
 table {{ border-collapse: collapse; width: 100%; }}
 th, td {{ text-align: left; padding: .25rem .5rem; border-bottom: 1px solid #e6e6e3;
           vertical-align: top; }}
 th {{ color: #6b6b68; font-weight: 600; }}
 .num {{ text-align: right; font-variant-numeric: tabular-nums; }}
 .wrap {{ max-width: 46ch; overflow-wrap: anywhere; }}
 tr.buggy td {{ background: #fdf0ee; }}
 tr.graded td {{ background: #f0f7f0; }}
 tr.abandoned td {{ background: #fdf0ee; }}
 tr.code-row td {{ border-bottom: 2px solid #e6e6e3; }}
 pre {{ background: #f4f4f1; padding: .6rem; overflow-x: auto; max-height: 22rem; }}
 summary {{ cursor: pointer; color: #6b6b68; }}
 .warn {{ color: #a3341f; }}
</style>
<h1>prelude — {html.escape(str(queue['stage']))}{stalled}</h1>
<div class="meta">
  {state['generated_at']} · graded {queue['graded']}/{queue['total']} ·
  remaining {queue['remaining']} ·
  container {html.escape(state['container'] or 'none')}
</div>
<div class="meta">
  {' · '.join(f'{k} {html.escape(v)}' for k, v in state['host'].items()) or 'no host metrics'}
</div>

<h2>Runs</h2>
<table>
 <tr><th>run</th><th>cond</th><th>status</th><th>score</th><th>pctile</th>
     <th>steps</th><th>error</th></tr>
 {_run_rows(runs=state['runs'])}
</table>

{_live_block(live=state['live'])}

<h2>Trajectory{' — ' + html.escape(state['journal_path']) if state['journal_path'] else ''}</h2>
<table>
 <tr><th>step</th><th>state</th><th>metric</th><th>exec</th><th>analysis</th></tr>
 {_node_rows(nodes=state['nodes'])}
</table>

<h2>Container log (last {LOG_LINES} lines)</h2>
<pre class="tail">{html.escape(state['logs']) or '(no live container)'}</pre>

<h2>Follow live from a terminal</h2>
<div class="meta">This page re-renders every 30s. To watch a log as it is written:</div>
<pre>{html.escape(FOLLOW_COMMANDS)}</pre>

<script>
 // Every pane above is a tail, so the newest lines are at the BOTTOM — but a
 // scrollable <pre> opens at the top, which shows the oldest lines and reads as
 // a stalled run. Scroll each one to its end after the page renders.
 for (const pane of document.querySelectorAll('pre.tail')) {{
   pane.scrollTop = pane.scrollHeight;
 }}
</script>
"""


class _Handler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler's interface
        # Rendered per request rather than cached: the page's whole purpose is to
        # reflect a run that is still moving.
        body = render(state=collect()).encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args: object) -> None:
        """Silence per-request logging; the page auto-refreshes every 30s."""


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--serve", action="store_true", help="serve, regenerating per request")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--out", type=Path, help="write one snapshot and exit")
    args = parser.parse_args()

    if args.out:
        args.out.write_text(render(state=collect()))
        print(f"wrote {args.out}", flush=True)
    elif args.serve:
        sys.stdout.reconfigure(line_buffering=True)
        # Bound to localhost deliberately: reach it over an ssh tunnel rather
        # than opening a port on a box that holds API keys.
        print(f"serving on http://localhost:{args.port}  (ssh -N -L {args.port}:localhost:{args.port} <box>)")
        HTTPServer(("127.0.0.1", args.port), partial(_Handler)).serve_forever()
    else:
        print(render(state=collect()))
