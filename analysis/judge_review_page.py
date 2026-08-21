"""Render the blinded review sample as a self-contained HTML page.

A markdown file asks the reviewer to hold a long solution, a multi-node
trajectory, and the rubric's three classes in their head at once while scrolling
a text editor. At 24 items that is where labeling errors come from, and a
labeling error is indistinguishable from judge disagreement in the final kappa.

The page addresses that with layout rather than with less evidence: items are
grouped by run so each run's artifacts are read once and reused across its
flags, the artifacts sit alongside the flag being judged rather than above it,
and labels are one click. Labels persist to localStorage, so a review can be
done in sittings, and are exported as JSON for `judge_agreement score`.

Blinding is structural here rather than procedural: the judge's classification
and the run's outcome are never written into the file, so viewing source cannot
reveal what a reviewer is asked not to look up.

No external assets — inline CSS/JS only, so the page opens from disk and can be
archived beside the results it validates.
"""

import hashlib
import html
import io
import json
import keyword
import random
import tokenize

CLASSES = ("not_acted_on", "acted_on_unclear", "acted_on_positive")
SHUFFLE_SEED = 0

_CSS = """
:root { --bg:#fff; --fg:#1a1a1a; --muted:#666; --line:#e2e2e2; --accent:#2b6cb0;
        --code-bg:#f6f8fa; --buggy:#c53030;
        --kw:#a626a4; --str:#50a14f; --com:#a0a1a7; --num:#986801; --bi:#0184bc; }
@media (prefers-color-scheme: dark) {
  :root { --bg:#16181d; --fg:#e6e6e6; --muted:#9aa0a6; --line:#2c2f36;
          --accent:#7aa7dd; --code-bg:#1e2128; --buggy:#f08a8a;
          --kw:#c678dd; --str:#98c379; --com:#7f848e; --num:#d19a66; --bi:#56b6c2; }
}
* { box-sizing: border-box; }
body { margin:0; background:var(--bg); color:var(--fg); font:15px/1.55 -apple-system,
       BlinkMacSystemFont,"Segoe UI",Helvetica,Arial,sans-serif; }
header { position:sticky; top:0; z-index:5; background:var(--bg);
         border-bottom:1px solid var(--line); padding:12px 20px; display:flex;
         gap:16px; align-items:center; flex-wrap:wrap; }
header h1 { font-size:16px; margin:0; font-weight:600; }
#progress { color:var(--muted); font-variant-numeric:tabular-nums; }
button { font:inherit; padding:6px 12px; border:1px solid var(--line);
         background:var(--code-bg); color:var(--fg); border-radius:6px; cursor:pointer; }
button:hover { border-color:var(--accent); }
main { padding:20px; max-width:1400px; margin:0 auto; }
.intro { color:var(--muted); max-width:70ch; margin-bottom:24px; }
.run { border:1px solid var(--line); border-radius:10px; margin-bottom:28px; overflow:hidden; }
.run > summary { padding:12px 16px; cursor:pointer; font-weight:600;
                 background:var(--code-bg); }
.split { display:grid; grid-template-columns:minmax(0,1fr) minmax(0,1fr); gap:0; }
@media (max-width:900px) { .split { grid-template-columns:minmax(0,1fr); } }
.artifacts { border-right:1px solid var(--line); padding:16px; min-width:0;
             position:sticky; top:56px; align-self:start; max-height:calc(100vh - 76px);
             overflow:auto; }
@media (max-width:900px) { .artifacts { position:static; max-height:none;
                                        border-right:none; border-bottom:1px solid var(--line); } }
.flags { padding:16px; min-width:0; }
h3 { font-size:13px; text-transform:uppercase; letter-spacing:.04em;
     color:var(--muted); margin:0 0 8px; }
pre { background:var(--code-bg); border:1px solid var(--line); border-radius:6px;
      padding:10px; overflow:auto; font:12px/1.5 ui-monospace,SFMono-Regular,Menlo,monospace;
      max-height:60vh; }
.node > summary { cursor:pointer; padding:6px 0; font:12px/1.4 ui-monospace,monospace;
                  color:var(--muted); }
.node.buggy > summary { color:var(--buggy); }
.item { border-top:1px solid var(--line); padding:16px 0; }
.item:first-of-type { border-top:none; padding-top:0; }
.cat { font:12px ui-monospace,monospace; color:var(--accent); }
.labels { display:flex; gap:8px; flex-wrap:wrap; margin-top:10px; }
.labels label { border:1px solid var(--line); border-radius:6px; padding:6px 10px;
                cursor:pointer; font:13px ui-monospace,monospace; }
.labels input { margin-right:6px; }
.labels label:has(input:checked) { border-color:var(--accent);
                                   background:color-mix(in srgb, var(--accent) 14%, transparent); }
.t-kw { color:var(--kw); } .t-str { color:var(--str); } .t-com { color:var(--com); font-style:italic; }
.t-num { color:var(--num); } .t-bi { color:var(--bi); }
.src { font:11px ui-monospace,monospace; text-transform:uppercase; letter-spacing:.04em;
       color:var(--muted); margin:8px 0 2px; }
#runs { display:flex; flex-wrap:wrap; gap:8px; margin-bottom:24px; }
.nav-run { font:12px ui-monospace,monospace; }
.nav-run span { color:var(--muted); }
.nav-run.done { border-color:var(--accent); }
.nav-run.done span { color:var(--accent); }
textarea.quote { width:100%; margin-top:8px; padding:8px; border:1px solid var(--line);
                 border-radius:6px; background:var(--code-bg); color:var(--fg); resize:vertical;
                 font:12px/1.5 ui-monospace,SFMono-Regular,Menlo,monospace; }
.item.needs-quote textarea.quote { border-color:var(--buggy); }
.item.needs-quote .labels::after { content:'quote required — counts as not_acted_on without one';
                                   color:var(--buggy); font-size:12px; align-self:center; }
#rubric { position:fixed; right:20px; bottom:20px; width:min(520px, 92vw); height:min(70vh, 640px);
          background:var(--bg); border:1px solid var(--line); border-radius:10px; z-index:20;
          box-shadow:0 8px 32px rgba(0,0,0,.28); display:none; flex-direction:column; resize:both;
          overflow:hidden; }
#rubric.open { display:flex; }
#rubric .bar { display:flex; justify-content:space-between; align-items:center; gap:12px;
               padding:10px 14px; border-bottom:1px solid var(--line); background:var(--code-bg); }
#rubric .bar strong { font-size:13px; }
#rubric .bar code { font-size:10px; color:var(--muted); }
#rubric pre { margin:0; border:none; border-radius:0; max-height:none; flex:1;
              white-space:pre-wrap; word-wrap:break-word; font-size:12px; }
"""

_JS = """
const KEY = 'prelude-judge-review';
const store = JSON.parse(localStorage.getItem(KEY) || '{}');
const total = document.querySelectorAll('.item').length;

function needsQuote(entry) {
  return entry && entry.label && entry.label !== 'not_acted_on' && !(entry.quote || '').trim();
}

function save() {
  localStorage.setItem(KEY, JSON.stringify(store));
  refresh();
}

function isDone(entry) {
  // Complete means the rubric would accept it: an acted_on_* without a quote is
  // void, so it does not count as done.
  return Boolean(entry && entry.label) && !needsQuote(entry);
}

function refresh() {
  const entries = Object.values(store).filter(entry => entry && entry.label);
  const pending = entries.filter(needsQuote).length;
  const done = entries.length - pending;
  document.getElementById('progress').textContent =
    `${done} / ${total} labeled` + (pending ? ` · ${pending} missing a quote` : '');
  document.querySelectorAll('.item').forEach(node => {
    node.classList.toggle('needs-quote', needsQuote(store[node.dataset.item]));
  });
  document.querySelectorAll('.run').forEach(run => {
    const items = run.querySelectorAll('.item');
    const complete = [...items].filter(item => isDone(store[item.dataset.item])).length;
    const badge = document.querySelector(`[data-count="${run.dataset.run}"]`);
    if (badge) badge.textContent = `${complete}/${items.length}`;
    const button = document.querySelector(`.nav-run[data-target="${run.dataset.run}"]`);
    if (button) button.classList.toggle('done', complete === items.length);
  });
}

// One run open at a time: a cross-competition sample would otherwise land the
// reviewer on every solution and trajectory at once.
document.querySelectorAll('.nav-run').forEach(button => {
  button.addEventListener('click', () => {
    const target = document.querySelector(`.run[data-run="${button.dataset.target}"]`);
    document.querySelectorAll('.run').forEach(run => { run.open = run === target; });
    target.scrollIntoView({behavior: 'smooth', block: 'start'});
  });
});

const firstRun = document.querySelector('.run');
if (firstRun) firstRun.open = true;

document.querySelectorAll('input[type=radio]').forEach(input => {
  const entry = store[input.name];
  if (entry && entry.label === input.value) input.checked = true;
  input.addEventListener('change', () => {
    store[input.name] = {...(store[input.name] || {}), label: input.value};
    save();
  });
});

document.querySelectorAll('textarea.quote').forEach(area => {
  const entry = store[area.dataset.for];
  if (entry && entry.quote) area.value = entry.quote;
  area.addEventListener('input', () => {
    store[area.dataset.for] = {...(store[area.dataset.for] || {}), quote: area.value};
    save();
  });
});

document.getElementById('export').addEventListener('click', () => {
  const blob = new Blob([JSON.stringify(store, null, 2)], {type: 'application/json'});
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = 'judge_review_labels.json';
  a.click();
});

document.getElementById('copy').addEventListener('click', async () => {
  await navigator.clipboard.writeText(JSON.stringify(store, null, 2));
  document.getElementById('copy').textContent = 'copied';
  setTimeout(() => { document.getElementById('copy').textContent = 'copy labels'; }, 1200);
});

const panel = document.getElementById('rubric');
document.getElementById('rubric-toggle').addEventListener('click', () => {
  panel.classList.toggle('open');
});
document.getElementById('rubric-close').addEventListener('click', () => {
  panel.classList.remove('open');
});
document.addEventListener('keydown', event => {
  if (event.key === 'Escape') panel.classList.remove('open');
});

refresh();
"""


def _escape(text: str) -> str:
    return html.escape(text or "")


def _highlight(*, source: str) -> str:
    """Syntax-highlight Python with the stdlib tokenizer.

    Render-time rather than a client-side library: the page must open from disk
    with no external assets, and shipping a highlighter's JS would dwarf the
    content it highlights.

    Agent code is frequently unparseable — a buggy node's snippet may be
    truncated mid-statement — so any tokenizer error falls back to plain escaped
    text. Highlighting is a reading aid; failing to render the code would not be.
    """
    if not source.strip():
        return ""
    try:
        tokens = list(tokenize.generate_tokens(io.StringIO(source).readline))
    except (tokenize.TokenError, IndentationError, SyntaxError):
        return _escape(source)

    out: list[str] = []
    row, col = 1, 0
    lines = source.splitlines(keepends=True)
    for token in tokens:
        start_row, start_col = token.start
        # Replay whitespace and anything the tokenizer skips, so the rendered
        # source is byte-faithful to what the judge was given.
        while (row, col) < (start_row, start_col):
            line = lines[row - 1] if row - 1 < len(lines) else ""
            if row < start_row:
                out.append(_escape(line[col:]))
                row, col = row + 1, 0
            else:
                out.append(_escape(line[col:start_col]))
                col = start_col
        css = _token_class(token=token)
        text = _escape(token.string)
        out.append(f"<span class='{css}'>{text}</span>" if css else text)
        row, col = token.end
    return "".join(out)


def _token_class(*, token: tokenize.TokenInfo) -> str:
    if token.type == tokenize.COMMENT:
        return "t-com"
    if token.type == tokenize.STRING:
        return "t-str"
    if token.type == tokenize.NUMBER:
        return "t-num"
    if token.type == tokenize.NAME:
        if keyword.iskeyword(token.string):
            return "t-kw"
        if token.string in ("self", "cls") or token.string in dir(__builtins__):
            return "t-bi"
    return ""


def _node_block(*, node: dict) -> str:
    label = (
        f"step {node.get('step', '?')} · "
        f"{'buggy' if node.get('is_buggy') else 'ok'} · metric={node.get('metric')}"
    )
    body = _escape(node.get("analysis", ""))
    output = _escape(node.get("term_out", ""))
    # The two fields differ in evidentiary weight and the rubric depends on the
    # difference: `analysis` is the agent narrating its own step, which the rubric
    # calls inadmissible on its own for acted_on_positive, while `term_out` is
    # observed. Presenting them as undifferentiated prose invites a rater to
    # credit fluent self-narration as evidence.
    return (
        f"<details class='node{' buggy' if node.get('is_buggy') else ''}'>"
        f"<summary>{_escape(label)}</summary>"
        f"<p class='src'>agent self-report — not evidence on its own</p><p>{body}</p>"
        f"<p class='src'>observed output</p><pre>{output}</pre></details>"
    )


def _item_block(*, item: dict) -> str:
    """One flag with its label control. Carries no classification — see blinding."""
    item_id = _escape(item["item_id"])
    radios = "".join(
        f"<label><input type='radio' name='{item_id}' value='{name}'>{name}</label>"
        for name in CLASSES
    )
    # The rubric requires a quote for any acted_on_* classification and voids the
    # classification without one. judge.py enforces that on the LLM; holding the
    # human to a weaker standard would mean the judge's positive labels survived
    # an evidence check the human's never faced.
    return (
        f"<div class='item' data-item='{item_id}'>"
        f"<div class='cat'>{_escape(item['category'])}</div>"
        f"<p>{_escape(item['explanation'])}</p>"
        f"<div class='labels'>{radios}</div>"
        f"<textarea class='quote' data-for='{item_id}' rows='2' "
        f"placeholder='Quote the code or log line supporting this (required for acted_on_*)'>"
        f"</textarea></div>"
    )


def _run_block(*, run_key: str, items: list[dict], nodes: list[dict]) -> str:
    solution = _highlight(source=items[0].get("solution_excerpt", ""))
    chain = "".join(_node_block(node=node) for node in nodes) or "<p>(no trajectory preserved)</p>"
    flags = "".join(_item_block(item=item) for item in items)
    return (
        f"<details class='run' id='run-{_escape(run_key)}' data-run='{_escape(run_key)}'>"
        f"<summary>{_escape(run_key)} — {len(items)} flag(s)</summary>"
        f"<div class='split'>"
        f"<div class='artifacts'><h3>Submitted solution</h3><pre>{solution}</pre>"
        f"<h3>Trajectory (ancestor chain)</h3>{chain}</div>"
        f"<div class='flags'><h3>Flags to classify</h3>{flags}</div>"
        f"</div></details>"
    )


def _rubric_panel(*, rubric_text: str) -> str:
    """The frozen rubric, on call, as preformatted text.

    Deliberately not markdown-rendered: a hand-rolled converter that silently
    mangles a clause would misinform a rater about a frozen document, and the
    rubric's own SHA-256 is shown so the page states which version was applied.
    """
    if not rubric_text:
        return ""
    digest = hashlib.sha256(rubric_text.encode()).hexdigest()
    return (
        f"<div id='rubric'><div class='bar'><strong>docs/JUDGE_RUBRIC.md (frozen)</strong>"
        f"<code>{digest[:12]}</code>"
        f"<button id='rubric-close'>close</button></div>"
        f"<pre>{_escape(rubric_text)}</pre></div>"
    )


def render_review_page(
    *,
    sample: list[dict],
    nodes_by_run: dict[str, list[dict]] | None = None,
    rubric_text: str = "",
) -> str:
    """Blinded HTML review page, items grouped by run so artifacts are read once."""
    nodes_by_run = nodes_by_run or {}
    grouped: dict[str, list[dict]] = {}
    for item in sample:
        grouped.setdefault(item.get("run_key", "unknown"), []).append(item)
    # stratified_sample emits round-robin over (category, classification), so
    # sample order correlates with the judged class — position would leak the
    # label the page otherwise withholds. Shuffle within each run, seeded so the
    # rendered page is reproducible from the same judgments file.
    for items in grouped.values():
        random.Random(SHUFFLE_SEED).shuffle(items)
    runs = "".join(
        _run_block(run_key=run_key, items=items, nodes=nodes_by_run.get(run_key, []))
        for run_key, items in grouped.items()
    )
    # A cross-competition sample spans several runs, each carrying a solution and
    # a trajectory. Without an index the reviewer scrolls past finished problems
    # to reach the current one; per-run counts also show which problem is done,
    # which the global count cannot.
    nav = "".join(
        f"<button class='nav-run' data-target='{_escape(run_key)}'>"
        f"{_escape(run_key)} <span data-count='{_escape(run_key)}'></span></button>"
        for run_key in grouped
    )
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Judge validation review</title><style>{_CSS}</style></head>
<body>
<header>
  <h1>Judge validation review (blinded)</h1>
  <span id="progress"></span>
  <button id="rubric-toggle">rubric</button>
  <button id="export">download labels</button>
  <button id="copy">copy labels</button>
</header>
{_rubric_panel(rubric_text=rubric_text)}
<main>
  <p class="intro">Classify each flag against <code>docs/JUDGE_RUBRIC.md</code> using the same
  three classes and the same evidence requirement the judge was held to — an
  <code>acted_on_*</code> classification needs a quoted code or log line, and without one the
  rubric voids it to <code>not_acted_on</code>. The judge's
  classification and the run's score are deliberately absent from this file — do not look them
  up before finishing, since anchoring defeats the purpose. Labels save as you go; download them
  when done and score with <code>python -m analysis.judge_agreement score</code>.</p>
  <nav id="runs">{nav}</nav>
  {runs}
</main>
<script>{_JS}</script>
</body></html>
"""


def parse_labels(*, text: str) -> dict[str, str]:
    """Read {item_id: label} from the page's exported JSON.

    Applies the rubric's evidence requirement exactly as `judge.judge_flags` does
    to the LLM: an `acted_on_*` label with no quote is void and becomes
    `not_acted_on`. Enforcing it here rather than only in the page means a
    hand-edited export cannot slip past the rule either, so both raters are held
    to the same standard whatever produced the file.
    """
    parsed: dict[str, str] = {}
    for key, value in json.loads(text).items():
        # A bare string is the pre-quote export shape; treat it as quoteless.
        entry = {"label": value} if isinstance(value, str) else (value or {})
        label = entry.get("label")
        if label not in CLASSES:
            continue
        if label != "not_acted_on" and not str(entry.get("quote", "")).strip():
            label = "not_acted_on"
        parsed[key] = label
    return parsed
