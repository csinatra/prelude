"""Corpus-representation probe (2026-08-03): does adding LLM-curated code cells
add transferable signal over a rich whole-notebook summary?

Motivation: cell-level practitioner retrieval surfaced import/boilerplate noise
(see the B2 spec inspection that prompted this), and the redesign injects the
notebook summary anyway. So the live question for the retrieval unit is: with a
detailed whole-notebook summary in hand, do curated code cells earn their token
cost? This probe makes one Haiku enriched-card call per notebook
{abstract, key_cell_indices, why}, then renders summary-alone vs summary+cells
for human judgement, with token-cost and cell-position stats.

Offline representation analysis only — no spec-time retrieval, no Chroma write,
no eval model. Mirrors the ingest summarization step (same operation, generic
per-notebook, not competition-conditioned), so it sits inside the same
leave-one-out regime as corpus build. Pins Haiku (dev model), tracing off.
Excludes the reserved smoke competition (random-acts-of-pizza).

Result + decision: notebooks/probe_summary_vs_cells.md; docs/DECISIONS.md
(2026-08-03 retrieval-unit entry); RESEARCH_DESIGN.md corpus/retrieval section.

Run: PYTHONPATH=. .venv/bin/python -m analysis.probe_representation
(needs ANTHROPIC_API_KEY in the environment or .env).
"""
import os
from collections import defaultdict
from pathlib import Path

import anthropic
import pandas as pd
from pydantic import BaseModel

from ingest.config import RAW_DIR

os.environ["LANGSMITH_TRACING"] = "false"
os.environ["LANGSMITH_TRACING_V2"] = "false"

HAIKU = "claude-haiku-4-5-20251001"
MAX_NOTEBOOK_CHARS = 60_000
FILES = ["code_blocks_upto_20.csv", "code_blocks_21.csv"]
OUT = Path("notebooks/probe_summary_vs_cells.md")

# One notebook per competition, spanning NLP / tabular / vision task families.
TARGETS = [
    "spooky-author-identification",
    "jigsaw-toxic-comment-classification-challenge",
    "text-normalization-challenge-english-language",
    "new-york-city-taxi-fare-prediction",
    "tabular-playground-series-may-2022",
    "nomad2018-predict-transparent-conductors",
    "dogs-vs-cats-redux-kernels-edition",
    "aerial-cactus-identification",
]

SYSTEM = (
    "You are building a retrieval card for a solution notebook so an engineer facing a "
    "DIFFERENT but related ML problem can learn from it. Return: (1) `abstract` — a detailed "
    "technical summary of the notebook's approach, emphasizing what transfers across problems "
    "of this class: the modeling approach (specific estimators/architectures and notable "
    "hyperparameter or configuration choices), feature engineering and data transformations "
    "(name the concrete derived features/representations), validation strategy (resampling "
    "scheme and target metric), preprocessing, and any distinctive or failure-mode-avoiding "
    "techniques. Plain prose, no headers, no competition/leaderboard framing. (2) "
    "`key_cell_indices` — the indices of the few code cells (if any) that carry reusable "
    "executable specifics NOT fully conveyable in prose (exact configs, non-obvious transforms, "
    "validation setup); EXCLUDE imports, boilerplate, EDA/plotting, and print-only cells. (3) "
    "`why_these_cells` — one sentence on what those cells add beyond the abstract."
)


class ProbeCard(BaseModel):
    abstract: str
    key_cell_indices: list[int]
    why_these_cells: str


def load_targets() -> dict[int, dict]:
    nb: dict[int, dict] = {}
    for filename in FILES:
        path = RAW_DIR / "code4ml" / filename
        for frame in pd.read_csv(path, index_col=0, chunksize=200_000):
            frame = frame[frame["data_sources"].isin(TARGETS)]
            for _, r in frame.iterrows():
                kid = int(r["kaggle_id"])
                e = nb.setdefault(kid, {
                    "competition": str(r["data_sources"]),
                    "score": float(r["kaggle_score"]) if pd.notna(r["kaggle_score"]) else None,
                    "cells": [],
                })
                e["cells"].append((int(r["code_block_id"]), str(r["code_block"])))
    for e in nb.values():
        e["cells"].sort(key=lambda c: c[0])  # notebook order
        e["blocks"] = [c[1] for c in e["cells"]]
    return nb


def pick_one_per_comp(nb: dict[int, dict]) -> dict[str, int]:
    """Substantive notebook per competition: total chars in [8k,60k]; highest score, then most cells."""
    by_comp: dict[str, list[int]] = defaultdict(list)
    for kid, e in nb.items():
        total = sum(len(b) for b in e["blocks"])
        if 8_000 <= total <= MAX_NOTEBOOK_CHARS and len(e["blocks"]) >= 6:
            by_comp[e["competition"]].append(kid)
    chosen: dict[str, int] = {}
    for comp in TARGETS:
        cands = by_comp.get(comp, [])
        if not cands:
            continue
        cands.sort(key=lambda k: (nb[k]["score"] or -1, len(nb[k]["blocks"])), reverse=True)
        chosen[comp] = cands[0]
    return chosen


def card_for(blocks: list[str], client: anthropic.Anthropic) -> ProbeCard:
    numbered = "\n\n".join(f"[{i}]\n{b}" for i, b in enumerate(blocks))[:MAX_NOTEBOOK_CHARS]
    resp = client.messages.parse(
        model=HAIKU, max_tokens=1200, system=SYSTEM,
        messages=[{"role": "user", "content": f"Notebook cells:\n{numbered}"}],
        output_format=ProbeCard,
    )
    return resp.parsed_output


def main() -> None:
    nb = load_targets()
    chosen = pick_one_per_comp(nb)
    print(f"selected {len(chosen)} notebooks")
    client = anthropic.Anthropic()
    out = ["# Probe: rich summary alone vs. summary + curated cells\n",
           "One Haiku enriched-card call per notebook. For each: read the **summary alone** "
           "first, then ask whether the **selected cells** add transferable specifics it missed.\n"]
    for comp, kid in chosen.items():
        e = nb[kid]
        blocks = e["blocks"]
        card = card_for(blocks, client)
        n = len(blocks)
        sel = [i for i in card.key_cell_indices if 0 <= i < n]
        pos = [i / (n - 1) for i in sel] if n > 1 else [0.0]
        sel_chars = sum(len(blocks[i]) for i in sel)
        out += [
            f"\n\n## {comp} — nb {kid}  (score={e['score']}, {n} cells)",
            f"\n*selected {len(sel)}/{n} cells; mean position {sum(pos)/len(pos):.2f} (0=start,1=end); "
            f"summary ~{len(card.abstract)//4} tok vs selected cells ~{sel_chars//4} tok*\n",
            f"\n**Summary (summary-alone condition):**\n\n{card.abstract}\n",
            f"\n**Why cells add value (LLM):** {card.why_these_cells}\n",
            f"\n**Selected cells (indices {sel}):**\n",
        ]
        for i in sel:
            out.append(f"\n```python\n# cell [{i}] of {n}\n{blocks[i]}\n```")
    OUT.write_text("\n".join(out))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
