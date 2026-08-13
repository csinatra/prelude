# Progress log

History extracted from CLAUDE.md ("What done looks like for the current
phase") on 2026-07-19. Entries are verbatim and in original order; append
new milestones at the bottom. Current state and next steps live in
CLAUDE.md.

## Completed milestones

- ✅ Toy two-stage pipeline runs end-to-end against the real Anthropic API
  and produces a LangSmith trace.
- ✅ Real four-stage chain (`parse_problem → surface_signals →
  flag_assumptions → advise_approach`) built and wired in `pipeline/graph.py`,
  runs end-to-end via `pipeline/condition_c2.py`.
- ✅ Backend is swappable (`LLM_PROVIDER=anthropic|ollama`) via
  `pipeline/llm_client.py`, verified against a local Ollama model.
- ✅ Unit tests (`tests/test_pipeline.py`) cover each node plus a full run,
  with `call_llm` mocked — no real API calls.
- ✅ Stage outputs are Pydantic models (`pipeline/schemas.py`), enforced via
  native structured outputs — `messages.parse(output_format=...)` on
  Anthropic, JSON-schema `format` on Ollama. Verified end-to-end against
  both backends.
- ✅ RAG layer built: `ingest/` package (Code4ML + mle-bench descriptions →
  two ChromaDB collections), `pipeline/retriever.py` seam with leave-one-out
  filter, one directed retrieval per stage wired into `pipeline/nodes.py`,
  schemas/prompts refit for MLE-bench competition descriptions. Unit-tested
  (retrieval + LLM mocked); voyage-code-3 verified live.
- ✅ Dev-subset corpus ingested and live-verified: 1,005 metadata chunks,
  62,379 practitioner chunks; four-stage run on spooky-author-identification
  (Haiku, 54s) with leave-one-out held in all four retrievals.
- ✅ Design-review refactor: conditions renamed/extended to B1/B2/C1/C2
  (see docs/RESEARCH_DESIGN.md); SpecificationFlag/Recommendation schemas
  with evidence citation and flag linkage; two-level (notebook→chunk)
  retrieval as the shared practitioner-knowledge unit; analysis scaffold
  (frozen judge rubric, artifact preservation); pipeline/config.py central
  tunables.
- ✅ Full dev corpus: notebook_summaries complete (5,937 abstracts, pinned
  Haiku, `summary_model`-stamped). Observability: prompts, resolved model,
  and token usage traced at the `llm_client` seam; bulk ingest and tests
  deliberately untraced (LangSmith quota). Provenance: manifest.json per
  run artifact (git commit, model, condition coordinates).
- ✅ Spec-side harness (`harness/`): per-condition spec.md renderer
  (additive composition — each grid step changes exactly one thing),
  append-only runs.jsonl registry (status lifecycle spec_built →
  agent_run → graded, mergeable across machines), CLI runner with
  block/synthesis token split. Live-verified: all four conditions on
  spooky-author-identification, leave-one-out held, zero dangling flag
  references.
- ✅ Threshold calibration: 10-competition sweep (`analysis/calibration.py`,
  re-runnable). Decision, recorded pre-run in RESEARCH_DESIGN.md:
  `SIMILARITY_THRESHOLD=None` for v1 — stage-directed queries score ~0.06
  lower than flat against the same chunks, so any global cutoff breaks
  knowledge parity; top-k rank ordering is the quality control.
- ✅ Cloud-box provisioned and verified on a Lambda A10.
  `scripts/setup_cloudbox.sh` encodes the full bring-up (Python 3.11, Sysbox
  runtime, git-lfs, mle-bench at the pinned commit, base + agent image builds,
  spec-mount patch). AIDE agent pinned to Haiku 4.5 on aideml v6.3.3 — pristine
  except an `httpx<0.28` env pin and an Anthropic function-calling backend
  patch, so the whole AIDE loop runs single-provider on Claude.
- ✅ Smoke run GREEN end-to-end on `random-acts-of-pizza` (off-eval): Haiku
  drove AIDE to valid graded submissions — A-style 0.608 and a C2-spec-mounted
  0.637, both above median — exercising provision → spec inject → AIDE → grade.
- ✅ B/C spec injection confirmed on the box: `PRELUDE_SPEC_PATH` mounts the
  spec at `/home/spec/spec.md`, appended as ADVISOR CONTEXT, with the C2
  structured content reaching AIDE's prompt. Batch driver (`harness/batch.py`)
  drains the run queue (park-and-manual-resubmit on failure) and preserves
  agent artifacts (submission, journal, best solution, per-call token log) onto
  the persistent volume, so `--terminate-on-done` can't lose the
  mechanistic-judge inputs.
- ✅ Dev corpus rebuilt under the generalized summary prompt and
  `voyage-4-large`: 5,937/5,937 notebook summaries via the Message Batches API,
  zero errors, ~$16.54 measured ($0.0030/notebook). Zero leading markdown
  headers and zero truncations at scale, `summary_model` homogeneous across the
  collection. `competition_metadata` re-embedded (1,005 chunks);
  `practitioner_knowledge` dropped as unqueried.
