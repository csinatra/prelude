# Judge validation protocol

*Added 2026-08-07, before any evaluation run. This file is not part of the
frozen rubric (`docs/JUDGE_RUBRIC.md`), which is unchanged. It defines how the
judge that applies that rubric is checked against a human reference.*

## Why this exists

Prelude's evaluation chain is closed. An LLM writes the specification, an LLM
agent acts on it, an LLM judge decides whether the agent acted on it, and the
corpus the specification draws from is itself LLM-summarized. No step touches an
outside reference point. Running more competitions or more seeds does not fix
that, because every additional run is scored by the same unvalidated judge. The
mechanistic results (H2) rest entirely on the judge's classifications being
faithful to the rubric, so those classifications need an anchor outside the
model chain.

A small human-labeled sample supplies that anchor. It costs a few hours of
review time and no API spend, which makes it the cheapest available increase in
the credibility of the headline mechanistic claim.

## Protocol

**Sample.** 20 to 30 judged flags drawn by `analysis.judge_agreement export`,
stratified over (flag category, judge classification) so all three rubric
classes appear even when one is rare. `not_acted_on` is expected to be the rare
class. Sampling is seeded, so the drawn sample is reproducible from the same
judgment file.

**Scope.** Drawn across the eval competitions' C2 runs by default, since
agreement estimated on several problems covers more of the judge's behavior than
agreement on one. Only C2 produces structured flags, so the pool is the C2 runs.

The competing pressure is reviewer cost, which is dominated by problem context
rather than artifact length: 24 items from 24 competitions means learning 24
problems, while 24 items from one means learning one and reusing it. Narrowing
to a single competition's runs across seeds is therefore the **fallback**, taken
only when the artifacts make the default untenable.

That choice is made after the eval grid completes and before any labeling
begins, which is when every run's evidence bundle exists and its size is known —
not at smoke time, when only one competition has been built. The rule is fixed
in advance and keys on measured artifact size, never on the result:

- **Default** — draw across competitions, capping the number of distinct runs so
  the artifact load stays reviewable.
- **Narrow to one competition** when the measured bundles make a multi-problem
  draw unreviewable, or when a cross-competition draw cannot populate all three
  rubric classes.
- **Narrow further** — fewer runs or a tighter chain — if even a single
  competition's bundles are too large.

A low kappa is **not** a trigger for changing scope. Rescoping because the
number disappointed would make the reported figure a product of that choice. If
scope is revised after any labeling has happened, both draws are reported, not
the second alone.

**Evidence parity.** The judge and the reviewer see the same artifacts, built
once by `analysis.judge_run.evidence_bundle()` and passed to both. This is a
correctness requirement, not a convenience: if the reviewer saw less, their
disagreement would measure the difference in what they were shown rather than
the judge's fidelity to the rubric.

The bundle is the submitted solution plus its ancestor chain in AIDE's search
tree — the lineage that produced the submission, selected structurally rather
than by any rater. Selecting logs by what the judge quoted was rejected for this
reason: such an excerpt is empty exactly when the judge saw nothing, so a
reviewer could verify evidence the judge found but never discover evidence it
missed, and agreement would be inflated rather than measured. Bounding is
unavoidable — AIDE runs at 500 steps, which fits neither a judge's context nor a
reviewer's afternoon. Discarded branches are therefore invisible to both, on the
reading that a flag addressed only in an abandoned branch did not shape the
submitted solution.

**Blinding.** The review artifact carries the flag category, the flag
explanation, and the evidence bundle. It omits the judge's classification, its
quoted evidence, its reasoning, and the run's score or medal outcome — omitted
from the file itself, so viewing source cannot reveal them. Item order within a
run is shuffled under a fixed seed, because the stratified sample is ordered by
(category, classification) and position would otherwise leak the label.

**Labeling.** The reviewer classifies each item against `docs/JUDGE_RUBRIC.md`
in a generated HTML page (`analysis.judge_review_page`), which presents a run's
artifacts alongside its flags, offers the rubric on demand, and persists labels
locally so a review can be done in sittings. The reviewer is held to the same
evidence requirement as the judge: an `acted_on` classification requires a
quoted code or log line, and without one the rubric voids it to `not_acted_on`.
That rule is enforced when the labels are read, not only in the page, so both
raters are held to it whatever produced the file.

**Scoring.** `analysis.judge_agreement score` reports percent agreement,
Cohen's kappa over the three classes, and the full confusion table. The
reviewer's quotes are retained alongside their labels: two raters can agree on a
class while pointing at different lines, which a confusion table cannot show.

**Timing.** This runs **before** the mechanistic results are written up, not
after. A low-agreement result has to be able to change how the results are
framed. Running it afterward would reduce it to a formality that can only
confirm what has already been written.

## Reporting

Report all of the following beside the mechanistic results, whatever they show:

- percent agreement and Cohen's kappa, with the sample size
- the confusion table, since which direction the judge errs in matters
  (systematically over-crediting `acted_on` would inflate H2, while
  over-calling `not_acted_on` would deflate it)
- the judge provenance recorded by `analysis.judge.judge_provenance()`: judge
  model, rubric SHA-256, and judge-prompt SHA-256, so a reported agreement
  figure is tied to the exact judge that produced it

## Interpretation, fixed in advance

Kappa thresholds are conventional rather than principled, so these bands are
stated before seeing results only to prevent a post-hoc reading:

- **kappa >= 0.6:** substantial agreement. Mechanistic results reported as
  planned, with the agreement figure quoted alongside.
- **0.4 <= kappa < 0.6:** moderate. Results reported with an explicit caveat
  that judge reliability is limited, and the confusion table is discussed rather
  than just cited.
- **kappa < 0.4:** weak. The per-flag mechanistic claims are downgraded to
  descriptive observations, and H2 is reported as untested rather than
  unsupported. The judge, not the pipeline, is the thing that failed.

At a sample of 20 to 30 the kappa estimate is itself imprecise. It is a
credibility check on the judging instrument, not a precise reliability
coefficient, and the writeup should say so.

## Known limits

This anchors the judge to a single human rater, who is also the author of the
rubric and of the system under test. That is a real limitation and is not fixed
by adding more items to the sample.

Two further asymmetries are inherent rather than oversights, and are reported
alongside the agreement figure:

- **Independence.** The rubric requires each flag to be judged independently.
  The judge satisfies this structurally — one isolated call per flag, no shared
  context. A reviewer reading a run's flags against shared artifacts is anchored
  by co-presence, and the only true fix, one flag at a time with the artifacts
  re-read each time, is the cost this design exists to avoid. Shuffling removes
  the ordering leak; it does not remove the co-presence effect.
- **Scope.** If the fallback fires and the sample comes from one competition,
  agreement cannot detect the judge behaving differently across problem types.
  Whether it fired is reported either way, since it changes what the figure
  covers.

Two extensions are on the roadmap rather than in v1 scope: an inter-model
reliability check (re-judging the same sample with a different model family,
separating rubric ambiguity from single-model idiosyncrasy) and, if this work
continues past POC, an independent second human rater.
