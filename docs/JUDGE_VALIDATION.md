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

**Blinding.** The exported review file contains the flag category, the flag
explanation, and the solution excerpt. It deliberately omits the judge's
classification and the run's score or medal outcome. Seeing the judge's label
first would anchor the reviewer and turn agreement into a measure of
suggestibility. Seeing the score would violate the same blinding the rubric
already imposes on the judge.

**Labeling.** The reviewer classifies each item against `docs/JUDGE_RUBRIC.md`,
using the same three classes and the same evidence requirement, and writes the
label on the item's `- human:` line.

**Scoring.** `analysis.judge_agreement score` reports percent agreement,
Cohen's kappa over the three classes, and the full confusion table.

**Timing.** This runs **before** the mechanistic results are written up, not
after. A low-agreement result has to be able to change how the results are
framed. Running it afterward would reduce it to a formality that can only
confirm what has already been written.

## Reporting

Report all of the following beside the mechanistic results, whatever they show:

- percent agreement and Cohen's kappa, with the sample size
- the confusion table, since which direction the judge errs in matters
  (systematically over-crediting `acted_on_positive` would inflate H2, while
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
by adding more items to the sample. Two extensions are on the roadmap rather
than in v1 scope: an inter-model reliability check (re-judging the same sample
with a different model family, separating rubric ambiguity from single-model
idiosyncrasy) and, if this work continues past POC, an independent second human
rater.
