# Acknowledgments

Prelude builds on the following external work. Where this repository adapts or
redistributes third-party code, the upstream license and notice are reproduced
below.

## MLE-bench (OpenAI)

- Repository: https://github.com/openai/mle-bench (pinned commit
  `507f92e1138bb6e40dac5c6ee7a6758e6424bf97`)
- `cloudbox/agents/aide-prelude/` is **adapted from** MLE-bench's
  `agents/aide/` directory at the pinned commit. The only behavioral change
  from upstream is the spec-injection block in `start.sh` (appending a mounted
  `/home/spec/spec.md` as an ADVISOR CONTEXT section); the Dockerfile,
  `requirements.txt`, and `additional_notes.txt` are otherwise upstream.

MLE-bench is MIT-licensed. The upstream copyright notice and permission notice,
reproduced verbatim:

```
MIT License

Copyright (c) 2024 OpenAI

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

NOTE: This license applies to the code in this repository, but not the external datasets and files that may be downloaded while using this package.
```

## AIDE (Weco AI)

- Original project: AIDE by Weco AI (https://github.com/WecoAI/aideml).
- Prelude does not select an AIDE fork independently. The AIDE agent runs
  against `github.com/thesofakillers/aideml` (pinned `v6.3.3`) **because that is
  the exact fork MLE-bench itself designates for its AIDE agent** — see the
  agents table in MLE-bench's `agents/README.md` at the pinned commit
  (https://github.com/openai/mle-bench/blob/507f92e1138bb6e40dac5c6ee7a6758e6424bf97/agents/README.md),
  which lists `[AIDE] | aide | https://github.com/thesofakillers/aideml` and
  notes "We slightly modified each agent to elicit better capabilities."
  Inheriting MLE-bench's designated fork keeps our baseline comparable to
  published MLE-bench AIDE results.

## Code4ML

- Practitioner-notebook corpus: Code4ML (Zenodo record 6607065,
  https://zenodo.org/record/6607065). Used to build the retrieval corpus
  (`ingest/`). Refer to the Zenodo record for the dataset's own license and
  citation terms.

## Behavioral guidelines (CLAUDE.md)

- The "Behavioral guidelines" section of `CLAUDE.md` is adapted from
  [multica-ai/andrej-karpathy-skills](https://github.com/multica-ai/andrej-karpathy-skills/blob/main/CLAUDE.md).
