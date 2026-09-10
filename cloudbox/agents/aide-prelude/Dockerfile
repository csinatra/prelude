FROM mlebench-env

# where to put submission.csv, will be extracted
ARG SUBMISSION_DIR
ENV SUBMISSION_DIR=${SUBMISSION_DIR}
# where to put any logs, will be extracted
ARG LOGS_DIR
ENV LOGS_DIR=${LOGS_DIR}
# where to put any code, will be extracted
ARG CODE_DIR
ENV CODE_DIR=${CODE_DIR}
# where to put any other agent-specific files, will not be necessarily extracted
ARG AGENT_DIR
ENV AGENT_DIR=${AGENT_DIR}

RUN mkdir ${LOGS_DIR} ${CODE_DIR} ${AGENT_DIR}

ARG CONDA_ENV_NAME=agent
ARG REQUIREMENTS=${AGENT_DIR}/requirements.txt

# copy just the requirements file, so that we can cache conda separately from the agent files
COPY requirements.txt ${AGENT_DIR}/requirements.txt

# Requirements for opencv
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

# create conda environment and install the requirements to it
RUN conda run -n ${CONDA_ENV_NAME} pip install -r ${AGENT_DIR}/requirements.txt && \
    conda clean -afy

# put all the agent files in the expected location
COPY . ${AGENT_DIR}

# Overlay our patched aideml Anthropic backend onto the pip-installed copy:
# implements the func_spec (function-calling) path upstream left as
# NotImplementedError, so the AIDE feedback step runs single-provider on Claude.
# Finds the installed file by import so there's no hardcoded site-packages path.
# See patches/backend_anthropic.py for the full file + what changed vs upstream.
RUN conda run -n ${CONDA_ENV_NAME} python -c \
    "import aide.backend.backend_anthropic as m, shutil; \
     shutil.copyfile('${AGENT_DIR}/patches/backend_anthropic.py', m.__file__)"

# Fix an upstream aideml bug that blinds the debug loop. filter_on_path()
# deepcopies the journal, then rebuilds the copy's node list from the ORIGINAL
# journal and blanks _term_out/exc_stack on those nodes — so a pass meant to
# sanitise the SAVED artifact destroys the LIVE one. save_run() runs after every
# step and _debug() feeds parent_node.term_out to the model, so from step 2
# onward the agent debugs with "<OMITTED>" where the traceback should be.
# Observed 2026-09-05 on uw-madison: 35 nodes, 70 calls, zero progress, every
# node failing in ~1.5s on an error the agent was never shown.
#
# This is a bug fix, not a behavioral change: it restores the information flow
# aideml intends, and it applies identically to every condition. Assert before
# replacing so a version bump fails the build rather than silently skipping it.
RUN conda run -n ${CONDA_ENV_NAME} python -c \
    "import aide.journal as m; \
     src = open(m.__file__).read(); \
     old = 'journal_copy.nodes = [n for n in journal.nodes if n.id in path]'; \
     new = 'journal_copy.nodes = [n for n in journal_copy.nodes if n.id in path]'; \
     assert src.count(old) == 1, 'aideml filter_on_path changed — re-check this patch'; \
     open(m.__file__, 'w').write(src.replace(old, new))"

# Fix a second upstream aideml bug: the REPL executes agent code with an empty
# globals dict, so `__name__` is not defined there and the lookup falls through
# to builtins, where it is 'builtins'. Every `if __name__ == "__main__":` guard
# therefore evaluates False and the program does nothing — exiting in ~2s with
# no output and no exception, which the reviewer can only read as "buggy".
# AIDE's own prompt asks for "a single-file python program that is
# self-contained and can be executed as-is", which invites exactly that idiom.
# Observed 2026-09-05 on uw-madison: 7 of 8 nodes silent, and the correlation
# with the guard was perfect — the one node without a guard was the one that ran.
# Setting __name__ makes exec match how the same file behaves under `python
# runfile.py`, which is the behavior both the prompt and the agent assume.
RUN conda run -n ${CONDA_ENV_NAME} python -c \
    "import aide.interpreter as m; \
     src = open(m.__file__).read(); \
     old = 'global_scope: dict = {}'; \
     new = 'global_scope: dict = {\"__name__\": \"__main__\"}'; \
     assert src.count(old) == 1, 'aideml _run_session changed — re-check this patch'; \
     open(m.__file__, 'w').write(src.replace(old, new))"

# Give the debug prompt the reviewer's diagnosis, not just the raw output.
# aideml reviews every node and stores the verdict on `node.analysis`, but
# `_debug` passes only the parent's code and term_out — so the agent sees the
# symptom and never the interpretation. That is fine for a crash, where the
# traceback speaks for itself, and useless for a semantic failure. Observed
# 2026-09-05 on uw-madison: five nodes ran to completion, produced valid
# submissions, and were each rejected for `val_dice=1.0000` (a degenerate
# all-empty solution). The reviewer diagnosed it correctly every time —
# "implausible, indicates data leakage or a broken validation loop" — and the
# agent was never shown that sentence, so it reproduced the same solution
# across ~7 hours of the run's 12h budget.
#
# This adds no knowledge of ours: the diagnosis is written by the same model
# from the same output, and we only stop discarding it. Anchored on the term_out
# line, which appears once and only in _debug.
RUN conda run -n ${CONDA_ENV_NAME} python -c \
    "import aide.agent as m; \
     src = open(m.__file__).read(); \
     old = 'wrap_code(parent_node.term_out, lang=\"\"),'; \
     new = old + ' \"Diagnosis of the failure\": (parent_node.analysis or \"(none recorded)\"),'; \
     assert src.count(old) == 1, 'aideml _debug prompt changed — re-check this patch'; \
     open(m.__file__, 'w').write(src.replace(old, new))"

# Give `_debug` the Memory block that `_draft` and `_improve` already receive.
# Upstream builds all three prompts the same way except for this key: `_draft`
# and `_improve` both pass "Memory": self.journal.generate_summary(), and
# `_debug` passes only the parent's code and output. So the one operator that
# dominates whenever the agent is struggling — `debug_prob` is 1.0, and the
# debug branch runs before the greedy branch — is the only one with no view of
# what the run has already learned. Verified against the installed source
# 2026-09-10: Memory-in-prompt was True/True/False for draft/improve/debug.
# Anchored on the buggy-implementation key, which appears only in `_debug`.
RUN conda run -n ${CONDA_ENV_NAME} python -c \
    "import aide.agent as m; \
     src = open(m.__file__).read(); \
     old = '\"Previous (buggy) implementation\": wrap_code(parent_node.code),'; \
     new = '\"Memory\": self.journal.generate_summary(), ' + old; \
     assert src.count(old) == 1, 'aideml _debug prompt changed — re-check this patch'; \
     open(m.__file__, 'w').write(src.replace(old, new))"

# Carry unresolved failures in the Memory block, not just successes. Multi-line
# logic (leaf filtering, dedup, cap), so it lives in a script rather than inline
# here — see patches/journal_failure_memory.py for the mechanism and the reason
# this is a design change rather than a bug fix.
RUN conda run -n ${CONDA_ENV_NAME} python ${AGENT_DIR}/patches/journal_failure_memory.py
