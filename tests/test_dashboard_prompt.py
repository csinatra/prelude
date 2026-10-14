"""Dashboard's current-prompt pane: latest Memory-onward block of the verbose log."""

from harness.dashboard import latest_prompt_from_memory

LOG = """\
[2026-10-14 18:00:00,000] INFO: system: # Introduction
...
# Task description
long problem statement
# Memory
old memory
# Instructions
- <TOTAL_TIME_REMAINING: 9.0hrs>
[2026-10-14 18:00:05,000] INFO: response: plan + code
[2026-10-14 19:00:00,000] INFO: system: # Introduction
# Task description
long problem statement
# Memory
Failed attempt (not yet fixed — do not repeat): key mismatch
# Previous (buggy) implementation
code
# Instructions
- <TOTAL_TIME_REMAINING: 7.0hrs>
- <TOTAL_STEPS_REMAINING: 491>
[2026-10-14 19:00:05,000] INFO: response: fixed code
"""


def test_returns_only_the_latest_prompt_from_memory_on():
    prompt = latest_prompt_from_memory(verbose=LOG)
    assert prompt.splitlines()[0] == "# Memory"
    assert "key mismatch" in prompt and "TOTAL_STEPS_REMAINING: 491" in prompt
    assert "old memory" not in prompt
    assert "Task description" not in prompt and "response:" not in prompt


def test_empty_when_no_prompt_logged_yet():
    assert latest_prompt_from_memory(verbose="[2026-10-14] INFO: Starting run\n") == ""
