"""Spec injection: local composition mirrors the container-side start.sh."""

from pathlib import Path

from harness.injection import SPEC_HEADER, compose_injected_instructions

START_SH = Path("cloudbox/agents/aide-prelude/start.sh")


def test_compose_appends_spec_under_header():
    doc = compose_injected_instructions(description="the description", spec="the spec\n")
    assert doc.startswith("the description")
    assert f"\n{SPEC_HEADER}\n\n" in doc
    assert doc.endswith("the spec\n")
    assert doc.index("the description") < doc.index(SPEC_HEADER) < doc.index("the spec")


def test_start_sh_header_matches_python_mirror():
    # start.sh printf's the header with escaped newlines; keep the two in sync.
    shell_header = SPEC_HEADER.replace("\n", "\\n")
    script = START_SH.read_text()
    assert f'printf "\\n{shell_header}\\n\\n"' in script


def test_start_sh_injection_is_conditional_on_mount():
    script = START_SH.read_text()
    assert "if [ -f /home/spec/spec.md ]" in script
    assert "cat /home/spec/spec.md" in script
