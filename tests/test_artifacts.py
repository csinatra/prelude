"""save_artifacts writes the run directory including the provenance manifest."""

import json

from analysis import artifacts


def test_save_artifacts_writes_manifest(tmp_path, monkeypatch):
    monkeypatch.setattr(artifacts, "RESULTS_DIR", tmp_path)
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    monkeypatch.setenv("MODEL", "claude-haiku-4-5-20251001")

    run_dir = artifacts.save_artifacts(
        competition_id="spooky-author-identification",
        condition="C2",
        seed=0,
        spec_document="# spec",
        retrievals=[],
        pipeline_output={"stage_trace": []},
    )

    assert (run_dir / "spec.md").read_text() == "# spec"
    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert manifest["competition_id"] == "spooky-author-identification"
    assert manifest["condition"] == "C2"
    assert manifest["seed"] == 0
    assert manifest["model"] == "claude-haiku-4-5-20251001"
    assert manifest["llm_provider"] == "anthropic"
    # Repo is a git checkout, so provenance must resolve to a real commit.
    assert isinstance(manifest["git_commit"], str) and len(manifest["git_commit"]) == 40
    assert isinstance(manifest["git_dirty"], bool)
    assert manifest["saved_at"]


def test_git_provenance_handles_missing_git(monkeypatch):
    def raise_missing(*args, **kwargs):
        raise FileNotFoundError("git not found")

    monkeypatch.setattr(artifacts.subprocess, "run", raise_missing)
    assert artifacts._git_provenance() == {"git_commit": None, "git_dirty": None}
