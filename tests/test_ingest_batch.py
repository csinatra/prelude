"""Batch-mode summary ingestion — anthropic client + add_documents mocked, no API."""

from types import SimpleNamespace

from ingest import ingest_summaries as ing

_ENTRY = {"blocks": ["import x", "model.fit()"], "competition_id": "comp-a", "kaggle_score": None}


def _succeeded(custom_id: str, text: str) -> SimpleNamespace:
    message = SimpleNamespace(content=[SimpleNamespace(text=text)])
    return SimpleNamespace(custom_id=custom_id, result=SimpleNamespace(type="succeeded", message=message))


def _errored(custom_id: str) -> SimpleNamespace:
    return SimpleNamespace(custom_id=custom_id, result=SimpleNamespace(type="errored"))


class _FakeBatches:
    def __init__(self, results):
        self._results = results
        self.created: list[int] = []

    def create(self, *, requests):
        self.created.append(len(requests))
        return SimpleNamespace(id=f"batch_{len(self.created)}")

    def retrieve(self, batch_id):
        return SimpleNamespace(processing_status="ended")

    def results(self, batch_id):
        return iter(self._results)


def _client(results=()) -> SimpleNamespace:
    return SimpleNamespace(messages=SimpleNamespace(batches=_FakeBatches(list(results))))


def test_clean_summary_strips_leading_header_only():
    assert ing._clean_summary("# Title Line\n\nThis notebook trains a CNN.") == "This notebook trains a CNN."


def test_clean_summary_leaves_prose_untouched():
    prose = "This notebook trains a CNN. It uses a 3x3 kernel."
    assert ing._clean_summary(prose) == prose


def test_clean_summary_never_eats_body():
    # a '#' mid-prose (e.g. in code-ish content) is not a leading header — keep it
    text = "This uses lr=1e-3. See section #2 for details."
    assert ing._clean_summary(text) == text


def test_clean_summary_strips_multiple_leading_headers():
    assert ing._clean_summary("## A\n### B\n\nbody text here") == "body text here"


def test_request_for_shape():
    req = ing._request_for(kaggle_id=42, entry=_ENTRY)
    assert req["custom_id"] == "nb_42"
    assert req["params"]["model"] == ing.SUMMARY_MODEL
    assert req["params"]["max_tokens"] == ing.MAX_SUMMARY_TOKENS
    assert req["params"]["system"] == ing.SUMMARY_SYSTEM
    assert "Notebook code cells:" in req["params"]["messages"][0]["content"]
    assert "model.fit()" in req["params"]["messages"][0]["content"]


def test_chunk_requests_splits_on_request_count(monkeypatch):
    monkeypatch.setattr(ing, "MAX_BATCH_REQUESTS", 2)
    reqs = [ing._request_for(kaggle_id=i, entry=_ENTRY) for i in range(5)]
    assert [len(c) for c in ing._chunk_requests(reqs)] == [2, 2, 1]


def test_submit_one_batch_per_chunk(monkeypatch):
    monkeypatch.setattr(ing, "MAX_BATCH_REQUESTS", 2)
    client = _client()
    ids = ing._submit(pending={i: _ENTRY for i in range(5)}, client=client)
    assert ids == ["batch_1", "batch_2", "batch_3"]
    assert client.messages.batches.created == [2, 2, 1]


def test_collect_upserts_succeeded_and_skips_errored(monkeypatch):
    notebooks = {
        1: {"competition_id": "comp-a", "kaggle_score": 0.9, "blocks": ["x"]},
        2: {"competition_id": "comp-a", "kaggle_score": None, "blocks": ["x"]},
    }
    client = _client([_succeeded("nb_1", "summary one"), _errored("nb_2")])
    captured: dict = {}
    monkeypatch.setattr(
        ing, "add_documents",
        lambda *, collection, ids, texts, metadatas: captured.update(ids=ids, texts=texts, metadatas=metadatas),
    )
    ing._collect(batch_ids=["batch_1"], client=client, notebooks=notebooks, collection=object())
    assert captured["ids"] == ["nb_1"]  # errored nb_2 dropped
    assert captured["texts"] == ["summary one"]
    assert captured["metadatas"][0] == {
        "kaggle_id": 1, "competition_id": "comp-a", "summary_model": ing.SUMMARY_MODEL, "kaggle_score": 0.9,
    }
