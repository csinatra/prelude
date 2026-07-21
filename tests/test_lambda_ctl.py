"""Lambda terminate helper: request shape (URL, method, auth, body) — HTTP mocked."""

import json
from contextlib import contextmanager

from harness import lambda_ctl


def test_terminate_instance_posts_expected_request(monkeypatch):
    captured = {}

    @contextmanager
    def fake_urlopen(request):
        captured["url"] = request.full_url
        captured["method"] = request.get_method()
        captured["auth"] = request.get_header("Authorization")
        captured["body"] = json.loads(request.data)

        class _Resp:
            def read(self):
                return json.dumps({"data": {"terminated_instances": [{"id": "i-1"}]}}).encode()

        yield _Resp()

    monkeypatch.setattr(lambda_ctl.urllib.request, "urlopen", fake_urlopen)

    result = lambda_ctl.terminate_instance(instance_id="i-1", api_key="secret")

    assert captured["url"] == lambda_ctl.TERMINATE_URL
    assert captured["method"] == "POST"
    assert captured["auth"] == "Bearer secret"
    assert captured["body"] == {"instance_ids": ["i-1"]}
    assert result["data"]["terminated_instances"][0]["id"] == "i-1"


def test_api_key_falls_back_to_env(monkeypatch):
    monkeypatch.setenv("LAMBDA_API_KEY", "from-env")
    captured = {}

    @contextmanager
    def fake_urlopen(request):
        captured["auth"] = request.get_header("Authorization")

        class _Resp:
            def read(self):
                return b"{}"

        yield _Resp()

    monkeypatch.setattr(lambda_ctl.urllib.request, "urlopen", fake_urlopen)
    lambda_ctl.terminate_instance(instance_id="i-1")
    assert captured["auth"] == "Bearer from-env"
