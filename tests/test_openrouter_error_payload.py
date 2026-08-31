"""OpenRouter sometimes returns HTTP 200 with an error PAYLOAD instead of an
HTTP error status (e.g. an upstream model being temporarily overloaded).
Indexing straight into ["choices"] used to surface as a bare, unrecognizable
KeyError that the retry logic's keyword matching couldn't catch — silently
killing that turn instead of retrying or falling back to Groq.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json  # noqa: E402
import pytest  # noqa: E402

from pit import llm  # noqa: E402


class _FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def read(self):
        return json.dumps(self._payload).encode()

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def test_error_payload_raises_a_message_containing_the_real_reason(monkeypatch):
    payload = {"error": {"message": "Upstream error from Nvidia: Service "
                                    "temporarily overloaded", "code": 502}}
    monkeypatch.setattr("urllib.request.urlopen", lambda *a, **k: _FakeResponse(payload))

    with pytest.raises(RuntimeError) as exc_info:
        llm._openrouter_post("some/model", [{"role": "user", "content": "hi"}],
                             0.6, True)
    assert "temporarily overloaded" in str(exc_info.value)


def test_that_error_message_is_recognized_as_retryable():
    exc = RuntimeError("OpenRouter error: Upstream error from Nvidia: "
                       "Service temporarily overloaded")
    assert llm._is_retryable(exc) is True


def test_normal_success_payload_still_returns_the_content(monkeypatch):
    payload = {"choices": [{"message": {"content": '{"ok": true}'}}]}
    monkeypatch.setattr("urllib.request.urlopen", lambda *a, **k: _FakeResponse(payload))
    out = llm._openrouter_post("some/model", [{"role": "user", "content": "hi"}],
                               0.6, True)
    assert out == '{"ok": true}'


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
