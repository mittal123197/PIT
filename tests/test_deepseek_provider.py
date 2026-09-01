"""DeepSeek: a cheap, paid, OpenAI-compatible provider — RONIN and VIPER's
default now, replacing OpenRouter's free (and heavily shared/rate-limited)
tier. Routes via a "deepseek:<model>" prefix, mirroring "openrouter:<model>".
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json  # noqa: E402
import pytest  # noqa: E402

from pit import forward, llm  # noqa: E402


class _FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def read(self):
        return json.dumps(self._payload).encode()

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def test_llm_chat_routes_deepseek_prefix_to_deepseek(monkeypatch):
    calls = []

    def fake_deepseek_chat(model, messages, temperature, json_mode):
        calls.append(model)
        return '{"ok": true}'

    monkeypatch.setattr(llm, "_deepseek_chat", fake_deepseek_chat)
    out = llm.llm_chat("deepseek:deepseek-chat", [{"role": "user", "content": "hi"}])
    assert out == '{"ok": true}'
    assert calls == ["deepseek-chat"]


def test_deepseek_chat_returns_content_on_success(monkeypatch):
    payload = {"choices": [{"message": {"content": '{"orders": []}'}}]}
    monkeypatch.setattr("urllib.request.urlopen", lambda *a, **k: _FakeResponse(payload))
    out = llm._deepseek_chat("deepseek-chat", [{"role": "user", "content": "hi"}],
                             0.6, True)
    assert out == '{"orders": []}'


def test_deepseek_chat_surfaces_an_error_payload(monkeypatch):
    payload = {"error": {"message": "Insufficient balance"}}
    monkeypatch.setattr("urllib.request.urlopen", lambda *a, **k: _FakeResponse(payload))
    with pytest.raises(RuntimeError) as exc_info:
        llm._deepseek_chat("deepseek-chat", [{"role": "user", "content": "hi"}],
                           0.6, True)
    assert "Insufficient balance" in str(exc_info.value)


def test_ronin_and_viper_default_to_deepseek_lynx_stays_on_groq():
    assert forward._RONIN_MODEL.startswith("deepseek:")
    assert forward._VIPER_MODEL.startswith("deepseek:")
    assert not forward._LYNX_MODEL.startswith("deepseek:")
    assert not forward._LYNX_MODEL.startswith("openrouter:")


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
