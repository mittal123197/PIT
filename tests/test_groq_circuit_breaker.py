"""An OpenRouter agent that exhausts its retries used to unconditionally fall
back to Groq's DEFAULT_MODEL — reasonable for one agent's bad moment, but
during a sustained OpenRouter outage (its free ":free" models share ONE
rate-limit pool across all of OpenRouter's free users) every OpenRouter
agent's failure piled onto the same Groq quota a Groq-native agent depends
on, spreading one provider's outage onto both. The circuit breaker tracks
whether Groq itself was recently rate-limited and skips the rescue if so.
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest  # noqa: E402

from pit import autonomous, llm  # noqa: E402


def test_breaker_is_closed_by_default():
    llm._last_groq_rate_limit = 0.0
    assert llm.groq_recently_rate_limited() is False


def test_note_rate_limited_trips_the_breaker():
    llm.note_groq_rate_limited()
    assert llm.groq_recently_rate_limited() is True


def test_breaker_expires_after_the_window():
    llm._last_groq_rate_limit = time.time() - 100
    assert llm.groq_recently_rate_limited(window_s=90) is False


def test_llm_chat_trips_the_breaker_on_a_groq_rate_limit(monkeypatch):
    llm._last_groq_rate_limit = 0.0

    class _FakeCompletions:
        def create(self, **kwargs):
            raise RuntimeError("429 rate limit exceeded")

    class _FakeChat:
        completions = _FakeCompletions()

    class _FakeClient:
        chat = _FakeChat()

    monkeypatch.setattr(llm, "_client", lambda: _FakeClient())
    with pytest.raises(RuntimeError):
        llm.llm_chat("openai/gpt-oss-20b", [{"role": "user", "content": "hi"}])
    assert llm.groq_recently_rate_limited() is True


def test_decide_skips_the_groq_rescue_when_groq_is_already_struggling(monkeypatch):
    """An OpenRouter agent's exhausted retries must NOT fall back to Groq
    while the breaker is tripped — it should just fail that turn instead of
    adding to an already-struggling shared quota."""
    llm.note_groq_rate_limited()  # breaker is tripped
    calls = []

    def fake_llm_chat(model, messages, temperature=0.6, json_mode=True):
        calls.append(model)
        raise RuntimeError("429 rate limit exceeded")

    monkeypatch.setattr(autonomous, "llm_chat", fake_llm_chat)
    monkeypatch.setattr(autonomous, "groq_available", lambda: True)
    monkeypatch.setattr(autonomous.time, "sleep", lambda s: None)
    monkeypatch.setattr(autonomous, "groq_available", lambda: True)
    # groq_available() being True normally; the breaker being tripped is what
    # must stop the rescue
    view = {"cash": 100000.0, "positions": [], "total_value": 100000.0,
            "return_pct": 0.0, "opponent_return_pct": 0.0, "notes": ""}
    orders, notes, message, trace = autonomous.decide(
        view, 1, 0, 1.0, [], model="openrouter:some/model:free")

    # every attempt stayed on the original OpenRouter model — never rescued
    # onto DEFAULT_MODEL (a bare, non-"openrouter:" name) mid-flight
    assert all(c == "openrouter:some/model:free" for c in calls)
    assert orders == []


def test_decide_still_rescues_to_groq_when_groq_is_healthy(monkeypatch):
    llm._last_groq_rate_limit = 0.0  # breaker NOT tripped
    calls = []

    def fake_llm_chat(model, messages, temperature=0.6, json_mode=True):
        calls.append(model)
        raise RuntimeError("429 rate limit exceeded")

    monkeypatch.setattr(autonomous, "llm_chat", fake_llm_chat)
    monkeypatch.setattr(autonomous, "groq_available", lambda: True)
    monkeypatch.setattr(autonomous.time, "sleep", lambda s: None)
    view = {"cash": 100000.0, "positions": [], "total_value": 100000.0,
            "return_pct": 0.0, "opponent_return_pct": 0.0, "notes": ""}
    autonomous.decide(view, 1, 0, 1.0, [], model="openrouter:some/model:free")

    # the LAST attempt should have switched to the plain DEFAULT_MODEL rescue
    assert calls[-1] == autonomous.DEFAULT_MODEL


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
