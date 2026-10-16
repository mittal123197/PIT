"""Model providers behind one call: `llm_chat(model, messages)`.

Agents name their model with a provider prefix — `anthropic:` (Claude),
`gemini:`, `deepseek:`, `openrouter:`, `ollama:` (local) — or a bare id, which
goes to Groq. `llm_available(model)` says whether that provider is configured
(its API key is set), so callers never gate on one provider's key for all.
"""
from __future__ import annotations

import json
import os
import time
import urllib.error


def groq_available() -> bool:
    if not os.getenv("GROQ_API_KEY"):
        return False
    try:
        import groq  # noqa: F401
        return True
    except Exception:
        return False


_PROVIDER_KEYS = {"anthropic:": "ANTHROPIC_API_KEY", "gemini:": "GEMINI_API_KEY",
                  "deepseek:": "DEEPSEEK_API_KEY", "openrouter:": "OPENROUTER_API_KEY"}


def llm_available(model: str | None) -> bool:
    """Is the provider behind `model` configured? Local Ollama always is."""
    model = model or DEFAULT_MODEL
    if model.startswith("ollama:"):
        return True
    for prefix, key in _PROVIDER_KEYS.items():
        if model.startswith(prefix):
            return bool(os.getenv(key))
    return groq_available()


def _client():
    import groq
    return groq.Groq(api_key=os.getenv("GROQ_API_KEY"))


DEFAULT_MODEL = os.getenv("PIT_LLM_MODEL", "openai/gpt-oss-20b")


def _extra_for(model: str) -> dict:
    """gpt-oss models support a reasoning_effort knob; low keeps latency down on
    the many calls an agent makes per round."""
    if "gpt-oss" in model:
        return {"reasoning_effort": os.getenv("PIT_LLM_REASONING", "low")}
    return {}


# ---- unified chat: Groq, or any OpenAI-compatible provider (OpenRouter) ----

def _is_retryable(exc) -> bool:
    s = f"{type(exc).__name__} {exc}".lower()
    return any(k in s for k in ("429", "ratelimit", "rate limit", "timeout",
                                "temporarily", "overloaded", "503"))


def _openrouter_post(model: str, messages: list, temperature: float,
                     use_json_mode: bool) -> str:
    import urllib.request
    body = {"model": model, "messages": messages, "temperature": temperature}
    if use_json_mode:
        body["response_format"] = {"type": "json_object"}
    req = urllib.request.Request(
        "https://openrouter.ai/api/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Authorization": f"Bearer {os.getenv('OPENROUTER_API_KEY', '')}",
                 "Content-Type": "application/json",
                 "X-Title": "PIT Arena"})
    with urllib.request.urlopen(req, timeout=90) as r:
        data = json.loads(r.read().decode())
    if "error" in data:
        # OpenRouter sometimes returns HTTP 200 with an error PAYLOAD instead
        # of an HTTP error status (e.g. "Upstream error from Nvidia: Service
        # temporarily overloaded") — indexing straight into ["choices"] here
        # used to surface as a bare, unrecognizable KeyError that _is_retryable
        # couldn't pattern-match on. Surface the real message so retry/backoff
        # (and the Groq fallback on the last attempt) actually kicks in.
        err = data["error"]
        msg = err.get("message", str(err)) if isinstance(err, dict) else str(err)
        raise RuntimeError(f"OpenRouter error: {msg}")
    return data["choices"][0]["message"]["content"]


def _openrouter_chat(model: str, messages: list, temperature: float,
                     json_mode: bool) -> str:
    try:
        return _openrouter_post(model, messages, temperature, json_mode)
    except urllib.error.HTTPError as exc:
        # Some free/community models on OpenRouter don't support the
        # structured-outputs feature — fall back to plain prompting (the
        # system prompt already asks for JSON-only) rather than failing.
        if json_mode and exc.code == 400:
            body = exc.read().decode() if hasattr(exc, "read") else ""
            if "structured-output" in body or "response_format" in body:
                return _openrouter_post(model, messages, temperature, False)
        raise


# DeepSeek — cheap, paid, OpenAI-compatible. Not a shared/rate-limited free
# pool like OpenRouter's ":free" models: it's your own metered account, so
# it doesn't compete with anyone else's traffic. See DEEPSEEK_API_KEY.
def _deepseek_chat(model: str, messages: list, temperature: float,
                   json_mode: bool) -> str:
    import urllib.request
    body = {"model": model, "messages": messages, "temperature": temperature}
    if json_mode:
        body["response_format"] = {"type": "json_object"}
    req = urllib.request.Request(
        "https://api.deepseek.com/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Authorization": f"Bearer {os.getenv('DEEPSEEK_API_KEY', '')}",
                 "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=90) as r:
        data = json.loads(r.read().decode())
    if "error" in data:
        err = data["error"]
        msg = err.get("message", str(err)) if isinstance(err, dict) else str(err)
        raise RuntimeError(f"DeepSeek error: {msg}")
    return data["choices"][0]["message"]["content"]


# Ollama — local, no key, no cost, no network. Uses Ollama's OpenAI-compatible
# endpoint (same request/response shape as DeepSeek's above). Slower than a
# hosted API on modest hardware and the timeout is set accordingly. PIT_OLLAMA_HOST
# lets this point at a non-default host/port if Ollama isn't on 127.0.0.1:11434.
def _ollama_chat(model: str, messages: list, temperature: float,
                 json_mode: bool) -> str:
    import urllib.request
    host = os.getenv("PIT_OLLAMA_HOST", "http://localhost:11434")
    body = {"model": model, "messages": messages, "temperature": temperature,
           "stream": False}
    if json_mode:
        body["response_format"] = {"type": "json_object"}
    req = urllib.request.Request(
        f"{host.rstrip('/')}/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:   # queued behind other agents on a single-slot server
        data = json.loads(r.read().decode())
    if "error" in data:
        err = data["error"]
        msg = err.get("message", str(err)) if isinstance(err, dict) else str(err)
        raise RuntimeError(f"Ollama error: {msg}")
    return data["choices"][0]["message"]["content"]


# Claude — Anthropic's own API via the official SDK. ANTHROPIC_API_KEY in .env.
# Current models reject non-default sampling (temperature) and assistant
# prefill, so JSON is asked for in the system prompt and the object is pulled
# out of the reply. Every call's token usage is appended to
# data/claude_usage.jsonl so a small prepaid credit can be watched.
_CLAUDE_PRICES = {  # $ per 1M tokens (input, output)
    "claude-sonnet-5-5": (2.00, 10.00),
    "claude-haiku-5-5": (0.10, 0.50),
    "claude-opus-5-5": (4.00, 20.00),
}


def _anthropic_chat(model: str, messages: list, temperature: float,
                    json_mode: bool) -> str:
    import anthropic
    system = "\n\n".join(m["content"] for m in messages if m["role"] == "system")
    if json_mode:
        system += ("\n\nRespond with exactly one JSON object and nothing else "
                   "— no prose, no code fences.")
    convo = [{"role": m["role"], "content": m["content"]}
             for m in messages if m["role"] != "system"]
    client = anthropic.Anthropic(max_retries=2, timeout=120.0)
    resp = client.messages.create(
        model=model, max_tokens=16000, system=system, messages=convo,
        output_config={"effort": os.getenv("PIT_CLAUDE_EFFORT", "medium")})
    if resp.stop_reason == "refusal":
        raise RuntimeError(f"Claude refused: {getattr(resp.stop_details, 'category', None)}")
    _log_claude_usage(model, resp.usage)
    text = "".join(b.text for b in resp.content if b.type == "text").strip()
    if json_mode and text and not text.startswith("{"):
        i, j = text.find("{"), text.rfind("}")
        if i != -1 and j > i:
            text = text[i:j + 1]
    return text


def _log_claude_usage(model: str, usage) -> None:
    try:
        from pathlib import Path
        inp, out = _CLAUDE_PRICES.get(model, (0.0, 0.0))
        cost = (usage.input_tokens * inp + usage.output_tokens * out) / 1e6
        path = Path(__file__).resolve().parent.parent / "data" / "claude_usage.jsonl"
        with path.open("a") as f:
            f.write(json.dumps({"ts": time.strftime("%Y-%m-%d %H:%M:%S"), "model": model,
                                "in": usage.input_tokens, "out": usage.output_tokens,
                                "usd": round(cost, 5)}) + "\n")
    except Exception:
        pass


def claude_spend() -> float:
    """Total estimated $ spent on Claude calls so far (from the usage log)."""
    from pathlib import Path
    path = Path(__file__).resolve().parent.parent / "data" / "claude_usage.jsonl"
    try:
        return round(sum(json.loads(l)["usd"] for l in path.read_text().splitlines() if l), 4)
    except Exception:
        return 0.0


# Gemini — Google's OpenAI-compatible endpoint. GEMINI_API_KEY in .env.
# JSON mode is requested; if the endpoint rejects it, the call is retried
# without it and the JSON object is pulled out of the reply.
_GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"


def _gemini_chat(model: str, messages: list, temperature: float,
                 json_mode: bool) -> str:
    import urllib.error
    import urllib.request

    def _post(body):
        req = urllib.request.Request(
            _GEMINI_URL, data=json.dumps(body).encode(),
            headers={"Authorization": f"Bearer {os.getenv('GEMINI_API_KEY', '')}",
                     "Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=120) as r:
            return json.loads(r.read().decode())

    body = {"model": model, "messages": messages, "temperature": temperature}
    if json_mode:
        body["response_format"] = {"type": "json_object"}
    data = None
    # free-tier keys often get 503 ("high demand") / 429 (rate limit): brief
    # backoff and retry instead of losing the whole decision
    for attempt in range(3):
        try:
            data = _post(body)
            break
        except urllib.error.HTTPError as e:
            if e.code == 400 and json_mode and "response_format" in body:
                body.pop("response_format", None)
                continue
            if e.code in (429, 503) and attempt < 2:
                time.sleep(4 * (attempt + 1))
                continue
            raise RuntimeError(f"Gemini HTTP {e.code}: {e.read().decode()[:300]}")
    if isinstance(data, list):
        data = data[0]
    if "error" in data:
        raise RuntimeError(f"Gemini error: {data['error']}")
    text = (data["choices"][0]["message"].get("content") or "").strip()
    if json_mode and text and not text.startswith("{"):
        i, j = text.find("{"), text.rfind("}")
        if i != -1 and j > i:
            text = text[i:j + 1]
    return text


def gemini_models() -> list[str]:
    """Model ids this key can use (for picking one)."""
    import urllib.request
    req = urllib.request.Request(
        "https://generativelanguage.googleapis.com/v1beta/openai/models",
        headers={"Authorization": f"Bearer {os.getenv('GEMINI_API_KEY', '')}"})
    with urllib.request.urlopen(req, timeout=30) as r:
        data = json.loads(r.read().decode())
    return sorted(m["id"].replace("models/", "") for m in data.get("data", []))


def llm_chat(model: str, messages: list, temperature: float = 0.6,
             json_mode: bool = True) -> str:
    """Return the assistant message content. `openrouter:<model>` routes to
    OpenRouter (any frontier model); `deepseek:<model>` routes to DeepSeek's
    own paid API (not a shared free pool); `ollama:<model>` routes to a local
    Ollama server; `anthropic:<model>` routes to Claude; anything else routes
    to Groq."""
    if model.startswith("openrouter:"):
        return _openrouter_chat(model[len("openrouter:"):], messages,
                                temperature, json_mode)
    if model.startswith("deepseek:"):
        return _deepseek_chat(model[len("deepseek:"):], messages,
                              temperature, json_mode)
    if model.startswith("ollama:"):
        return _ollama_chat(model[len("ollama:"):], messages,
                            temperature, json_mode)
    if model.startswith("gemini:"):
        return _gemini_chat(model[len("gemini:"):], messages, temperature, json_mode)
    if model.startswith("anthropic:"):
        return _anthropic_chat(model[len("anthropic:"):], messages,
                               temperature, json_mode)
    kwargs = {"response_format": {"type": "json_object"}} if json_mode else {}
    kwargs.update(_extra_for(model))
    resp = _client().chat.completions.create(
        model=model, messages=messages, temperature=temperature, **kwargs)
    return resp.choices[0].message.content


# Whether agents are told the evolutionary stakes. Toggle to run the same
# arena with "aware" vs "blind" agents and compare — it can cut both ways
# (sharper play, or meta-gaming the win condition). Set PIT_AGENT_AWARE=0 to
# run blind.
AGENT_AWARE = os.getenv("PIT_AGENT_AWARE", "1") not in ("0", "false", "False")
