"""Optional Groq-backed intelligence.

Everything here is gated on `GROQ_API_KEY` being present. When it is missing (or
the `groq` package isn't installed), callers fall back to the deterministic
`SimplePolicy` / `mutate_config`, so the arena always runs. This module is where
the Phase 2 LLM agents live; it deliberately mirrors the deterministic
signatures so it can be swapped in without touching the engine.
"""
from __future__ import annotations

import json
import os
import urllib.error

from .actions import Action, Hold, PlaceOrder, SetHeartbeat, SetWatch
from .policies import AgentContext, AgentPolicy


def groq_available() -> bool:
    if not os.getenv("GROQ_API_KEY"):
        return False
    try:
        import groq  # noqa: F401
        return True
    except Exception:
        return False


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


# ---- Groq circuit breaker ----------------------------------------------
# An OpenRouter agent that exhausts its retries used to unconditionally fall
# back to Groq's DEFAULT_MODEL as a last resort — reasonable for ONE agent
# having a bad moment, but during a sustained OpenRouter outage (its free
# ":free" models share a single 20/min, 50/day pool across ALL of
# OpenRouter's free users — not just us) it meant every OpenRouter agent's
# traffic quietly spilled onto the SAME Groq quota a Groq-native agent (LYNX)
# depends on, defeating the whole point of putting it on a separate provider
# and taking the "safe" agent down too. This tracks whether Groq itself has
# been rate-limited recently, in-process, so that rescue can be skipped when
# it would just add to an already-struggling shared quota.
_last_groq_rate_limit: float = 0.0


def note_groq_rate_limited() -> None:
    global _last_groq_rate_limit
    import time
    _last_groq_rate_limit = time.time()


def groq_recently_rate_limited(window_s: float = 90.0) -> bool:
    import time
    return (time.time() - _last_groq_rate_limit) < window_s


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


def llm_chat(model: str, messages: list, temperature: float = 0.6,
             json_mode: bool = True) -> str:
    """Return the assistant message content. `openrouter:<model>` routes to
    OpenRouter (any frontier model); `deepseek:<model>` routes to DeepSeek's
    own paid API (not a shared free pool); anything else routes to Groq."""
    if model.startswith("openrouter:"):
        return _openrouter_chat(model[len("openrouter:"):], messages,
                                temperature, json_mode)
    if model.startswith("deepseek:"):
        return _deepseek_chat(model[len("deepseek:"):], messages,
                              temperature, json_mode)
    kwargs = {"response_format": {"type": "json_object"}} if json_mode else {}
    kwargs.update(_extra_for(model))
    try:
        resp = _client().chat.completions.create(
            model=model, messages=messages, temperature=temperature, **kwargs)
    except Exception as exc:
        if _is_retryable(exc):
            note_groq_rate_limited()
        raise
    return resp.choices[0].message.content


# Whether agents are told the evolutionary stakes. Toggle to run the same
# arena with "aware" vs "blind" agents and compare — it can cut both ways
# (sharper play, or meta-gaming the win condition). Set PIT_AGENT_AWARE=0 to
# run blind.
AGENT_AWARE = os.getenv("PIT_AGENT_AWARE", "1") not in ("0", "false", "False")

_STAKES = """This is a survival duel. Only one of you wins the round. If you \
LOSE, this strategy is retired and your lineage's next generation is rebuilt by \
mutating the WINNER's trade log — so losing means the next version of you is \
built from the playbook of whoever just beat you. You are graded on return by \
the deadline (ties break on fewer trades, then lower drawdown). Play to win \
within the stop-loss; a blown stop-loss freezes you for the rest of the round.

You can see your opponent's live return % ("opponent_return_pct") — the \
scoreboard — but NOT their trades. If they are ahead of you, you must take \
more risk to catch up; if you are comfortably ahead, protect the lead. Survive.
"""

_AGENT_SYSTEM = """You are an autonomous paper-trading agent competing in a \
head-to-head duel. You start each round with fixed capital and a deadline. Your \
goal: generate the highest return while avoiding losses. Your only hard \
constraint is a stop-loss; everything else — WHICH stocks to trade, when, how \
much, how often to check the market — is entirely your call.

You are given a `market_scan`: every stock you may trade, with its current price \
and recent momentum (ret5 = 5-bar % change, ret20 = 20-bar). Do your own \
research on it — pick whichever stocks you judge best, concentrate or diversify \
as you see fit. You are NOT limited to any shortlist; the whole scan is yours.
""" + (_STAKES if AGENT_AWARE else "") + """
You may return a JSON object with an "actions" array. Each action is one of:
  {"tool":"place_order","symbol":S,"side":"buy"|"sell","qty":N,"reason":R}
  {"tool":"set_watch","symbol":S,"trigger_type":"pct_move"|"price_above"|"price_below","threshold":N}
  {"tool":"set_heartbeat","minutes":N}
  {"tool":"hold","reason":R}
Only trade symbols in the provided universe. Buy only what your cash allows; \
sell only what you hold. Respond with JSON only, no prose."""


class LLMPolicy(AgentPolicy):
    """Groq tool-style policy. Best-effort: any error degrades to Hold."""

    def __init__(self, model: str = DEFAULT_MODEL) -> None:
        self.model = model

    def _extra(self) -> dict:
        return _extra_for(self.model)

    @staticmethod
    def _market_scan(ctx: AgentContext) -> list[dict]:
        """The agent's research surface: every tradeable stock with recent
        momentum, sorted strongest-first so the model can rank and choose."""
        def ret(hist, n):
            return round((hist[-1] / hist[-1 - n] - 1) * 100, 2) if len(hist) > n else None
        rows = []
        for sym in ctx.universe:
            h = ctx.history.get(sym, [])
            rows.append({
                "sym": sym,
                "price": round(ctx.prices.get(sym, h[-1] if h else 0.0), 2),
                "ret5": ret(h, 5),
                "ret20": ret(h, 20),
                "held": ctx.positions.get(sym, 0),
            })
        rows.sort(key=lambda r: (r["ret5"] is not None, r["ret5"] or -999), reverse=True)
        return rows

    def decide(self, ctx: AgentContext) -> list[Action]:
        try:
            payload = {
                "now": ctx.now.isoformat(),
                "market_scan": self._market_scan(ctx),
                "cash": round(ctx.cash, 2),
                "positions": ctx.positions,
                "return_pct": round(ctx.return_pct, 3),
                "opponent_return_pct": round(ctx.opponent_return_pct, 3),
                "opponent_liquidated": ctx.opponent_liquidated,
                "goal_pct": ctx.goal_pct,
                "bars_remaining": ctx.bars_remaining,
                "guidelines": ctx.guidelines,
                "your_strategy_notes": ctx.strategy_config.get("notes", ""),
            }
            resp = _client().chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": _AGENT_SYSTEM},
                    {"role": "user", "content": json.dumps(payload)},
                ],
                temperature=0.7,
                response_format={"type": "json_object"},
                **self._extra(),
            )
            data = json.loads(resp.choices[0].message.content)
            return self._parse(data)
        except Exception as exc:  # never let the LLM break a round
            return [Hold(reason=f"llm-error: {type(exc).__name__}")]

    @staticmethod
    def _parse(data: dict) -> list[Action]:
        out: list[Action] = []
        for a in data.get("actions", []):
            tool = a.get("tool")
            try:
                if tool == "place_order":
                    out.append(PlaceOrder(a["symbol"], a["side"], float(a["qty"]),
                                          reason=a.get("reason")))
                elif tool == "set_watch":
                    out.append(SetWatch(a["symbol"], a["trigger_type"],
                                        float(a["threshold"]), reason=a.get("reason")))
                elif tool == "set_heartbeat":
                    out.append(SetHeartbeat(int(a["minutes"]), reason=a.get("reason")))
                elif tool == "hold":
                    out.append(Hold(reason=a.get("reason")))
            except (KeyError, ValueError, TypeError):
                continue
        return out or [Hold(reason="no actions")]


def llm_draft_guideline(sample: list[dict], active: list[str]) -> dict | None:
    """Ask the model for at most one shared guideline, or None. Fallback-safe."""
    if not groq_available():
        return None
    try:
        prompt = {
            "recent_rounds_winner_vs_loser": sample,
            "already_active_guidelines": active,
            "instruction": (
                "You maintain a shared rulebook for a pool of trading agents. "
                "Looking at recent rounds, if — and only if — a pattern RECURS "
                "across multiple rounds (not a one-off), propose ONE new guideline "
                "to add, or one existing one to remove if it no longer holds. "
                "Return JSON {\"kind\":\"add\"|\"remove\",\"text\":\"...\",\"tag\":\"...\","
                "\"evidence\":\"...\"} or {\"kind\":\"none\"} if nothing recurs."
            ),
        }
        resp = _client().chat.completions.create(
            model=DEFAULT_MODEL,
            messages=[
                {"role": "system", "content": "You curate trading guidelines. JSON only."},
                {"role": "user", "content": json.dumps(prompt)},
            ],
            temperature=0.4,
            response_format={"type": "json_object"},
        )
        data = json.loads(resp.choices[0].message.content)
        if data.get("kind") not in ("add", "remove") or not data.get("text"):
            return None
        data.setdefault("guideline_id", None)
        data.setdefault("evidence", "llm-identified pattern")
        return data
    except Exception:
        return None


def llm_draft_guideline_from_experience(experience: list[dict],
                                        active: list[str]) -> dict | None:
    """The primary way guidelines now form: grounded in what both agents
    actually SAID about their own trades (self-reflection notes from every
    generation, both lineages), not just win/loss statistics. Simulates the
    pool "discussing" recent experience and agreeing on a lesson — one call
    plays neutral summarizer across both lineages' notes at once, since
    voting (a separate, later step) is where each lineage's own agreement or
    resistance actually gets decided.

    Returns {"kind": "add"|"remove", "practice": "good"|"bad", "text": "...",
    "tag": "...", "evidence": "..."} or None if nothing recurs yet."""
    if not groq_available() or not experience:
        return None
    try:
        prompt = {
            "both_lineages_recent_self_reflection": experience,
            "already_active_guidelines": active,
            "instruction": (
                "Two trading agents have been reflecting on their OWN trades "
                "after every round. Read both lineages' notes above. If — and "
                "only if — the SAME lesson shows up independently across "
                "multiple generations or both lineages (not a one-off from a "
                "single note), propose ONE new shared guideline: either a GOOD "
                "practice worth both agents following, or a BAD practice both "
                "should avoid. If an existing guideline no longer matches recent "
                "experience, propose removing it instead. Return JSON "
                "{\"kind\":\"add\"|\"remove\",\"practice\":\"good\"|\"bad\","
                "\"text\":\"...\",\"tag\":\"short_snake_case\",\"evidence\":\"...\"} "
                "or {\"kind\":\"none\"} if nothing recurs yet."
            ),
        }
        resp = _client().chat.completions.create(
            model=DEFAULT_MODEL,
            messages=[
                {"role": "system",
                 "content": "You summarize what a pool of trading agents has "
                            "collectively learned from their own experience. "
                            "JSON only."},
                {"role": "user", "content": json.dumps(prompt)},
            ],
            temperature=0.4,
            response_format={"type": "json_object"},
        )
        data = json.loads(resp.choices[0].message.content)
        if data.get("kind") not in ("add", "remove") or not data.get("text"):
            return None
        data.setdefault("guideline_id", None)
        data.setdefault("evidence", "llm-identified pattern across agent notes")
        data["practice"] = data.get("practice") if data.get("practice") in ("good", "bad") else "good"
        return data
    except Exception:
        return None


def llm_vote_guideline(strategy_cfg: dict, proposal: dict) -> tuple[str, str]:
    """One lineage's agent votes on a proposal. Fallback handled by caller."""
    prompt = {
        "your_strategy": strategy_cfg,
        "proposal_kind": proposal["kind"],
        "practice_type": proposal.get("practice", "good"),
        "proposed_guideline": proposal["text"],
        "instruction": ("Vote whether this shared guideline should apply to the "
                        "whole pool (including you), based on whether it matches "
                        "your own trading experience. Return JSON "
                        "{\"vote\":\"agree\"|\"disagree\",\"reasoning\":\"...\"}."),
    }
    resp = _client().chat.completions.create(
        model=DEFAULT_MODEL,
        messages=[
            {"role": "system", "content": "You vote on trading guidelines. JSON only."},
            {"role": "user", "content": json.dumps(prompt)},
        ],
        temperature=0.5,
        response_format={"type": "json_object"},
    )
    data = json.loads(resp.choices[0].message.content)
    vote = "agree" if data.get("vote") == "agree" else "disagree"
    return vote, str(data.get("reasoning", ""))[:200]


def mutate_with_llm(
    winner_cfg: dict, loser_cfg: dict, winner_trades: list[dict], seed: int = 0
) -> tuple[dict, str]:
    """LLM mutation. Falls back to the deterministic mutation on any error."""
    from .mutate import mutate_config

    if not groq_available():
        return mutate_config(winner_cfg, loser_cfg, winner_trades, seed)
    try:
        prompt = {
            "winner_strategy": winner_cfg,
            "your_current_strategy": loser_cfg,
            "winner_trade_log": winner_trades[:40],
            "instruction": (
                "You lost. Study the winner's log and produce a MUTATED version "
                "of your own strategy config — learn from them but do NOT copy "
                "exactly (an exact copy will lose to their next evolution). "
                "Return JSON: {\"config\": {...}, \"note\": \"...\"}. Keep the same "
                "field names as the configs shown."
            ),
        }
        resp = _client().chat.completions.create(
            model=DEFAULT_MODEL,
            messages=[
                {"role": "system", "content": "You evolve trading strategies. JSON only."},
                {"role": "user", "content": json.dumps(prompt)},
            ],
            temperature=0.9,
            response_format={"type": "json_object"},
        )
        data = json.loads(resp.choices[0].message.content)
        cfg = data.get("config") or {}
        if not cfg:
            raise ValueError("empty config")
        cfg.pop("_announced_heartbeat", None)
        note = "LLM: " + str(data.get("note", "mutated"))
        return cfg, note
    except Exception:
        return mutate_config(winner_cfg, loser_cfg, winner_trades, seed)
