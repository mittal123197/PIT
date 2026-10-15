"""Strategy charters: each lineage writes — and evolves — its OWN trading style.

Same model family + same data table + same incentives pushed all three agents
into the same momentum style (book overlap up to 70-100%), so rounds were
decided by sizing noise rather than by different ideas. Instead of assigning
styles (that would be our bias), each lineage chooses one itself: a short
charter it sees at every decision, plus a checkable version of it (entry and
exit conditions over the data table's columns). The arena backtests that
spec on ~60 days of market history (pit/history.py) and hands the agent the
result — before the round it may refine the charter once from that card;
after each round it keeps / refines / rewrites it from its own results.
"""
from __future__ import annotations

import json

MAX_CHARS = 520

_SPEC_HELP = (
    "Also give a CHECKABLE version of your charter: \"entry\": 1-4 conditions "
    "that must ALL hold to buy, \"exit\": 1-4 conditions where ANY one means sell "
    "(exits may also use pos_pnl% = the position's P&L vs entry and held_min = "
    "minutes held), each as [column, op, value] with op one of <, <=, >, >= and "
    "column exactly as named in the data table (e.g. [\"vsVWAP%\", \"<\", -1.5]); "
    "and \"max_hold_min\". The arena backtests this spec on ~60 days of real "
    "market data and shows you the result.")


def _clean(text: str) -> str:
    from .forward import _redact_tickers
    return _redact_tickers(" ".join(str(text or "").split()))[:MAX_CHARS]


def _conds(raw, action: str, dropped: list) -> list:
    """Valid [column, op, value] conditions; unknown columns are dropped (and
    reported back to the agent) instead of sinking the whole spec."""
    from .guidelines import parse_spec
    out = []
    for c in (raw or [])[:6]:
        one = parse_spec([c], action)
        if one:
            out.append(one["when"][0])
        else:
            dropped.append(str(c[0] if isinstance(c, (list, tuple)) and c else c))
    return out


def _parse_spec(data: dict) -> dict | None:
    dropped: list = []
    entry = _conds(data.get("entry"), "buy", dropped)
    if not entry:
        return None
    exit_ = _conds(data.get("exit"), "sell", dropped)
    try:
        mh = max(5, min(390, int(float(data.get("max_hold_min") or 60))))
    except (TypeError, ValueError):
        mh = 60
    spec = {"entry": entry, "exit": exit_, "max_hold_min": mh}
    if dropped:
        spec["ignored"] = dropped
    return spec


def _ask(model: str | None, payload: dict) -> dict:
    from .llm import DEFAULT_MODEL, llm_chat
    try:
        out = llm_chat(model or DEFAULT_MODEL,
                       [{"role": "system", "content": "You are a trading agent. JSON only."},
                        {"role": "user", "content": json.dumps(payload)}],
                       temperature=0.6, json_mode=True)
        data = json.loads(out)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def backtest(spec: dict | None) -> dict | None:
    if not spec:
        return None
    try:
        from . import history
        from .market import TRADE_UNIVERSE_MODE
        if TRADE_UNIVERSE_MODE != "top100":
            return None
        return history.backtest_charter(spec)
    except Exception:
        return None


def card_text(card: dict | None, spec: dict | None = None) -> str:
    from .history import card_text as _ct
    txt = _ct(card)
    if spec and spec.get("ignored"):
        txt += (f" (ignored, not columns in the table: {', '.join(spec['ignored'])})")
    return txt


def write_charter(model: str | None, name: str, rival_charters: dict[str, str],
                  arena: str) -> dict:
    data = _ask(model, {
        "you": name,
        "arena": arena,
        "rivals_charters": rival_charters or "none written yet",
        "instruction": (
            "Before trading, write YOUR trading charter: the style you will trade "
            "by in this arena. Your own choice — any style at all. Make it concrete "
            "enough that someone could check a trade against it using ONLY the data "
            "columns listed in the arena description: what you look for "
            "before buying, what makes you sell, how you size, roughly how many "
            "positions. No specific tickers. Your core approach must be DIFFERENT "
            "from every rival charter listed (a different idea of where the edge "
            "is — not the same signal with other numbers): if every agent trades "
            "the same style, the race is decided by noise. 2-4 sentences, <=350 chars. "
            + _SPEC_HELP + " Return JSON {\"charter\":\"...\",\"entry\":[...],"
            "\"exit\":[...],\"max_hold_min\":N}."),
    })
    return {"charter": _clean(data.get("charter", "")), "spec": _parse_spec(data)}


def revise_charter(model: str | None, charter: str, own_trades: list, return_pct: float,
                   rank: int, n: int, discipline: dict | None = None,
                   card: dict | None = None, spec: dict | None = None) -> dict:
    """Keep or rewrite the charter from this round's own results and its
    backtest. Returns {"charter", "spec"} (unchanged on keep / failure)."""
    data = _ask(model, {
        "your_charter": charter,
        "your_charter_spec": spec,
        "your_charter_backtest": card_text(card, spec),
        "your_result": (f"finished {rank} of {n} at {return_pct:+.2f}%" if rank
                        else "pre-round review: no trades yet"),
        "your_own_trades": own_trades[:30],
        "your_charter_discipline": discipline or {},
        "instruction": (
            "Decide whether your charter should change. your_charter_backtest is "
            "how your checkable spec did on ~60 days of real market data (after "
            "trading costs); weigh it above one round. Keep it if it works or the "
            "evidence is thin; refine it if a specific part failed; rewrite it if "
            "the approach has no edge. your_charter_discipline shows buys that fit "
            "your charter vs. deviated and what each made — if you deviated a lot "
            "(e.g. your setup rarely appears), commit to waiting for it or rewrite "
            "it into one you will follow. Keep a core approach different from your "
            "rivals'. No specific tickers. The charter text must describe the "
            "style itself (what you buy, when you sell, sizing) — not commentary on "
            "the backtest. <=350 chars. " + _SPEC_HELP +
            " Return JSON {\"decision\":\"keep\"|\"refine\"|\"rewrite\",\"charter\":\"...\","
            "\"entry\":[...],\"exit\":[...],\"max_hold_min\":N}."),
    })
    new = _clean(data.get("charter", ""))
    new_spec = _parse_spec(data)
    if data.get("decision") in ("refine", "rewrite") and len(new) > 30:
        return {"charter": new, "spec": new_spec or spec}
    return {"charter": charter, "spec": spec or new_spec}


def apply(cfg: dict, result: dict) -> dict:
    """Store a charter + spec on an agent config and backtest the spec."""
    cfg["charter"] = result.get("charter") or cfg.get("charter", "")
    if result.get("spec"):
        cfg["charter_spec"] = result["spec"]
    cfg["charter_card"] = backtest(cfg.get("charter_spec"))
    return cfg


def ensure_charters(conn, arena: str, verbose: bool = True) -> None:
    """At round start: any current agent without a charter writes one, then
    every agent sees its charter's backtest and may refine it once. Written
    one at a time in random order so each must differ from those before it.
    PIT_CHARTERS=0 turns this off (tests: no network)."""
    import os
    import random
    if os.getenv("PIT_CHARTERS", "1") == "0":
        return
    rows = conn.execute("""SELECT l.name, a.id, a.strategy_config FROM lineages l
                           JOIN agents a ON a.id = l.current_agent_id ORDER BY l.id""").fetchall()
    cfgs = {r["name"]: (r["id"], json.loads(r["strategy_config"] or "{}")) for r in rows}
    names = list(cfgs)
    random.shuffle(names)
    existing = {n: c["charter"] for n, (_, c) in cfgs.items() if c.get("charter")}
    for n in names:
        aid, cfg = cfgs[n]
        model = cfg.get("model")
        if not cfg.get("charter"):
            res = write_charter(model, n, dict(existing), arena)
            if not res["charter"]:
                continue
            apply(cfg, res)
            existing[n] = cfg["charter"]
        elif not cfg.get("charter_spec") or "charter_card" not in cfg:
            # older charter: get its checkable spec via a keep-or-refine pass
            apply(cfg, revise_charter(model, cfg["charter"], [], 0.0, 0, 0, spec=None))
        # one pre-round refinement from the backtest card
        if cfg.get("charter_card") is not None:
            res = revise_charter(model, cfg["charter"], [], 0.0, 0, 0,
                                 card=cfg["charter_card"], spec=cfg.get("charter_spec"))
            if res["charter"] != cfg["charter"] or res["spec"] != cfg.get("charter_spec"):
                apply(cfg, res)
                existing[n] = cfg["charter"]
        conn.execute("UPDATE agents SET strategy_config=? WHERE id=?", (json.dumps(cfg), aid))
        conn.commit()
        if verbose:
            print(f"[charter] {n}: {cfg.get('charter')}\n           backtest: "
                  f"{card_text(cfg.get('charter_card'), cfg.get('charter_spec'))}", flush=True)
