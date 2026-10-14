"""Strategy charters: each lineage writes — and evolves — its OWN trading style.

Same model family + same data table + same incentives pushed all three agents
into the same momentum style (book overlap up to 70-100%), so rounds were
decided by sizing noise rather than by different ideas. Instead of assigning
styles (that would be our bias), each lineage chooses one itself: a short,
checkable charter it sees at every decision. After each round it may keep or
rewrite it from its own results — winners refine, losers reconsider. Rivals'
charters are visible only while writing one, so an agent can see whether it
is about to copy a style already in the pool.
"""
from __future__ import annotations

import json

MAX_CHARS = 520


def _clean(text: str) -> str:
    from .forward import _redact_tickers
    return _redact_tickers(" ".join(str(text or "").split()))[:MAX_CHARS]


def write_charter(model: str | None, name: str, rival_charters: dict[str, str],
                  arena: str) -> str:
    from .llm import DEFAULT_MODEL, llm_chat
    payload = {
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
            "Return JSON {\"charter\":\"...\"}."),
    }
    try:
        out = llm_chat(model or DEFAULT_MODEL,
                       [{"role": "system", "content": "You are a trading agent. JSON only."},
                        {"role": "user", "content": json.dumps(payload)}],
                       temperature=0.7, json_mode=True)
        return _clean(json.loads(out).get("charter", ""))
    except Exception:
        return ""


def revise_charter(model: str | None, charter: str, own_trades: list, return_pct: float,
                   rank: int, n: int, discipline: dict | None = None) -> str:
    """Keep or rewrite the charter from this round's own results."""
    from .llm import DEFAULT_MODEL, llm_chat
    payload = {
        "your_charter": charter,
        "your_result": f"finished {rank} of {n} at {return_pct:+.2f}%",
        "your_own_trades": own_trades[:30],
        "your_charter_discipline": discipline or {},
        "instruction": (
            "Decide whether your charter should change after this round. Keep it "
            "if it worked or if one round is too little evidence; refine it if a "
            "specific part failed; rewrite it if the whole approach was wrong. "
            "your_charter_discipline shows how many buys fit your charter vs. "
            "deviated and what each group made. If you deviated a lot (e.g. "
            "because your setup rarely appears), either commit to waiting for it "
            "or rewrite the charter into one you will actually follow. No "
            "specific tickers. Keep a core approach different from your rivals'. "
            "<=350 chars. Return JSON "
            "{\"decision\":\"keep\"|\"refine\"|\"rewrite\",\"charter\":\"...\"}."),
    }
    try:
        out = json.loads(llm_chat(model or DEFAULT_MODEL,
                                  [{"role": "system", "content": "You are a trading agent. JSON only."},
                                   {"role": "user", "content": json.dumps(payload)}],
                                  temperature=0.5, json_mode=True))
        new = _clean(out.get("charter", ""))
        return new if out.get("decision") in ("refine", "rewrite") and len(new) > 30 else charter
    except Exception:
        return charter


def ensure_charters(conn, arena: str, verbose: bool = True) -> None:
    """At round start: any current agent without a charter writes one, each
    with its own model. PIT_CHARTERS=0 turns this off (tests: no network)."""
    import os
    if os.getenv("PIT_CHARTERS", "1") == "0":
        return
    rows = conn.execute("""SELECT l.name, a.id, a.strategy_config FROM lineages l
                           JOIN agents a ON a.id = l.current_agent_id ORDER BY l.id""").fetchall()
    cfgs = {r["name"]: (r["id"], json.loads(r["strategy_config"] or "{}")) for r in rows}
    todo = [n for n, (_, c) in cfgs.items() if not c.get("charter")]
    if not todo:
        return
    # one at a time, in random order: each writer sees the charters already
    # written and must differ from them (written in parallel, all three
    # independently picked the same "intraday momentum" style)
    import random
    random.shuffle(todo)
    existing = {n: c["charter"] for n, (_, c) in cfgs.items() if c.get("charter")}
    for n in todo:
        text = write_charter(cfgs[n][1].get("model"), n, dict(existing), arena)
        if not text:
            continue
        aid, cfg = cfgs[n]
        cfg["charter"] = text
        existing[n] = text
        conn.execute("UPDATE agents SET strategy_config=? WHERE id=?", (json.dumps(cfg), aid))
        if verbose:
            print(f"[charter] {n}: {text}", flush=True)
    conn.commit()
