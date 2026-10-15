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


def _panel():
    try:
        from . import history
        from .market import TRADE_UNIVERSE_MODE
        return history.panel() if TRADE_UNIVERSE_MODE == "top100" else None
    except Exception:
        return None


def backtest(spec: dict | None, part: str = "both") -> dict | None:
    """part='train' — the earlier days the search may look at; 'both' — a
    card with the train result AND the unseen later days (what agents and the
    page show)."""
    df = _panel()
    if not spec or df is None:
        return None
    from . import history
    try:
        tr, te = history.split(df)
        train = history.backtest_charter(spec, tr)
        if part == "train":
            return train
        return {"train": train, "test": history.backtest_charter(spec, te)}
    except Exception:
        return None


def _score(card: dict | None) -> float:
    """Rank charter versions by the search days' result."""
    c = (card or {}).get("train", card) or {}
    if not c.get("trades"):
        return -99.0
    t = c.get("t")
    if t is None:
        return -50.0                 # too few trades to judge
    if t <= -2:
        return -80.0 + t             # reliably losing ranks below "too few to judge"
    return float(t)


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
                   card: dict | None = None, spec: dict | None = None,
                   tried: list | None = None, rivals: dict | None = None) -> dict:
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
        **({"versions_already_tried": tried} if tried else {}),
        **({"rivals_charters": rivals} if rivals else {}),
        "instruction": (
            "Decide whether your charter should change. your_charter_backtest is "
            "how your checkable spec did on real market data (after trading costs; "
            "'search days' are the history you may tune on, 'unseen later days' "
            "show whether it holds up) — weigh it above one round. Don't repeat a "
            "version already tried; a real edge should be a genuine idea, not a "
            "lucky threshold. Keep it if it works or the "
            "evidence is thin; refine it if a specific part failed; rewrite it if "
            "the approach has no edge. your_charter_discipline shows buys that fit "
            "your charter vs. deviated and what each made — if you deviated a lot "
            "(e.g. your setup rarely appears), commit to waiting for it or rewrite "
            "it into one you will follow. Your core approach must stay DIFFERENT "
            "from every charter in rivals_charters (a different idea of where the "
            "edge is, not the same signal with other numbers). No specific tickers. The charter text must describe the "
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


def search(cfg: dict, model: str | None, iters: int | None = None,
           verbose: bool = False, name: str = "", rivals: dict | None = None) -> dict:
    """Backtest -> refine -> backtest, up to `iters` times (PIT_CHARTER_ITERS,
    default 3), stopping early once the search days show an edge. Every
    version tried is shown to the agent so it doesn't repeat itself; the
    version with the best search-days result is kept and then scored on the
    unseen later days."""
    import os
    iters = iters if iters is not None else int(os.getenv("PIT_CHARTER_ITERS", "3"))
    tried = [{"charter": cfg["charter"], "spec": cfg.get("charter_spec"),
              "card": cfg.get("charter_card")}]
    for _ in range(iters):
        best = max(tried, key=lambda v: _score(v["card"]))
        if ((best["card"] or {}).get("train") or {}).get("verdict") == "edge":
            break
        res = revise_charter(model, best["charter"], [], 0.0, 0, 0,
                             card={"train": (best["card"] or {}).get("train")},
                             spec=best["spec"], rivals=rivals,
                             tried=[{"spec": v["spec"],
                                     "search_days": card_text((v["card"] or {}).get("train"))}
                                    for v in tried])
        if res["spec"] == best["spec"]:
            break                                           # agent chose to keep it
        tried.append({"charter": res["charter"], "spec": res["spec"],
                      "card": backtest(res["spec"])})
        if verbose:
            print(f"[charter] {name} tried: {card_text(tried[-1]['card'])}", flush=True)
    best = max(tried, key=lambda v: _score(v["card"]))
    cfg["charter"], cfg["charter_spec"], cfg["charter_card"] = (best["charter"], best["spec"],
                                                               best["card"])
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
            apply(cfg, revise_charter(model, cfg["charter"], [], 0.0, 0, 0, spec=None,
                                      rivals={k: v for k, v in existing.items() if k != n}))
        # pre-round search: refine from the backtest a few times, keep the best
        if cfg.get("charter_card") is not None:
            search(cfg, model, verbose=verbose, name=n,
                   rivals={k: v for k, v in existing.items() if k != n})
            existing[n] = cfg["charter"]
        conn.execute("UPDATE agents SET strategy_config=? WHERE id=?", (json.dumps(cfg), aid))
        conn.commit()
        if verbose:
            print(f"[charter] {n}: {cfg.get('charter')}\n           backtest: "
                  f"{card_text(cfg.get('charter_card'), cfg.get('charter_spec'))}", flush=True)
