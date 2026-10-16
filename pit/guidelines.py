"""The pool's rulebook ("constitution"): shared rules the agents write, the
arena researches, and the pool debates and votes on.

After each round (pool_constitution):
  1. review   — every active rule is re-tested; contradicted / no-edge rules
                are retired, probation rules with real support become proven.
  2. propose  — each agent may propose one machine-checkable rule: an entry
                ("buy when …") or exit ("sell when …") condition over the data
                table's columns, or the removal of a rule.
  3. research — the arena tests it on ~60 days of market history (pit/history)
                first and on the pool's own trades second; contradicted, no
                edge, duplicate or evidence-free proposals are vetoed here.
  4. debate   — each other agent raises one objection (optionally a split
                test the arena runs); the proposer rebuts or withdraws.
  5. vote     — proven edge: pool majority; untested / weak lean: every other
                agent, and it enters on probation. At most N adoptions a round.
Active rules are shown to every agent at every decision.
"""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone

from .config import ArenaConfig


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ---- reading -----------------------------------------------------------

def active_guidelines(conn) -> list[dict]:
    return [dict(r) for r in conn.execute(
        "SELECT * FROM guidelines WHERE status='active' ORDER BY id"
    ).fetchall()]


def active_texts(conn) -> list[str]:
    """Labeled so an agent can tell a DO from an AVOID at a glance — this is
    the "leverage to check the guidelines" every decide() call gets: the full
    current list, right in its context, every single turn."""
    out = []
    for g in active_guidelines(conn):
        line = f"{'DO' if g['practice'] == 'good' else 'AVOID'}: {g['text']}"
        if g.get("spec"):
            ev = json.loads(g["evidence"]) if g.get("evidence") else None
            line += f"  [{g.get('tier') or 'probation'}; {evidence_label(ev)}]"
        out.append(line)
    return out


def current_lineages(conn) -> list[dict]:
    return [dict(r) for r in conn.execute(
        """SELECT l.id, l.name, l.wins, l.losses, a.strategy_config
           FROM lineages l JOIN agents a ON a.id = l.current_agent_id
           ORDER BY l.id"""
    ).fetchall()]


def _apply(conn, proposal: dict, tier: str | None = None,
           ledger: dict | None = None) -> None:
    if proposal["kind"] == "add":
        version = conn.execute(
            "SELECT COALESCE(MAX(version),0)+1 v FROM guidelines"
        ).fetchone()["v"]
        conn.execute(
            "INSERT INTO guidelines (text, practice, status, version, created_at, "
            "spec, evidence, tier) VALUES (?, ?, 'active', ?, ?, ?, ?, ?)",
            (proposal["text"], proposal.get("practice", "good"), version, _now(),
             json.dumps(proposal["spec"]) if proposal.get("spec") else None,
             json.dumps(ledger) if ledger else None, tier),
        )
    else:  # remove
        conn.execute(
            "UPDATE guidelines SET status='retired', retired_at=?, retired_reason=? "
            "WHERE id=?",
            (_now(), f"voted out (proposed by {proposal.get('proposer', '?')})",
             proposal["guideline_id"]),
        )


# ---- the agent-driven constitution ---------------------------------------
#
# The pool writes its own rulebook. After a round, EACH agent — using its own
# model and its own experience — may propose one change: a DO (good practice),
# an AVOID (bad practice), or retiring an existing rule its experience
# contradicts. Every agent then votes on every proposal, again with its own
# model and its own notes. A proposal passes only with a strict majority of
# the whole pool (abstentions don't help it). Replaces two biases of the older
# path: a single neutral "summarizer" model deciding what the pool learned,
# and a fallback vote rule that made winners reflexively reject new rules.

ARENA_FACTS = ("Arena facts: every fill pays half the bid/ask spread; "
               "long-only paper trading (no leverage, no shorting, "
               "no options); orders are immediate buys/sells at the current price; "
               "every buy automatically carries a per-position stop-loss (default "
               "distance, or the agent's own stop_loss_pct set on the buy); there "
               "are no limit or take-profit orders; fractional quantities allowed; portfolio and "
               "per-position stop-losses fire automatically, and every open "
               "position is force-closed by the arena at the round's deadline "
               "(agents can't choose to keep or close it); each agent decides "
               "once per wake-up (every few minutes) from a shared data table of "
               "price, returns, moving-average trend, RSI14, 52-week position, "
               "volume and fundamentals; the pool is ranked by return at the deadline.")


def _universe_symbols() -> set[str]:
    try:
        from . import full_market
        from .market import TRADE_UNIVERSE_MODE
        syms = {r["symbol"].upper() for r in full_market.universe_for(TRADE_UNIVERSE_MODE)}
    except Exception:
        syms = set()
    return syms | {x.split("-")[0] for x in syms if x.endswith("-USD")}














def _lineage_cfg(lin: dict) -> dict:
    try:
        return json.loads(lin["strategy_config"] or "{}")
    except Exception:
        return {}




def _ask_json(model: str | None, system: str, payload: dict) -> dict | None:
    from .llm import DEFAULT_MODEL, llm_chat
    try:
        out = llm_chat(model or DEFAULT_MODEL,
                       [{"role": "system", "content": system},
                        {"role": "user", "content": json.dumps(payload)}],
                       temperature=0.5, json_mode=True)
        data = json.loads(out)
        return data if isinstance(data, dict) else None
    except Exception:
        return None


def _round_trades(conn, round_id, lineage_id) -> list[dict]:
    if round_id is None:
        return []
    from .forward import is_forced
    rows = [dict(r) for r in conn.execute(
        """SELECT t.side, t.symbol, round(t.price, 6) AS price, t.reason
           FROM trades t JOIN agents a ON a.id=t.agent_id
           WHERE t.round_id=? AND a.lineage_id=? ORDER BY t.id LIMIT 30""",
        (round_id, lineage_id))]
    for r in rows:
        r["forced_by_arena"] = is_forced(r["reason"])
    return rows


# ---- evidence-checked rules ----------------------------------------------
#
# Why: with every agent on the same small model, the vote was a rubber stamp —
# every proposal got 3/3, however vague or wrong. Asking the model to be "more
# skeptical" didn't change that. So the vote no longer stands alone:
#
#  1. A rule must be MACHINE-CHECKABLE: an entry rule over the data-table
#     columns ("DO: buy when RSI14 < 30", "AVOID: buying when 1d% > 8").
#     Vague, ticker-specific or arena-mechanics rules can't even be expressed.
#  2. The ARENA backtests it against the pool's real ledger: every logged buy
#     carries the table row the agent saw (trades.features) and its realised
#     outcome. Buys matching the condition vs. buys that didn't.
#  3. Evidence gates the vote: contradicted, inconclusive (enough trades,
#     no edge), no matching trades at all, or a duplicate/mirror of an active
#     rule -> vetoed without a vote; supported -> a majority of the pool
#     adopts it; untested (a few trades, not enough to judge) -> needs every
#     other agent too, and enters on probation.
#  4. Every round all active rules are re-tested; a rule the ledger turns
#     against is retired automatically, a probation rule the data backs
#     becomes proven. The rulebook is kept honest by outcomes, not opinions.

_OPS = {"<": lambda a, b: a < b, "<=": lambda a, b: a <= b,
        ">": lambda a, b: a > b, ">=": lambda a, b: a >= b}


def _min_trades() -> int:
    import os
    return int(os.getenv("PIT_RULE_MIN_TRADES", "5"))


# Adoption needs strong evidence (t >= 2, ~95%): several proposals are tested
# every round, so a t of 1 "proved" coincidences (a rule went in at t=+1.00).
# A veto needs less (t <= -1): rejecting a rule costs little, adopting a
# wrong one binds every agent.
_T_SUPPORT = 2.0
_T_CONTRA = 1.0


def _min_probation() -> int:
    """Matching buys needed before an untested rule may even go to a vote."""
    import os
    return int(os.getenv("PIT_RULE_MIN_PROBATION", "2"))


POSITION_FIELDS = {"pos_pnl%": "pos_pnl", "held_min": "held_min"}


def _rule_fields(action: str = "buy") -> dict:
    from .brief import RULE_FIELDS
    return dict(RULE_FIELDS, **POSITION_FIELDS) if action == "sell" else RULE_FIELDS


def parse_spec(when, action: str = "buy") -> dict | None:
    """[{"field","op","value"}, ...] or [[f, op, v], ...] -> {"when": [[f, op, v]]}
    — 1 to 3 conditions over the data-table columns (plus, for an exit rule,
    the position's pos_pnl% and held_min), else None."""
    action = "sell" if action == "sell" else "buy"
    fields = _rule_fields(action)
    if not isinstance(when, list) or not 1 <= len(when) <= 3:
        return None
    out = []
    for c in when:
        if isinstance(c, dict):
            c = [c.get("field"), c.get("op"), c.get("value")]
        if not isinstance(c, (list, tuple)) or len(c) != 3:
            return None
        f, op, v = str(c[0]).strip(), str(c[1]).strip(), c[2]
        if f not in fields or op not in _OPS:
            return None
        try:
            v = float(v)
        except (TypeError, ValueError):
            return None
        out.append([f, op, v])
    return {"when": out, "action": action} if action == "sell" else {"when": out}


def spec_text(spec: dict, practice: str) -> str:
    cond = " and ".join(f"{f} {op} {v:g}" for f, op, v in spec["when"])
    verb = spec.get("action", "buy")
    return (f"{verb} when {cond}" if practice == "good"
            else f"{'selling' if verb == 'sell' else 'buying'} when {cond}")


def _matches(spec: dict, feat: dict) -> bool | None:
    fields = _rule_fields(spec.get("action", "buy"))
    for f, op, v in spec["when"]:
        x = feat.get(fields[f])
        if x is None:
            return None
        if not _OPS[op](float(x), v):
            return False
    return True


def buy_outcomes(conn, round_id: int | None = None,
                 lineage_id: int | None = None) -> list[dict]:
    """Every logged buy with the data row seen at entry and its realised
    return %: exit = qty-weighted price of that agent's later sells of the
    symbol in the same round (forced closes included — that's the real P&L),
    else the last mark if still open."""
    q = """SELECT t.id, t.round_id, t.agent_id, t.symbol, t.price, t.features
           FROM trades t JOIN agents a ON a.id = t.agent_id
           WHERE t.side='buy' AND t.features IS NOT NULL"""
    args: list = []
    if round_id is not None:
        q += " AND t.round_id=?"
        args.append(round_id)
    if lineage_id is not None:
        q += " AND a.lineage_id=?"
        args.append(lineage_id)
    out = []
    for b in conn.execute(q + " ORDER BY t.id", args).fetchall():
        sells = conn.execute(
            """SELECT qty, price FROM trades WHERE round_id=? AND agent_id=?
               AND symbol=? AND side='sell' AND id>?""",
            (b["round_id"], b["agent_id"], b["symbol"], b["id"])).fetchall()
        qty = sum(r["qty"] for r in sells)
        if qty > 0:
            exit_px = sum(r["qty"] * r["price"] for r in sells) / qty
        else:
            m = conn.execute("SELECT mark_prices FROM round_states WHERE round_id=? "
                             "AND agent_id=?", (b["round_id"], b["agent_id"])).fetchone()
            exit_px = json.loads((m and m["mark_prices"]) or "{}").get(b["symbol"])
        if not exit_px or not b["price"]:
            continue
        out.append({"symbol": b["symbol"], "features": json.loads(b["features"]),
                    "ret": (exit_px / b["price"] - 1) * 100,
                    "round_id": b["round_id"], "agent_id": b["agent_id"]})
    return _dedupe(out)


def sell_outcomes(conn, round_id: int | None = None,
                  lineage_id: int | None = None) -> list[dict]:
    """Every sell the AGENT chose (not stops / deadline) with the data row and
    position state at that moment, scored against holding to the deadline:
    ret = (sell price / deadline price - 1) * 100 — positive = the sell beat
    holding. The same backtest then judges exit rules like entry rules."""
    from .forward import is_forced
    q = """SELECT t.round_id, t.agent_id, t.symbol, t.price, t.reason, t.features, r.end_prices
           FROM trades t JOIN agents a ON a.id = t.agent_id JOIN rounds r ON r.id = t.round_id
           WHERE t.side='sell' AND t.features IS NOT NULL AND r.end_prices IS NOT NULL"""
    args: list = []
    if round_id is not None:
        q += " AND t.round_id=?"
        args.append(round_id)
    if lineage_id is not None:
        q += " AND a.lineage_id=?"
        args.append(lineage_id)
    out = []
    for r in conn.execute(q + " ORDER BY t.id", args).fetchall():
        if is_forced(r["reason"]):
            continue
        end = json.loads(r["end_prices"] or "{}").get(r["symbol"])
        if not end or not r["price"]:
            continue
        out.append({"symbol": r["symbol"], "features": json.loads(r["features"]),
                    "ret": (r["price"] / end - 1) * 100,
                    "round_id": r["round_id"], "agent_id": r["agent_id"]})
    return _dedupe(out)


def outcomes_for(conn, spec: dict | None, cache: dict | None = None) -> list[dict]:
    action = (spec or {}).get("action", "buy")
    if cache is not None and action in cache:
        return cache[action]
    out = sell_outcomes(conn) if action == "sell" else buy_outcomes(conn)
    if cache is not None:
        cache[action] = out
    return out


def _dedupe(outcomes: list[dict]) -> list[dict]:
    """One data point per (round, agent, symbol): an agent adding to the same
    name three times in a round is one idea, not three independent trials —
    counting each add inflated n and the t-stat."""
    seen = {}
    for r in outcomes:
        k = (r.get("round_id"), r.get("agent_id"), r["symbol"])
        if k not in seen:
            seen[k] = r
    return list(seen.values())


def _min_rounds() -> int:
    import os
    return int(os.getenv("PIT_RULE_MIN_ROUNDS", "2"))


def backtest(spec: dict, practice: str, outcomes: list[dict]) -> dict:
    """Matching buys vs. the rest. For a DO, matching buys must have done
    better; for an AVOID, worse. t = signed Welch statistic of that edge."""
    import math
    import statistics as stx
    m, o = [], []
    for r in outcomes:
        hit = _matches(spec, r["features"])
        if hit is True:
            m.append(r["ret"])
        elif hit is False:
            o.append(r["ret"])
    ev = {"n_match": len(m), "n_other": len(o),
          "unit": "sells" if spec.get("action") == "sell" else "buys",
          "avg_match": round(stx.mean(m), 3) if m else None,
          "avg_other": round(stx.mean(o), 3) if o else None,
          "win_rate_match": round(sum(x > 0 for x in m) / len(m) * 100) if m else None}
    n = _min_trades()
    m_rounds = {r.get("round_id") for r in outcomes if _matches(spec, r["features"]) is True}
    ev["rounds"] = len(m_rounds)
    if len(m) < n or len(o) < n or len(m_rounds) < _min_rounds():
        # one 30-minute window is one market regime: everything bought in
        # the same minutes moves together, so it can't prove a rule
        ev.update(t=None, verdict="untested")
        return ev
    se = math.sqrt(stx.variance(m) / len(m) + stx.variance(o) / len(o)) or 1e-9
    edge = (stx.mean(m) - stx.mean(o)) * (1 if practice == "good" else -1)
    t = round(edge / se, 2)
    ev.update(t=t, verdict=("supported" if t >= _T_SUPPORT else
                            "contradicted" if t <= -_T_CONTRA else "inconclusive"))
    return ev


def _match_ids(spec: dict, outcomes: list[dict]) -> tuple[set, set]:
    m, o = set(), set()
    for i, r in enumerate(outcomes):
        hit = _matches(spec, r["features"])
        if hit is True:
            m.add(i)
        elif hit is False:
            o.add(i)
    return m, o


def _duplicate_of(p: dict, active: list[dict], outcomes: list[dict],
                  thresh: float = 0.85) -> str | None:
    """An active rule that selects (nearly) the same trades — or, with the
    opposite practice, the mirror image ("AVOID 5d% < 0" vs "DO 5d% > 0 ...")
    — is the same evidence counted twice. Jaccard overlap on the ledger."""
    new_m, _ = _match_ids(p["spec"], outcomes)
    if len(new_m) < 3:
        return None
    for g in active:
        if not g.get("spec"):
            continue
        gspec = json.loads(g["spec"])
        if gspec.get("action", "buy") != p["spec"].get("action", "buy"):
            continue
        gm, go = _match_ids(gspec, outcomes)
        other = gm if g["practice"] == p["practice"] else go
        if other and len(new_m & other) / len(new_m | other) >= thresh:
            return f"{'DO' if g['practice'] == 'good' else 'AVOID'}: {g['text']}"
    return None


def research(spec: dict, practice: str, ledger_outcomes: list[dict]) -> dict:
    """Evidence for a rule: 60 days of market history first (thousands of
    snapshots, day-clustered t), the pool's own trades as a second check.
    Falls back to the trades alone when there's no matching history (crypto,
    or an exit rule on pos_pnl% / held_min)."""
    ledger = backtest(spec, practice, ledger_outcomes)
    hist = None
    try:
        from . import history
        from .market import TRADE_UNIVERSE_MODE
        if TRADE_UNIVERSE_MODE == "top100":
            hist = history.research_rule(spec, practice)
    except Exception:
        hist = None
    if not hist or hist["verdict"] == "untested":
        return ledger
    ev = dict(hist)
    ev["ledger"] = {k: ledger.get(k) for k in ("verdict", "n_match", "n_other", "t",
                                                "avg_match", "avg_other", "unit")}
    return ev


def evidence_label(ev: dict | None) -> str:
    if not ev:
        return "unverifiable (legacy free-text rule)"
    if ev.get("source") == "history":
        lg = ev.get("ledger") or {}
        own = (f"; pool's own trades: {lg.get('verdict')} ({lg.get('n_match')} matching {lg.get('unit', 'buys')})"
               if lg else "")
        return (f"{ev['verdict']} — 60-day market research: {ev['n_match']:,} matching snapshots "
                f"avg {ev['avg_match']:+.3f}% next 30m (win {ev['win_rate_match']}%) vs "
                f"{ev['avg_other']:+.3f}% otherwise, t={ev['t']:+.2f} over {ev['days']} days" + own)
    if ev["verdict"] == "untested":
        return (f"untested — {ev['n_match']} matching / {ev['n_other']} other {ev.get('unit', 'buys')} "
                f"across {ev.get('rounds', '?')} round(s); need {_min_trades()} each "
                f"from at least {_min_rounds()} rounds")
    u = ev.get("unit", "buys")
    return (f"{ev['verdict']} — {ev['n_match']} matching {u} avg {ev['avg_match']:+.2f}% "
            f"(win {ev['win_rate_match']}%) vs {ev['n_other']} others avg "
            f"{ev['avg_other']:+.2f}%, t={ev['t']:+.2f}")


def review_rules(conn) -> list[dict]:
    """Re-test every active rule against the ledger: retire the ones the data
    now contradicts, promote probation rules it supports."""
    cache: dict = {}
    events = []
    for g in active_guidelines(conn):
        if not g.get("spec"):
            continue
        spec = json.loads(g["spec"])
        ev = research(spec, g["practice"], outcomes_for(conn, spec, cache))
        conn.execute("UPDATE guidelines SET evidence=? WHERE id=?", (json.dumps(ev), g["id"]))
        if ev["verdict"] == "inconclusive" and g.get("tier") != "proven" \
                and (ev.get("t") or 0) < _T_CONTRA:
            # a probation rule that now has enough trades and still shows no edge
            conn.execute("UPDATE guidelines SET status='retired', retired_at=?, "
                         "retired_reason=? WHERE id=?",
                         (_now(), "probation ended with no edge: " + evidence_label(ev), g["id"]))
            events.append({"action": "retired", "text": g["text"], "practice": g["practice"],
                           "evidence": ev})
        elif ev["verdict"] == "contradicted":
            conn.execute("UPDATE guidelines SET status='retired', retired_at=?, "
                         "retired_reason=? WHERE id=?",
                         (_now(), "ledger contradicts it: " + evidence_label(ev), g["id"]))
            events.append({"action": "retired", "text": g["text"], "practice": g["practice"],
                           "evidence": ev})
        elif ev["verdict"] in ("untested", "inconclusive") and g.get("tier") == "proven":
            conn.execute("UPDATE guidelines SET tier='probation' WHERE id=?", (g["id"],))
            events.append({"action": "demoted to probation", "text": g["text"],
                           "practice": g["practice"], "evidence": ev})
        elif ev["verdict"] == "supported" and g.get("tier") != "proven":
            conn.execute("UPDATE guidelines SET tier='proven' WHERE id=?", (g["id"],))
            events.append({"action": "proven", "text": g["text"], "practice": g["practice"],
                           "evidence": ev})
    conn.commit()
    return events


def _agent_propose(lin: dict, active: list[dict], trades: list[dict] | None = None,
                   symbols: set[str] | None = None,
                   my_buys: list[dict] | None = None,
                   my_sells: list[dict] | None = None) -> dict | None:
    cfg = _lineage_cfg(lin)
    fields = list(_rule_fields())
    sell_fields = list(_rule_fields("sell"))
    data = _ask_json(cfg.get("model"),
        "You are a trading agent helping write your pool's shared rulebook. JSON only.",
        {"you": lin["name"], "your_record": f"{lin['wins']}W-{lin['losses']}L",
         "your_notes_from_your_own_trades": cfg.get("notes", ""),
         "your_trades_last_round": trades or [],
         "your_buys_last_round_with_the_data_you_saw_and_outcome": [
             {"symbol": b["symbol"], "outcome_pct": round(b["ret"], 3),
              **{f: b["features"].get(k) for f, k in _rule_fields().items()}}
             for b in (my_buys or [])][:20],
         "your_own_sells_last_round_with_the_data_you_saw_and_edge": [
             {"symbol": x["symbol"], "sell_vs_holding_to_deadline_pct": round(x["ret"], 3),
              **{f: x["features"].get(k) for f, k in _rule_fields("sell").items()}}
             for x in (my_sells or [])][:20],
         "arena": ARENA_FACTS,
         "current_rules": [{"id": g["id"],
                            "rule": f"{'DO' if g['practice'] == 'good' else 'AVOID'}: {g['text']}",
                            "evidence": evidence_label(json.loads(g["evidence"]) if g.get("evidence") else None)}
                           for g in active],
         "instruction": (
             "Propose at most ONE change to the rulebook every agent will follow. "
             "A rule is either an ENTRY rule (action 'buy': when to buy / not buy) "
             "or an EXIT rule (action 'sell': when to sell / not sell an open "
             "position). The arena BACKTESTS it against every logged trade — "
             "entries by their outcome, sells against simply holding to the "
             "deadline — and a rule the data contradicts is rejected "
             "automatically, so propose only a pattern your own trades above "
             "actually show. DO (practice 'good': doing it tends to work) or "
             "AVOID (practice 'bad': doing it tends to cost). Conditions: 1-3 of "
             f"[field, op, value]; entry fields {fields}; exit fields {sell_fields} "
             "(pos_pnl% = the position's P&L vs its entry, held_min = minutes "
             "held); op one of <, <=, >, >=. Or remove a current rule your "
             "experience contradicts. Most rounds deserve no new rule — say none "
             "if nothing is clear. Return JSON {\"kind\":\"add\"|\"remove\"|\"none\","
             "\"action\":\"buy\"|\"sell\",\"practice\":\"good\"|\"bad\","
             "\"when\":[[\"RSI14\",\"<\",30]],\"remove_id\":N,\"why\":\"...\"}")})
    if not data or data.get("kind") not in ("add", "remove"):
        return None
    if data["kind"] == "remove":
        ids = {g["id"]: g for g in active}
        try:
            gid = int(data.get("remove_id"))
        except (TypeError, ValueError):
            return None
        if gid not in ids:
            return None
        g = ids[gid]
        return {"kind": "remove", "practice": g["practice"], "text": g["text"],
                "guideline_id": gid, "proposer": lin["name"],
                "spec": json.loads(g["spec"]) if g.get("spec") else None,
                "evidence": str(data.get("why", ""))[:300]}
    spec = parse_spec(data.get("when"), data.get("action", "buy"))
    if not spec:
        return None
    practice = "bad" if data.get("practice") == "bad" else "good"
    return {"kind": "add", "practice": practice, "text": spec_text(spec, practice),
            "spec": spec, "guideline_id": None, "proposer": lin["name"],
            "evidence": str(data.get("why", ""))[:300]}


def _agent_vote(lin: dict, proposal: dict, active: list[str],
                trades: list[dict] | None = None, ledger: dict | None = None,
                debate_rec: dict | None = None) -> tuple[str, str]:
    cfg = _lineage_cfg(lin)
    label = "DO" if proposal.get("practice") == "good" else "AVOID"
    data = _ask_json(cfg.get("model"),
        "You are a trading agent voting on your pool's shared rulebook. JSON only.",
        {"you": lin["name"], "your_record": f"{lin['wins']}W-{lin['losses']}L",
         "your_notes_from_your_own_trades": cfg.get("notes", ""),
         "your_trades_last_round": trades or [],
         "arena": ARENA_FACTS,
         "current_rules": active,
         "proposal": (f"{proposal['kind'].upper()} rule — {label}: {proposal['text']}"),
         "proposer_says": proposal.get("evidence", ""),
         "arena_research": evidence_label(ledger) if ledger else "not applicable",
         **({"debate": {"objections": debate_rec["objections"],
                        "proposer_rebuttal": debate_rec["rebuttal"]}} if debate_rec else {}),
         "instruction": (
             "This rule would bind every agent, including you. The arena backtest "
             "above is hard data from the pool's real trades — weigh it above the "
             "proposer's story. 'untested' means too few trades to tell: then agree "
             "only if your own trades clearly back it. 'inconclusive' means the data "
             "shows no real edge. Disagree if it would have hurt your own trades or "
             "duplicates a current rule. Return JSON "
             "{\"evidence_from_my_trades\":\"...\",\"vote\":\"agree\"|\"disagree\","
             "\"reason\":\"<=200 chars\"}")})
    if not data or data.get("vote") not in ("agree", "disagree"):
        return "abstain", "no valid vote returned"
    ev = str(data.get("evidence_from_my_trades", "")).strip()
    return data["vote"], (str(data.get("reason", "")) + (f" | evidence: {ev}" if ev else ""))[:300]


def _split_test(spec: dict, practice: str, cond) -> dict | None:
    """Re-run the market research only where an extra condition holds — the
    narrower test an objector asked for ("I bet it fails on down days")."""
    one = parse_spec([cond], spec.get("action", "buy"))
    if not one:
        return None
    try:
        from . import history
        from .market import TRADE_UNIVERSE_MODE
        df = history.panel() if TRADE_UNIVERSE_MODE == "top100" else None
        if df is None:
            return None
        m = history._mask(df, one["when"])
        if m is None:
            return None
        ev = history.research_rule(spec, practice, df[m])
        if ev:
            ev["condition"] = f"{one['when'][0][0]} {one['when'][0][1]} {one['when'][0][2]:g}"
        return ev
    except Exception:
        return None


def debate(p: dict, ledger: dict | None, lineages: list[dict]) -> dict:
    """Case -> objections (each other agent, own model, may request a split
    test the arena runs on the history) -> proposer's rebuttal / withdrawal.
    The record goes to the voters and onto the leaderboard."""
    from concurrent.futures import ThreadPoolExecutor
    label = "DO" if p["practice"] == "good" else "AVOID"
    case = {"rule": f"{label}: {p['text']}", "proposer": p["proposer"],
            "proposer_says": p.get("evidence", ""), "research": evidence_label(ledger)}
    others = [l for l in lineages if l["name"] != p["proposer"]]

    def _object(l):
        data = _ask_json(_lineage_cfg(l).get("model"),
            "You are a trading agent debating a proposed rule for your pool. JSON only.",
            {"you": l["name"], "case": case, "arena": ARENA_FACTS,
             "instruction": (
                 "Raise ONE concrete objection to this rule — a specific reason it may "
                 "not hold (a market condition where it breaks, a confound, overlap "
                 "with an existing rule, too small an effect after costs). You may "
                 "request ONE split test: a condition [column, op, value] over the "
                 "data-table columns; the arena will re-run the research only where it "
                 "holds. Return JSON {\"objection\":\"<=200 chars\",\"test\":[col,op,val] or null}")})
        data = data or {}
        test = data.get("test")
        res = _split_test(p["spec"], p["practice"], test) if isinstance(test, list) and len(test) == 3 else None
        return {"by": l["name"], "objection": str(data.get("objection", ""))[:240],
                "test": res and {"condition": res.get("condition"), "verdict": res.get("verdict"),
                                 "t": res.get("t"), "avg_match": res.get("avg_match"),
                                 "avg_other": res.get("avg_other"), "days": res.get("days")}}
    with ThreadPoolExecutor(max_workers=max(1, len(others))) as ex:
        objections = list(ex.map(_object, others))
    prop = next((l for l in lineages if l["name"] == p["proposer"]), None)
    reb = _ask_json(_lineage_cfg(prop).get("model") if prop else None,
        "You are a trading agent defending your proposed rule. JSON only.",
        {"you": p["proposer"], "case": case, "objections": objections,
         "instruction": ("Answer the objections in <=240 chars, citing the split-test "
                         "results where given. If they show the rule doesn't hold, "
                         "withdraw it. Return JSON {\"rebuttal\":\"...\",\"withdraw\":true|false}")}) or {}
    return {"case": case, "objections": objections,
            "rebuttal": str(reb.get("rebuttal", ""))[:300], "withdrawn": bool(reb.get("withdraw"))}


def _spec_key(p: dict):
    if p["kind"] == "remove":
        return ("rm", p["guideline_id"])
    return (p["practice"], p["spec"].get("action", "buy"), json.dumps(sorted(p["spec"]["when"])))


def pool_constitution(conn, config: ArenaConfig, source_round_id: int | None) -> list[dict]:
    """Re-test the rulebook, then every agent may propose one change; the
    arena backtests each and the other agents vote with that evidence.
    Returns one summary dict per proposal (plus review events under
    "action")."""
    from concurrent.futures import ThreadPoolExecutor
    lineages = current_lineages(conn)
    if not lineages:
        return []
    reviews = review_rules(conn)
    active = active_guidelines(conn)
    symbols = _universe_symbols()
    trades = {l["id"]: _round_trades(conn, source_round_id, l["id"]) for l in lineages}
    my_buys = {l["id"]: (buy_outcomes(conn, source_round_id, l["id"])
                         if source_round_id is not None else []) for l in lineages}
    my_sells = {l["id"]: (sell_outcomes(conn, source_round_id, l["id"])
                          if source_round_id is not None else []) for l in lineages}
    with ThreadPoolExecutor(max_workers=len(lineages)) as ex:
        raw = list(ex.map(lambda l: _agent_propose(l, active, trades[l["id"]], symbols,
                                                   my_buys[l["id"]], my_sells[l["id"]]),
                          lineages))

    seen = {(g["practice"], json.loads(g["spec"]).get("action", "buy"),
             json.dumps(sorted(json.loads(g["spec"])["when"])))
            for g in active if g.get("spec")}
    proposals, keys = [], set()
    for p in raw:
        if not p:
            continue
        k = _spec_key(p)
        if (p["kind"] == "add" and k in seen) or k in keys:
            continue
        keys.add(k)
        proposals.append(p)

    ocache: dict = {}
    results = []
    for p in proposals[:config.max_proposals_per_round]:
        if p["kind"] == "add" and len(active_guidelines(conn)) >= config.max_active_guidelines:
            continue   # rulebook full — only removals can make room
        outcomes = outcomes_for(conn, p.get("spec"), ocache)
        ledger = research(p["spec"], p["practice"], outcomes) if p.get("spec") else None
        pid = conn.execute(
            """INSERT INTO guideline_proposals
               (kind, practice, guideline_id, proposed_text, source_round_id,
                created_at, proposer, spec, evidence)
               VALUES (?,?,?,?,?,?,?,?,?)""",
            (p["kind"], p["practice"], p.get("guideline_id"), p["text"],
             source_round_id, _now(), p["proposer"],
             json.dumps(p["spec"]) if p.get("spec") else None,
             json.dumps(ledger) if ledger else None)).lastrowid
        conn.commit()
        n_others = len(lineages) - 1
        r = {"proposal_id": pid, "kind": p["kind"], "practice": p["practice"],
             "text": p["text"], "proposer": p["proposer"], "evidence": p.get("evidence", ""),
             "ledger": ledger, "agree": 1, "total": len(lineages), "accepted": False,
             "_p": p}
        results.append(r)
        # the data vetoes first: an add the ledger contradicts, or removing a
        # rule the ledger supports, never reaches a vote
        verdict = (ledger or {}).get("verdict")
        if (p["kind"] == "add" and verdict == "contradicted") or \
           (p["kind"] == "remove" and verdict == "supported"):
            r["note"] = "vetoed by the ledger: " + evidence_label(ledger)
            continue
        # Enough trades on both sides and still no edge: the data has spoken
        # — probation is for rules we can't judge yet, not ones that don't work
        # t < 1: the data shows essentially no edge -> veto. 1 <= t < 2: a
        # promising lean on a small sample (e.g. 14 vs 9 buys, t=+1.13) —
        # it may go to the stricter probation vote instead of dying here.
        if p["kind"] == "add" and verdict == "inconclusive" and (ledger.get("t") or 0) < _T_CONTRA:
            r["note"] = "vetoed: no edge in the data — " + evidence_label(ledger)
            continue
        dup = _duplicate_of(p, active_guidelines(conn), outcomes) if p["kind"] == "add" else None
        if dup:
            r["note"] = f"vetoed: selects the same trades as active rule \"{dup}\""
            continue
        # No logged buy has ever matched the condition: there is no evidence
        # at all, and an AVOID rule adopted now could never be tested later
        # (agents obeying it never produce a matching buy). Not adoptable.
        if p["kind"] == "add" and ledger and ledger["n_match"] < _min_probation():
            r["note"] = (f"vetoed: no evidence — {ledger['n_match']} logged {ledger.get('unit', 'buys')} match "
                         f"this condition (need {_min_probation()})")
            continue
        texts = active_texts(conn)
        deb = None
        if p.get("spec"):
            try:
                deb = debate(p, ledger, lineages)
            except Exception:
                deb = None
            if deb:
                conn.execute("UPDATE guideline_proposals SET debate=? WHERE id=?",
                             (json.dumps(deb), pid))
                r["debate"] = deb
                if deb.get("withdrawn"):
                    r["note"] = "withdrawn by the proposer after the debate"
                    continue

        def _v(l):
            if l["name"] == p["proposer"]:
                return "agree", "proposer"
            return _agent_vote(l, p, texts, trades[l["id"]], ledger, deb)
        with ThreadPoolExecutor(max_workers=len(lineages)) as ex:
            votes = list(ex.map(_v, lineages))
        for lin, (vote, reason) in zip(lineages, votes):
            if vote != "abstain":
                conn.execute(
                    """INSERT OR REPLACE INTO guideline_votes
                       (proposal_id, lineage_id, vote, reasoning, created_at)
                       VALUES (?,?,?,?,?)""",
                    (pid, lin["id"], vote, reason, _now()))
        others = sum(1 for l, (v, _) in zip(lineages, votes)
                     if v == "agree" and l["name"] != p["proposer"])
        r["agree"] = others + 1
        # Backed by the data: a strict majority of the whole pool (proposer
        # included) adopts it. Not yet testable: EVERY other agent must agree,
        # and it enters on probation until the ledger can judge it.
        if p["kind"] == "add" and verdict != "supported":
            r["required"] = f"all {n_others} other agents (no proven edge yet)"
            r["accepted"] = others == n_others
            r["tier"] = "probation"
        else:
            r["required"] = "a majority of the pool"
            r["accepted"] = r["agree"] * 2 > len(lineages)
            r["tier"] = "proven" if verdict == "supported" else None

    # adopt at most N per round: proven edges first, then the strongest support
    def _rank(r):
        t = (r["ledger"] or {}).get("t") or 0
        return (-(1 if (r["ledger"] or {}).get("verdict") == "supported" else 0), -t, -r["agree"])
    winners = sorted((r for r in results if r["accepted"]), key=_rank)[:config.max_adoptions_per_round]
    keep = {r["proposal_id"] for r in winners}
    for r in results:
        if r["accepted"] and r["proposal_id"] not in keep:
            r["accepted"], r["note"] = False, "passed but over this round's adoption limit"
        if r["accepted"]:
            _apply(conn, r["_p"], tier=r.get("tier"), ledger=r["ledger"])
        res = ("accepted" if r["accepted"] else
               "over_limit" if "adoption limit" in r.get("note", "") else
               "withdrawn" if "withdrawn" in r.get("note", "") else
               "vetoed" if r.get("note") else "rejected")
        conn.execute("UPDATE guideline_proposals SET resolution=?, resolved_at=? WHERE id=?",
                     (res, _now(), r["proposal_id"]))
        r["resolution"] = res
        r.pop("_p")
    conn.commit()
    return results + reviews
