"""The pool-level "constitution" — shared rules changed only by a vote.

A second evolution mechanism, independent of per-lineage mutation: every N
rounds a *reflection* pass looks across recent rounds for a pattern that keeps
recurring (never a one-off), drafts at most one proposal, and every lineage
votes. A strict majority carries it; a tie keeps the status quo. Active
guidelines are injected into every agent's context.

The whole mechanism is deterministic and offline by default; if Groq is
available the drafting/voting can be delegated to it (with fallback here).
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


def _recent_experience(conn, per_lineage: int) -> list[dict]:
    """Both lineages' own self-reflection notes — what they actually SAID
    they learned from their own trades (forward._reflect_winner/_loser),
    newest generations first per lineage. This is the "agents communicate
    about their own experience" raw material the reflection pass now drafts
    guidelines from, instead of only inferring a pattern from trade-count/
    drawdown numbers."""
    rows = conn.execute(
        """SELECT l.name AS lineage, a.generation,
                  json_extract(a.strategy_config, '$.notes') AS notes,
                  a.mutation_note
           FROM agents a JOIN lineages l ON l.id = a.lineage_id
           ORDER BY l.id, a.generation DESC"""
    ).fetchall()
    by_lineage: dict[str, list[dict]] = {}
    for r in rows:
        note = r["notes"] or r["mutation_note"]
        if not note:
            continue
        by_lineage.setdefault(r["lineage"], [])
        if len(by_lineage[r["lineage"]]) < per_lineage:
            by_lineage[r["lineage"]].append(
                {"generation": r["generation"], "note": note})
    return [{"lineage": name, "notes": notes}
            for name, notes in by_lineage.items() if notes]


def _recent_stats(conn, limit: int) -> list[dict]:
    """Per-round winner-vs-loser trade counts and drawdowns, newest first."""
    return [dict(r) for r in conn.execute(
        """SELECT r.round_number,
                  ws.trade_count AS w_tc, ls.trade_count AS l_tc,
                  ws.max_drawdown_pct AS w_dd, ls.max_drawdown_pct AS l_dd
           FROM round_results rr
           JOIN rounds r ON r.id = rr.round_id
           JOIN round_states ws ON ws.round_id = rr.round_id
                AND ws.agent_id = rr.winner_agent_id
           JOIN round_states ls ON ls.round_id = rr.round_id
                AND ls.agent_id = rr.loser_agent_id
           ORDER BY r.round_number DESC LIMIT ?""",
        (limit,),
    ).fetchall()]


# ---- drafting a proposal ----------------------------------------------

# Each detectable pattern maps to a tag (so we don't re-propose a duplicate)
# and the guideline text it would add.
def draft_proposal(conn, config: ArenaConfig, use_llm: bool = False) -> dict | None:
    """Return a proposal dict, or None if no recurring pattern is found.

    dict shape: {"kind": "add"|"remove", "text": str, "tag": str,
                 "guideline_id": int|None, "evidence": str}
    """
    if use_llm:
        # Primary path: ground the proposal in what both agents actually said
        # about their own trades, not just win/loss statistics.
        experience = _recent_experience(conn, config.reflection_interval)
        got = _llm_draft_from_experience(experience, active_texts(conn))
        if got is not None:
            return got
        got = _llm_draft(conn, config)
        if got is not None:
            return got  # includes explicit None only via fallback below

    # Offline/no-pattern-yet fallback: the older stats heuristic (winners'
    # trade-count/drawdown vs losers'). Always tags practice="good" — both
    # detectable patterns here describe something worth DOING, not avoiding.
    sample = _recent_stats(conn, config.reflection_interval * 2)
    if len(sample) < config.reflection_min_sample:
        return None
    n = len(sample)
    frac = config.reflection_pattern_frac
    active = {_tag_of(t): (gid, t) for gid, t in _active_tagged(conn)}

    fewer = sum(1 for s in sample if s["w_tc"] < s["l_tc"]) / n
    lower_dd = sum(1 for s in sample if s["w_dd"] < s["l_dd"]) / n

    # ADD if a pattern holds and isn't already codified.
    if fewer >= frac and "fewer_trades" not in active:
        return {"kind": "add", "practice": "good", "tag": "fewer_trades",
                "text": f"Prefer fewer, higher-conviction trades — winners "
                        f"out-traded by fewer fills in {fewer*100:.0f}% of the "
                        f"last {n} rounds.",
                "guideline_id": None,
                "evidence": f"fewer-trade winners {fewer*100:.0f}% of {n} rounds"}
    if lower_dd >= frac and "low_drawdown" not in active:
        return {"kind": "add", "practice": "good", "tag": "low_drawdown",
                "text": f"Control drawdown tightly — winners held a smaller max "
                        f"drawdown in {lower_dd*100:.0f}% of the last {n} rounds.",
                "guideline_id": None,
                "evidence": f"lower-drawdown winners {lower_dd*100:.0f}% of {n} rounds"}

    # REMOVE if a codified pattern has clearly stopped holding.
    if "fewer_trades" in active and fewer < (1 - frac):
        gid, text = active["fewer_trades"]
        return {"kind": "remove", "practice": "good", "tag": "fewer_trades", "text": text,
                "guideline_id": gid,
                "evidence": f"fewer-trade winners only {fewer*100:.0f}% lately"}
    if "low_drawdown" in active and lower_dd < (1 - frac):
        gid, text = active["low_drawdown"]
        return {"kind": "remove", "practice": "good", "tag": "low_drawdown", "text": text,
                "guideline_id": gid,
                "evidence": f"lower-drawdown winners only {lower_dd*100:.0f}% lately"}
    return None


def _llm_draft_from_experience(experience: list[dict], active: list[str]) -> dict | None:
    try:
        from .llm import groq_available, llm_draft_guideline_from_experience
        if not groq_available() or not experience:
            return None
        got = llm_draft_guideline_from_experience(experience, active)
        if got is None or got.get("kind") not in ("add", "remove"):
            return None
        got.setdefault("tag", got.get("kind", "") + "_" + str(hash(got.get("text", "")))[:6])
        return got
    except Exception:
        return None


_TAG_KEYWORDS = {"fewer_trades": "fewer", "low_drawdown": "drawdown"}


def _tag_of(text: str) -> str:
    low = text.lower()
    for tag, kw in _TAG_KEYWORDS.items():
        if kw in low:
            return tag
    return "other"


def _active_tagged(conn) -> list[tuple[int, str]]:
    return [(g["id"], g["text"]) for g in active_guidelines(conn)]


# ---- voting ------------------------------------------------------------

def cast_votes(conn, proposal_id: int, proposal: dict, config: ArenaConfig,
               use_llm: bool = False) -> None:
    for lin in current_lineages(conn):
        if use_llm:
            vote, reason = _llm_vote(lin, proposal)
        else:
            vote, reason = _heuristic_vote(lin, proposal)
        conn.execute(
            """INSERT OR REPLACE INTO guideline_votes
               (proposal_id, lineage_id, vote, reasoning, created_at)
               VALUES (?,?,?,?,?)""",
            (proposal_id, lin["id"], vote, reason, _now()),
        )
    conn.commit()


def _heuristic_vote(lin: dict, proposal: dict) -> tuple[str, str]:
    """A trailing lineage welcomes new guidance; a winning one resists it.

    For a removal, invert: a confident (winning) lineage is happy to drop a
    constraint, a struggling one wants to keep any help it has.
    """
    winning = lin["wins"] > lin["losses"]
    if proposal["kind"] == "add":
        if not winning:
            return "agree", "trailing/level record — open to shared guidance"
        return "disagree", "winning record — reluctant to add constraints"
    else:  # remove
        if winning:
            return "agree", "winning — comfortable dropping the constraint"
        return "disagree", "still struggling — keep the guidance for now"


# ---- resolving ---------------------------------------------------------

def open_and_resolve(conn, config: ArenaConfig, source_round_id: int | None,
                     use_llm: bool = False) -> dict | None:
    """Full reflection cycle: draft → open → vote → tally → apply.

    Returns a summary dict (or None if nothing was proposed).
    """
    proposal = draft_proposal(conn, config, use_llm)
    if proposal is None:
        return None

    cur = conn.execute(
        """INSERT INTO guideline_proposals
           (kind, practice, guideline_id, proposed_text, source_round_id, created_at)
           VALUES (?,?,?,?,?,?)""",
        (proposal["kind"], proposal.get("practice", "good"),
         proposal.get("guideline_id"), proposal["text"], source_round_id, _now()),
    )
    proposal_id = cur.lastrowid
    conn.commit()

    cast_votes(conn, proposal_id, proposal, config, use_llm)

    votes = conn.execute(
        "SELECT vote, COUNT(*) c FROM guideline_votes WHERE proposal_id=? GROUP BY vote",
        (proposal_id,),
    ).fetchall()
    tally = {r["vote"]: r["c"] for r in votes}
    agree = tally.get("agree", 0)
    total = agree + tally.get("disagree", 0)
    accepted = total > 0 and agree * 2 > total  # strict majority; tie => status quo

    if accepted:
        _apply(conn, proposal)
    conn.execute(
        "UPDATE guideline_proposals SET resolution=?, resolved_at=? WHERE id=?",
        ("accepted" if accepted else "rejected", _now(), proposal_id),
    )
    conn.commit()

    return {
        "proposal_id": proposal_id,
        "kind": proposal["kind"],
        "practice": proposal.get("practice", "good"),
        "text": proposal["text"],
        "evidence": proposal.get("evidence", ""),
        "agree": agree,
        "total": total,
        "accepted": accepted,
    }


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


# ---- optional LLM delegation (best-effort, falls back to heuristics) ----

def _llm_draft(conn, config):
    try:
        from .llm import groq_available
        if not groq_available():
            return None
        from .llm import llm_draft_guideline
        sample = _recent_stats(conn, config.reflection_interval * 2)
        if len(sample) < config.reflection_min_sample:
            return None
        return llm_draft_guideline(sample, active_texts(conn))
    except Exception:
        return None


def _llm_vote(lin, proposal):
    try:
        from .llm import groq_available
        if not groq_available():
            return _heuristic_vote(lin, proposal)
        from .llm import llm_vote_guideline
        cfg = json.loads(lin["strategy_config"])
        return llm_vote_guideline(cfg, proposal)
    except Exception:
        return _heuristic_vote(lin, proposal)


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

ARENA_FACTS = ("Arena facts: long-only paper trading (no leverage, no shorting, "
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


def _names_a_ticker(text: str, symbols: set[str]) -> bool:
    """A shared rule must apply to ANY name. Rules that name a specific
    ticker ('Monitor AAVE-USD…') are just one agent's trade idea broadcast to
    the whole pool — they'd herd everyone into the same names. Only ALL-CAPS
    tokens count, so ordinary words ('now', 'all', 'on') don't false-match."""
    import re
    for tok in re.findall(r"\b[A-Z][A-Z0-9.\-]{1,9}\b", text):
        if tok in symbols or tok.endswith("-USD"):
            return True
    return False


_VAGUE_STARTS = ("monitor", "consider", "keep an eye", "be careful", "be mindful",
                 "stay ", "remember", "think about", "pay attention", "watch ")


_OUT_OF_CONTROL = ("forced", "round-end", "round end", "deadline", "close-out",
                   "closeout", "end of the round", "end of round")


def _about_arena_mechanics(text: str) -> bool:
    """Rules about the arena's own automatic actions (deadline close-out,
    forced sells) — agents can't control them, yet kept proposing rules about
    them even with those trades labelled forced_by_arena."""
    low = text.lower()
    return any(k in low for k in _OUT_OF_CONTROL)


def _vague(text: str) -> bool:
    """'Monitor RSI…', 'Consider the trend…' — advice, not a rule anyone can
    follow or check. The first agent-written rulebook filled up with these."""
    return text.lower().lstrip().startswith(_VAGUE_STARTS)


def _normalise_rule(text: str, practice: str) -> tuple[str, str]:
    """'DO: Avoid X' / 'Never X' / "Don't X" -> practice='bad', text='X'.
    One idea per rule: anything after a ';' is dropped (the model tends to
    staple a DO onto an AVOID, which makes the label wrong for half of it)."""
    t = text.strip().split(";")[0].strip()
    for pre in ("DO:", "Do:", "AVOID:", "Avoid:"):
        if t.startswith(pre):
            practice = "bad" if pre.lower().startswith("avoid") else practice
            t = t[len(pre):].strip()
    low = t.lower()
    for verb in ("avoid ", "don't ", "do not ", "never "):
        if low.startswith(verb):
            return t[len(verb):].strip().rstrip("."), "bad"
    return t.rstrip("."), practice


def _lineage_cfg(lin: dict) -> dict:
    try:
        return json.loads(lin["strategy_config"] or "{}")
    except Exception:
        return {}


def _norm(text: str) -> str:
    return " ".join(text.lower().replace(".", " ").split())


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


_T_SUPPORT = 1.0   # |t| at which the ledger counts as taking a side


def _min_probation() -> int:
    """Matching buys needed before an untested rule may even go to a vote."""
    import os
    return int(os.getenv("PIT_RULE_MIN_PROBATION", "2"))


def _rule_fields() -> dict:
    from .brief import RULE_FIELDS
    return RULE_FIELDS


def parse_spec(when) -> dict | None:
    """[{"field","op","value"}, ...] or [[f, op, v], ...] -> {"when": [[f, op, v]]}
    — 1 to 3 conditions over the data-table columns, else None."""
    fields = _rule_fields()
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
    return {"when": out}


def spec_text(spec: dict, practice: str) -> str:
    cond = " and ".join(f"{f} {op} {v:g}" for f, op, v in spec["when"])
    return f"buy when {cond}" if practice == "good" else f"buying when {cond}"


def _matches(spec: dict, feat: dict) -> bool | None:
    fields = _rule_fields()
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
                    "ret": (exit_px / b["price"] - 1) * 100})
    return out


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
          "avg_match": round(stx.mean(m), 3) if m else None,
          "avg_other": round(stx.mean(o), 3) if o else None,
          "win_rate_match": round(sum(x > 0 for x in m) / len(m) * 100) if m else None}
    n = _min_trades()
    if len(m) < n or len(o) < n:
        ev.update(t=None, verdict="untested")
        return ev
    se = math.sqrt(stx.variance(m) / len(m) + stx.variance(o) / len(o)) or 1e-9
    edge = (stx.mean(m) - stx.mean(o)) * (1 if practice == "good" else -1)
    t = round(edge / se, 2)
    ev.update(t=t, verdict=("supported" if t >= _T_SUPPORT else
                            "contradicted" if t <= -_T_SUPPORT else "inconclusive"))
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
        gm, go = _match_ids(json.loads(g["spec"]), outcomes)
        other = gm if g["practice"] == p["practice"] else go
        if other and len(new_m & other) / len(new_m | other) >= thresh:
            return f"{'DO' if g['practice'] == 'good' else 'AVOID'}: {g['text']}"
    return None


def evidence_label(ev: dict | None) -> str:
    if not ev:
        return "unverifiable (legacy free-text rule)"
    if ev["verdict"] == "untested":
        return (f"untested — {ev['n_match']} matching / {ev['n_other']} other buys "
                f"so far, need {_min_trades()} each")
    return (f"{ev['verdict']} — {ev['n_match']} matching buys avg {ev['avg_match']:+.2f}% "
            f"(win {ev['win_rate_match']}%) vs {ev['n_other']} others avg "
            f"{ev['avg_other']:+.2f}%, t={ev['t']:+.2f}")


def review_rules(conn) -> list[dict]:
    """Re-test every active rule against the ledger: retire the ones the data
    now contradicts, promote probation rules it supports."""
    outcomes = buy_outcomes(conn)
    events = []
    for g in active_guidelines(conn):
        if not g.get("spec"):
            continue
        spec = json.loads(g["spec"])
        ev = backtest(spec, g["practice"], outcomes)
        conn.execute("UPDATE guidelines SET evidence=? WHERE id=?", (json.dumps(ev), g["id"]))
        if ev["verdict"] == "inconclusive" and g.get("tier") != "proven":
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
        elif ev["verdict"] == "supported" and g.get("tier") != "proven":
            conn.execute("UPDATE guidelines SET tier='proven' WHERE id=?", (g["id"],))
            events.append({"action": "proven", "text": g["text"], "practice": g["practice"],
                           "evidence": ev})
    conn.commit()
    return events


def _agent_propose(lin: dict, active: list[dict], trades: list[dict] | None = None,
                   symbols: set[str] | None = None,
                   my_buys: list[dict] | None = None) -> dict | None:
    cfg = _lineage_cfg(lin)
    fields = list(_rule_fields())
    data = _ask_json(cfg.get("model"),
        "You are a trading agent helping write your pool's shared rulebook. JSON only.",
        {"you": lin["name"], "your_record": f"{lin['wins']}W-{lin['losses']}L",
         "your_notes_from_your_own_trades": cfg.get("notes", ""),
         "your_trades_last_round": trades or [],
         "your_buys_last_round_with_the_data_you_saw_and_outcome": [
             {"symbol": b["symbol"], "outcome_pct": round(b["ret"], 3),
              **{f: b["features"].get(k) for f, k in _rule_fields().items()}}
             for b in (my_buys or [])][:20],
         "arena": ARENA_FACTS,
         "current_rules": [{"id": g["id"],
                            "rule": f"{'DO' if g['practice'] == 'good' else 'AVOID'}: {g['text']}",
                            "evidence": evidence_label(json.loads(g["evidence"]) if g.get("evidence") else None)}
                           for g in active],
         "instruction": (
             "Propose at most ONE change to the rulebook every agent will follow. "
             "Rules are ENTRY rules over the data-table columns, and the arena "
             "will BACKTEST yours against every buy the pool has logged — a rule "
             "the data contradicts is rejected automatically, so propose only a "
             "pattern your own buys above actually show. Either a DO (practice "
             "'good': buying when the condition holds tends to work) or an AVOID "
             "(practice 'bad': buying when it holds tends to lose). Conditions: "
             f"1-3 of [field, op, value], field one of {fields}, op one of "
             "<, <=, >, >=. Or remove a current rule your experience contradicts. "
             "Most rounds deserve no new rule — say none if nothing is clear. "
             "Return JSON {\"kind\":\"add\"|\"remove\"|\"none\",\"practice\":\"good\"|\"bad\","
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
    spec = parse_spec(data.get("when"))
    if not spec:
        return None
    practice = "bad" if data.get("practice") == "bad" else "good"
    return {"kind": "add", "practice": practice, "text": spec_text(spec, practice),
            "spec": spec, "guideline_id": None, "proposer": lin["name"],
            "evidence": str(data.get("why", ""))[:300]}


def _agent_vote(lin: dict, proposal: dict, active: list[str],
                trades: list[dict] | None = None, ledger: dict | None = None) -> tuple[str, str]:
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
         "arena_backtest_on_all_logged_buys": evidence_label(ledger) if ledger else
             "not applicable",
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


def _spec_key(p: dict):
    if p["kind"] == "remove":
        return ("rm", p["guideline_id"])
    return (p["practice"], json.dumps(sorted(p["spec"]["when"])))


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
    with ThreadPoolExecutor(max_workers=len(lineages)) as ex:
        raw = list(ex.map(lambda l: _agent_propose(l, active, trades[l["id"]], symbols,
                                                   my_buys[l["id"]]), lineages))

    seen = {(g["practice"], json.dumps(sorted(json.loads(g["spec"])["when"])))
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

    outcomes = buy_outcomes(conn)
    results = []
    for p in proposals[:config.max_proposals_per_round]:
        if p["kind"] == "add" and len(active_guidelines(conn)) >= config.max_active_guidelines:
            continue   # rulebook full — only removals can make room
        ledger = backtest(p["spec"], p["practice"], outcomes) if p.get("spec") else None
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
        if p["kind"] == "add" and verdict == "inconclusive":
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
            r["note"] = (f"vetoed: no evidence — {ledger['n_match']} logged buys match "
                         f"this condition (need {_min_probation()})")
            continue
        texts = active_texts(conn)

        def _v(l):
            if l["name"] == p["proposer"]:
                return "agree", "proposer"
            return _agent_vote(l, p, texts, trades[l["id"]], ledger)
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
               "vetoed" if r.get("note") else "rejected")
        conn.execute("UPDATE guideline_proposals SET resolution=?, resolved_at=? WHERE id=?",
                     (res, _now(), r["proposal_id"]))
        r["resolution"] = res
        r.pop("_p")
    conn.commit()
    return results + reviews
