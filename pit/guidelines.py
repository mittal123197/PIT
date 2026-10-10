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
    return [f"{'DO' if g['practice'] == 'good' else 'AVOID'}: {g['text']}"
            for g in active_guidelines(conn)]


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


def _apply(conn, proposal: dict) -> None:
    if proposal["kind"] == "add":
        version = conn.execute(
            "SELECT COALESCE(MAX(version),0)+1 v FROM guidelines"
        ).fetchone()["v"]
        conn.execute(
            "INSERT INTO guidelines (text, practice, status, version, created_at) "
            "VALUES (?, ?, 'active', ?, ?)",
            (proposal["text"], proposal.get("practice", "good"), version, _now()),
        )
    else:  # remove
        conn.execute(
            "UPDATE guidelines SET status='retired', retired_at=? WHERE id=?",
            (_now(), proposal["guideline_id"]),
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


def _agent_propose(lin: dict, active: list[dict], trades: list[dict] | None = None,
                   symbols: set[str] | None = None) -> dict | None:
    cfg = _lineage_cfg(lin)
    data = _ask_json(cfg.get("model"),
        "You are a trading agent helping write your pool's shared rulebook. JSON only.",
        {"you": lin["name"], "your_record": f"{lin['wins']}W-{lin['losses']}L",
         "your_notes_from_your_own_trades": cfg.get("notes", ""),
         "your_trades_last_round": trades or [],
         "arena": ARENA_FACTS,
         "current_rules": [{"id": g["id"],
                            "rule": f"{'DO' if g['practice'] == 'good' else 'AVOID'}: {g['text']}"}
                           for g in active],
         "instruction": (
             "Based ONLY on your own experience, propose at most ONE change to the "
             "rulebook every agent in the pool will see: a good practice to DO, a "
             "bad practice to AVOID, or removing a current rule your experience "
             "contradicts. A rule must be GENERAL — it must apply to any stock or "
             "coin, so NEVER name a ticker, coin or company. It must be concrete "
             "and actionable: a decision rule of the form 'when <condition you can "
             "read in the data table or your book>, <buy / sell / size / don't "
             "buy>' — words like 'monitor' or 'consider' alone are not rules. ONE "
             "idea in one sentence, possible in this arena (see arena facts), and not repeat "
             "a current rule. Generic advice ('trade carefully') is useless. Most "
             "rounds deserve no new rule — if nothing is clearly worth it, say "
             "none. Return JSON "
             "{\"kind\":\"add\"|\"remove\"|\"none\",\"practice\":\"good\"|\"bad\","
             "\"text\":\"the rule, <=160 chars\",\"remove_id\":N,\"why\":\"...\"}")})
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
                "evidence": str(data.get("why", ""))[:300]}
    text, practice = _normalise_rule(str(data.get("text", "")),
                                     "bad" if data.get("practice") == "bad" else "good")
    if len(text) < 12 or _names_a_ticker(text, symbols or set()):
        return None
    if _vague(text):
        return None
    return {"kind": "add", "practice": practice,
            "text": text[:200], "guideline_id": None, "proposer": lin["name"],
            "evidence": str(data.get("why", ""))[:300]}


def _agent_vote(lin: dict, proposal: dict, active: list[str],
                trades: list[dict] | None = None) -> tuple[str, str]:
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
         "instruction": (
             "This rule would bind every agent, including you, from now on — a bad "
             "rule costs everyone. Be a skeptic: most proposals should be "
             "rejected. Agree ONLY if you can point to something specific in your "
             "own trades where following it would have improved your result. "
             "Disagree if it is vague, obvious, already covered, impossible in this "
             "arena, or not supported by your own experience. Return JSON "
             "{\"evidence_from_my_trades\":\"...\",\"vote\":\"agree\"|\"disagree\","
             "\"reason\":\"<=200 chars\"}")})
    if not data or data.get("vote") not in ("agree", "disagree"):
        return "abstain", "no valid vote returned"
    ev = str(data.get("evidence_from_my_trades", "")).strip()
    if data["vote"] == "agree" and len(ev) < 15:
        return "disagree", "agreed without citing own evidence — counted as no"
    return data["vote"], (str(data.get("reason", "")) + (f" | evidence: {ev}" if ev else ""))[:300]


def pool_constitution(conn, config: ArenaConfig, source_round_id: int | None) -> list[dict]:
    """Every agent may propose one rule change; every agent votes on each.
    Returns one summary dict per proposal voted on (possibly empty)."""
    from concurrent.futures import ThreadPoolExecutor
    lineages = current_lineages(conn)
    if not lineages:
        return []
    active = active_guidelines(conn)
    symbols = _universe_symbols()
    trades = {l["id"]: _round_trades(conn, source_round_id, l["id"]) for l in lineages}
    with ThreadPoolExecutor(max_workers=len(lineages)) as ex:
        raw = list(ex.map(lambda l: _agent_propose(l, active, trades[l["id"]], symbols),
                          lineages))

    seen = {_norm(g["text"]) for g in active}
    proposals = []
    for p in raw:
        if not p:
            continue
        key = ("rm", p["guideline_id"]) if p["kind"] == "remove" else _norm(p["text"])
        if p["kind"] == "add" and key in seen:
            continue
        if key in {(("rm", q["guideline_id"]) if q["kind"] == "remove" else _norm(q["text"]))
                   for q in proposals}:
            continue
        proposals.append(p)

    results = []
    for p in proposals[:config.max_proposals_per_round]:
        n_active = len(active_guidelines(conn))
        if p["kind"] == "add" and n_active >= config.max_active_guidelines:
            continue   # rulebook full — only removals can make room
        pid = conn.execute(
            """INSERT INTO guideline_proposals
               (kind, practice, guideline_id, proposed_text, source_round_id,
                created_at, proposer)
               VALUES (?,?,?,?,?,?,?)""",
            (p["kind"], p["practice"], p.get("guideline_id"), p["text"],
             source_round_id, _now(), p["proposer"])).lastrowid
        conn.commit()
        texts = active_texts(conn)
        # the proposer votes for its own proposal (no need to ask it); the
        # others are asked — so a rule needs at least one OTHER agent's support
        def _v(l):
            if l["name"] == p["proposer"]:
                return "agree", "proposer"
            return _agent_vote(l, p, texts, trades[l["id"]])
        with ThreadPoolExecutor(max_workers=len(lineages)) as ex:
            votes = list(ex.map(_v, lineages))
        for lin, (vote, reason) in zip(lineages, votes):
            if vote == "abstain":
                continue
            conn.execute(
                """INSERT OR REPLACE INTO guideline_votes
                   (proposal_id, lineage_id, vote, reasoning, created_at)
                   VALUES (?,?,?,?,?)""",
                (pid, lin["id"], vote, reason, _now()))
        agree = sum(1 for v, _ in votes if v == "agree")
        # strict majority of the WHOLE pool — two abstentions can't let one
        # agent's vote write a rule for everyone
        passed = agree * 2 > len(lineages)
        results.append({"proposal_id": pid, "kind": p["kind"], "practice": p["practice"],
                        "text": p["text"], "proposer": p["proposer"],
                        "evidence": p.get("evidence", ""), "agree": agree,
                        "total": len(lineages), "accepted": passed, "_p": p})
    # Rule inflation guard: adopt at most N per round — the ones with the most
    # support (ties: whoever proposed first). Without it the 7B agents voted
    # in nearly every proposal and the rulebook grew by ~3 rules a round.
    winners = sorted((r for r in results if r["accepted"]),
                     key=lambda r: -r["agree"])[:config.max_adoptions_per_round]
    keep = {r["proposal_id"] for r in winners}
    for r in results:
        if r["accepted"] and r["proposal_id"] not in keep:
            r["accepted"], r["note"] = False, "passed but over this round's adoption limit"
        if r["accepted"]:
            _apply(conn, r["_p"])
        res = ("accepted" if r["accepted"] else
               "over_limit" if r.get("note") else "rejected")
        conn.execute("UPDATE guideline_proposals SET resolution=?, resolved_at=? WHERE id=?",
                     (res, _now(), r["proposal_id"]))
        r["resolution"] = res
        r.pop("_p")
    conn.commit()
    return results
