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
    return [g["text"] for g in active_guidelines(conn)]


def current_lineages(conn) -> list[dict]:
    return [dict(r) for r in conn.execute(
        """SELECT l.id, l.name, l.wins, l.losses, a.strategy_config
           FROM lineages l JOIN agents a ON a.id = l.current_agent_id
           ORDER BY l.id"""
    ).fetchall()]


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
        got = _llm_draft(conn, config)
        if got is not None:
            return got  # includes explicit None only via fallback below

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
        return {"kind": "add", "tag": "fewer_trades",
                "text": f"Prefer fewer, higher-conviction trades — winners "
                        f"out-traded by fewer fills in {fewer*100:.0f}% of the "
                        f"last {n} rounds.",
                "guideline_id": None,
                "evidence": f"fewer-trade winners {fewer*100:.0f}% of {n} rounds"}
    if lower_dd >= frac and "low_drawdown" not in active:
        return {"kind": "add", "tag": "low_drawdown",
                "text": f"Control drawdown tightly — winners held a smaller max "
                        f"drawdown in {lower_dd*100:.0f}% of the last {n} rounds.",
                "guideline_id": None,
                "evidence": f"lower-drawdown winners {lower_dd*100:.0f}% of {n} rounds"}

    # REMOVE if a codified pattern has clearly stopped holding.
    if "fewer_trades" in active and fewer < (1 - frac):
        gid, text = active["fewer_trades"]
        return {"kind": "remove", "tag": "fewer_trades", "text": text,
                "guideline_id": gid,
                "evidence": f"fewer-trade winners only {fewer*100:.0f}% lately"}
    if "low_drawdown" in active and lower_dd < (1 - frac):
        gid, text = active["low_drawdown"]
        return {"kind": "remove", "tag": "low_drawdown", "text": text,
                "guideline_id": gid,
                "evidence": f"lower-drawdown winners only {lower_dd*100:.0f}% lately"}
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
           (kind, guideline_id, proposed_text, source_round_id, created_at)
           VALUES (?,?,?,?,?)""",
        (proposal["kind"], proposal.get("guideline_id"), proposal["text"],
         source_round_id, _now()),
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
            "INSERT INTO guidelines (text, status, version, created_at) "
            "VALUES (?, 'active', ?, ?)",
            (proposal["text"], version, _now()),
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
