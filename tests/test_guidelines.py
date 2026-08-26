"""The shared-guideline mechanism: pattern detection, voting, resolution.

Uses controlled, injected round data so the assertions don't depend on the
stochastic feed — we're testing the constitution logic, not the strategies.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import sqlite3  # noqa: E402
from datetime import datetime, timezone  # noqa: E402

from pit import db as dbm  # noqa: E402
from pit import guidelines as g  # noqa: E402
from pit.config import ArenaConfig  # noqa: E402
from pit.engine import Engine  # noqa: E402


def _fresh():
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.executescript(dbm.SCHEMA_PATH.read_text())
    eng = Engine(conn, config=ArenaConfig())
    a = eng.create_lineage("A", {"type": "momentum", "heartbeat_minutes": 60})
    b = eng.create_lineage("B", {"type": "mean_reversion", "heartbeat_minutes": 90})
    return conn, a, b


def _agent_of(conn, lineage_id):
    return conn.execute(
        "SELECT current_agent_id FROM lineages WHERE id=?", (lineage_id,)
    ).fetchone()["current_agent_id"]


def _inject_round(conn, n, winner_agent, loser_agent, w_tc, l_tc, w_dd, l_dd):
    now = datetime.now(timezone.utc).isoformat()
    rid = conn.execute(
        """INSERT INTO rounds (round_number, agent_a_id, agent_b_id, start_at,
           deadline, length_days, goal_pct, status, created_at)
           VALUES (?,?,?,?,?,?,?, 'resolved', ?)""",
        (n, winner_agent, loser_agent, now, now, 5, 1.0, now),
    ).lastrowid
    for aid, tc, dd, ret in ((winner_agent, w_tc, w_dd, 2.0),
                             (loser_agent, l_tc, l_dd, -1.0)):
        conn.execute(
            """INSERT INTO round_states (round_id, agent_id, starting_capital,
               current_capital, trade_count, max_drawdown_pct, final_return_pct)
               VALUES (?,?,?,?,?,?,?)""",
            (rid, aid, 100000, 100000, tc, dd, ret),
        )
    conn.execute(
        """INSERT INTO round_results (round_id, winner_agent_id, loser_agent_id,
           resolution_reason, winner_return_pct, loser_return_pct, rating_delta,
           created_at) VALUES (?,?,?, 'return', 2.0, -1.0, 16.0, ?)""",
        (rid, winner_agent, loser_agent, now),
    )
    conn.commit()
    return rid


def test_pattern_detected_when_winners_trade_less():
    conn, a, b = _fresh()
    aw, bw = _agent_of(conn, a), _agent_of(conn, b)
    # 5 rounds where the winner always traded fewer times
    for i in range(1, 6):
        _inject_round(conn, i, aw, bw, w_tc=2, l_tc=9, w_dd=3.0, l_dd=8.0)
    prop = g.draft_proposal(conn, ArenaConfig(), use_llm=False)
    assert prop is not None
    assert prop["kind"] == "add" and prop["tag"] == "fewer_trades"


def test_no_proposal_without_a_pattern():
    conn, a, b = _fresh()
    aw, bw = _agent_of(conn, a), _agent_of(conn, b)
    # neither pattern reaches the 60% threshold: winner trades fewer in 2/5,
    # winner holds lower drawdown in 2/5.
    rounds = [
        # (w_tc, l_tc, w_dd, l_dd)
        (2, 9, 8.0, 3.0),   # fewer yes, lower-dd no
        (2, 9, 8.0, 3.0),   # fewer yes, lower-dd no
        (9, 2, 3.0, 8.0),   # fewer no,  lower-dd yes
        (9, 2, 3.0, 8.0),   # fewer no,  lower-dd yes
        (9, 2, 8.0, 3.0),   # fewer no,  lower-dd no
    ]
    for i, (w_tc, l_tc, w_dd, l_dd) in enumerate(rounds, 1):
        _inject_round(conn, i, aw, bw, w_tc, l_tc, w_dd, l_dd)
    assert g.draft_proposal(conn, ArenaConfig(), use_llm=False) is None


def test_unanimous_when_both_trailing_adopts():
    conn, a, b = _fresh()
    aw, bw = _agent_of(conn, a), _agent_of(conn, b)
    for i in range(1, 6):
        _inject_round(conn, i, aw, bw, w_tc=2, l_tc=9, w_dd=3.0, l_dd=8.0)
    # both lineages have losing records -> both agree to added guidance
    conn.execute("UPDATE lineages SET wins=1, losses=3")
    conn.commit()
    res = g.open_and_resolve(conn, ArenaConfig(), source_round_id=None)
    assert res["accepted"] is True and res["agree"] == 2 and res["total"] == 2
    assert len(g.active_texts(conn)) == 1


def test_split_vote_keeps_status_quo():
    conn, a, b = _fresh()
    aw, bw = _agent_of(conn, a), _agent_of(conn, b)
    for i in range(1, 6):
        _inject_round(conn, i, aw, bw, w_tc=2, l_tc=9, w_dd=3.0, l_dd=8.0)
    # one winning (resists), one losing (accepts) -> 1/2, tie fails
    conn.execute("UPDATE lineages SET wins=4, losses=1 WHERE id=?", (a,))
    conn.execute("UPDATE lineages SET wins=1, losses=4 WHERE id=?", (b,))
    conn.commit()
    res = g.open_and_resolve(conn, ArenaConfig(), source_round_id=None)
    assert res["accepted"] is False and res["agree"] == 1
    assert len(g.active_texts(conn)) == 0  # nothing adopted


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-v"]))
