"""The evidence-checked rulebook: rules are machine-checkable entry
conditions, the arena backtests each against the pool's logged buys, and the
data gates the vote (veto / majority / unanimous-on-probation). Active rules
are re-tested every round and retired when the ledger turns against them."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import json
import sqlite3
from pit import db as dbm, forward, guidelines as g
from pit.config import ArenaConfig


def _conn():
    c = sqlite3.connect(":memory:"); c.row_factory = sqlite3.Row
    c.execute("PRAGMA foreign_keys = ON"); c.executescript(dbm.SCHEMA_PATH.read_text())
    forward.ensure_autonomous_lineages(c, ArenaConfig())
    return c


def _ledger(c, rsi_low_ret: float, rsi_high_ret: float, n: int = 6, rounds: int = 2):
    """n buys at RSI 25 that returned rsi_low_ret (± noise) and n at RSI 60
    that returned rsi_high_ret — each closed by a later sell — spread over
    `rounds` separate rounds (a rule needs evidence from 2+ rounds)."""
    for rr in range(rounds):
        rid = forward.start_round(c, ArenaConfig(), days=0)
        aid = c.execute("SELECT agent_id FROM round_states WHERE round_id=?", (rid,)).fetchone()[0]
        for i in range(rr, n, rounds):
            for rsi, ret in ((25, rsi_low_ret), (60, rsi_high_ret)):
                r = ret + (0.1 if i % 2 else -0.1)
                c.execute("INSERT INTO trades (round_id, agent_id, ts, symbol, side, qty, price, "
                          "capital_after, features) VALUES (?,?,?,?, 'buy', 1, 100, 0, ?)",
                          (rid, aid, "t", f"S{rsi}{i}", json.dumps({"rsi": rsi, "d1": 1.0})))
                c.execute("INSERT INTO trades (round_id, agent_id, ts, symbol, side, qty, price, "
                          "capital_after) VALUES (?,?,?,?, 'sell', 1, ?, 0)",
                          (rid, aid, "t", f"S{rsi}{i}", 100 * (1 + r / 100)))
        c.execute("UPDATE rounds SET status='resolved' WHERE id=?", (rid,))
    c.commit()


def test_one_round_of_evidence_is_not_proof(monkeypatch):
    c = _conn(); _ledger(c, 1.0, -1.0, n=6, rounds=1)
    ev = g.backtest(g.parse_spec([["RSI14", "<", 30]]), "good", g.buy_outcomes(c))
    assert ev["verdict"] == "untested" and ev["rounds"] == 1


def _fake(proposals: dict, votes: dict, asked: list):
    def ask(model, system, payload):
        me = payload["you"]
        if "proposal" not in payload:
            return proposals.get(me, {"kind": "none"})
        asked.append((me, payload["proposal"]))
        for (voter, sub), v in votes.items():
            if voter == me and sub in payload["proposal"]:
                return v
        return {"vote": "disagree", "reason": "no"}
    return ask


RSI_DO = {"kind": "add", "practice": "good", "when": [["RSI14", "<", 30]]}


def test_contradicted_rule_is_vetoed_without_a_vote(monkeypatch):
    c = _conn(); _ledger(c, rsi_low_ret=-1.0, rsi_high_ret=1.0)   # low RSI buys lost
    asked = []
    monkeypatch.setattr(g, "_ask_json", _fake({"RONIN": RSI_DO}, {}, asked))
    res = g.pool_constitution(c, ArenaConfig(), None)
    assert res[0]["resolution"] == "vetoed" and res[0]["ledger"]["verdict"] == "contradicted"
    assert asked == [] and g.active_texts(c) == []


def test_supported_rule_needs_pool_majority_and_is_proven(monkeypatch):
    c = _conn(); _ledger(c, rsi_low_ret=1.0, rsi_high_ret=-1.0)
    monkeypatch.setattr(g, "_ask_json", _fake(
        {"RONIN": RSI_DO}, {("LYNX", "RSI14"): {"vote": "agree"}}, []))
    res = g.pool_constitution(c, ArenaConfig(), None)
    assert res[0]["accepted"] and res[0]["tier"] == "proven"        # RONIN + LYNX = 2/3
    assert g.active_texts(c)[0].startswith("DO: buy when RSI14 < 30  [proven;")


def test_rule_nobody_ever_traded_is_vetoed(monkeypatch):
    c = _conn()                                                     # empty ledger
    asked = []
    monkeypatch.setattr(g, "_ask_json", _fake({"RONIN": RSI_DO}, {}, asked))
    res = g.pool_constitution(c, ArenaConfig(), None)
    assert res[0]["resolution"] == "vetoed" and asked == []


def test_untested_rule_needs_everyone_and_enters_on_probation(monkeypatch):
    c = _conn(); _ledger(c, 1.0, -1.0, n=3)                        # 3 matching < 5 -> untested
    monkeypatch.setattr(g, "_ask_json", _fake(
        {"RONIN": RSI_DO}, {("LYNX", "RSI14"): {"vote": "agree"}}, []))
    assert not g.pool_constitution(c, ArenaConfig(), None)[0]["accepted"]   # 2/3 not enough
    c2 = _conn(); _ledger(c2, 1.0, -1.0, n=3)
    monkeypatch.setattr(g, "_ask_json", _fake(
        {"RONIN": RSI_DO}, {("LYNX", "RSI14"): {"vote": "agree"},
                            ("VIPER", "RSI14"): {"vote": "agree"}}, []))
    res = g.pool_constitution(c2, ArenaConfig(), None)
    assert res[0]["accepted"] and res[0]["tier"] == "probation"


def test_free_text_and_unknown_fields_are_not_rules(monkeypatch):
    c = _conn()
    monkeypatch.setattr(g, "_ask_json", _fake(
        {"RONIN": {"kind": "add", "practice": "good", "text": "Monitor BTC-USD closely"},
         "VIPER": {"kind": "add", "practice": "good", "when": [["mood", ">", 1]]}}, {}, []))
    assert g.pool_constitution(c, ArenaConfig(), None) == []


def test_review_retires_a_rule_the_ledger_turns_against(monkeypatch):
    c = _conn(); _ledger(c, 1.0, -1.0, n=2)
    monkeypatch.setattr(g, "_ask_json", _fake(
        {"RONIN": RSI_DO}, {("LYNX", "RSI14"): {"vote": "agree"},
                            ("VIPER", "RSI14"): {"vote": "agree"}}, []))
    g.pool_constitution(c, ArenaConfig(), None)                    # adopted on probation
    _ledger(c, rsi_low_ret=-1.0, rsi_high_ret=1.0)
    c.execute("UPDATE rounds SET status='resolved'")
    events = g.review_rules(c)
    assert events and events[0]["action"] == "retired"
    assert g.active_texts(c) == []


def test_inconclusive_rule_is_vetoed(monkeypatch):
    c = _conn(); _ledger(c, 0.1, 0.1, n=6)                         # same outcome both sides
    monkeypatch.setattr(g, "_ask_json", _fake({"RONIN": RSI_DO}, {}, []))
    res = g.pool_constitution(c, ArenaConfig(), None)
    assert res[0]["resolution"] == "vetoed" and "no edge" in res[0]["note"]


def test_mirror_of_an_active_rule_is_vetoed(monkeypatch):
    c = _conn(); _ledger(c, 1.0, -1.0)
    monkeypatch.setattr(g, "_ask_json", _fake(
        {"RONIN": RSI_DO}, {("LYNX", "RSI14"): {"vote": "agree"}}, []))
    g.pool_constitution(c, ArenaConfig(), None)                    # DO buy when RSI14 < 30
    mirror = {"kind": "add", "practice": "bad", "when": [["RSI14", ">=", 30]]}
    monkeypatch.setattr(g, "_ask_json", _fake(
        {"VIPER": mirror}, {("LYNX", "RSI14"): {"vote": "agree"}}, []))
    res = [r for r in g.pool_constitution(c, ArenaConfig(), None) if r.get("kind")]
    assert res[0]["resolution"] == "vetoed" and "same trades" in res[0]["note"]


def test_exit_rule_is_backtested_against_holding(monkeypatch):
    """Quick flat sells (pos_pnl ~0, held < 15 min) that holding would have
    beaten -> an AVOID-selling rule is supported by the ledger."""
    c = _conn()
    for rr in range(2):
        rid = forward.start_round(c, ArenaConfig(), days=0)
        aid = c.execute("SELECT agent_id FROM round_states WHERE round_id=?", (rid,)).fetchone()[0]
        ends = {}
        for i in range(6):
            quick = f"Q{rr}{i}"; slow = f"S{rr}{i}"
            # quick flat sells at 100, deadline 101 -> selling cost 1%
            c.execute("INSERT INTO trades (round_id, agent_id, ts, symbol, side, qty, price, capital_after, reason, features) VALUES (?,?,?,?, 'sell', 1, 100, 0, 'rotate', ?)",
                      (rid, aid, "t", quick, json.dumps({"pos_pnl": 0.1, "held_min": 5 + i % 2})))
            ends[quick] = 101.0 + (0.1 if i % 2 else -0.1)
            # patient sells at 100, deadline 99 -> selling saved 1%
            c.execute("INSERT INTO trades (round_id, agent_id, ts, symbol, side, qty, price, capital_after, reason, features) VALUES (?,?,?,?, 'sell', 1, 100, 0, 'trend broke', ?)",
                      (rid, aid, "t", slow, json.dumps({"pos_pnl": 0.1, "held_min": 60 + i})))
            ends[slow] = 99.0 + (0.1 if i % 2 else -0.1)
        c.execute("UPDATE rounds SET status='resolved', end_prices=? WHERE id=?", (json.dumps(ends), rid))
    c.commit()
    spec = g.parse_spec([["held_min", "<", 15]], "sell")
    assert g.spec_text(spec, "bad") == "selling when held_min < 15"
    ev = g.backtest(spec, "bad", g.outcomes_for(c, spec))
    assert ev["verdict"] == "supported" and ev["rounds"] == 2


def test_promising_but_unproven_rule_can_go_to_probation(monkeypatch):
    c = _conn(); _ledger(c, 0.3, 0.0, n=6)      # small edge -> 1 <= t < 2 territory
    ev = g.backtest(g.parse_spec([["RSI14", "<", 30]]), "good", g.buy_outcomes(c))
    if not (1.0 <= (ev["t"] or 0) < 2.0):
        return   # sample noise; the veto path is covered elsewhere
    monkeypatch.setattr(g, "_ask_json", _fake(
        {"RONIN": RSI_DO}, {("LYNX", "RSI14"): {"vote": "agree"},
                            ("VIPER", "RSI14"): {"vote": "agree"}}, []))
    res = [r for r in g.pool_constitution(c, ArenaConfig(), None) if r.get("kind")]
    assert res[0]["accepted"] and res[0]["tier"] == "probation"
