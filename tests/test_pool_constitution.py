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


def _ledger(c, rsi_low_ret: float, rsi_high_ret: float, n: int = 6):
    """n buys at RSI 25 that returned rsi_low_ret (± noise) and n at RSI 60
    that returned rsi_high_ret — each closed by a later sell."""
    rid = forward.start_round(c, ArenaConfig(), days=0)
    aid = c.execute("SELECT agent_id FROM round_states WHERE round_id=?", (rid,)).fetchone()[0]
    for i in range(n):
        for rsi, ret in ((25, rsi_low_ret), (60, rsi_high_ret)):
            r = ret + (0.1 if i % 2 else -0.1)
            c.execute("INSERT INTO trades (round_id, agent_id, ts, symbol, side, qty, price, "
                      "capital_after, features) VALUES (?,?,?,?, 'buy', 1, 100, 0, ?)",
                      (rid, aid, "t", f"S{rsi}{i}", json.dumps({"rsi": rsi, "d1": 1.0})))
            c.execute("INSERT INTO trades (round_id, agent_id, ts, symbol, side, qty, price, "
                      "capital_after) VALUES (?,?,?,?, 'sell', 1, ?, 0)",
                      (rid, aid, "t", f"S{rsi}{i}", 100 * (1 + r / 100)))
    c.commit()


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


def test_untested_rule_needs_everyone_and_enters_on_probation(monkeypatch):
    c = _conn()                                                     # empty ledger
    monkeypatch.setattr(g, "_ask_json", _fake(
        {"RONIN": RSI_DO}, {("LYNX", "RSI14"): {"vote": "agree"}}, []))
    assert not g.pool_constitution(c, ArenaConfig(), None)[0]["accepted"]   # 2/3 not enough
    c2 = _conn()
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
    c = _conn()
    monkeypatch.setattr(g, "_ask_json", _fake(
        {"RONIN": RSI_DO}, {("LYNX", "RSI14"): {"vote": "agree"},
                            ("VIPER", "RSI14"): {"vote": "agree"}}, []))
    g.pool_constitution(c, ArenaConfig(), None)                    # adopted on probation
    _ledger(c, rsi_low_ret=-1.0, rsi_high_ret=1.0)
    c.execute("UPDATE rounds SET status='resolved'")
    events = g.review_rules(c)
    assert events and events[0]["action"] == "retired"
    assert g.active_texts(c) == []
