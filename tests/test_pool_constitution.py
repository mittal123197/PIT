"""The agent-driven rulebook: each agent proposes (own model), every other
agent votes; strict majority of the whole pool; no ticker-specific rules;
practice labels normalised; no rubber-stamp 'agree' without evidence."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import sqlite3
from pit import db as dbm, forward, guidelines as g
from pit.config import ArenaConfig


def _conn():
    c = sqlite3.connect(":memory:"); c.row_factory = sqlite3.Row
    c.execute("PRAGMA foreign_keys = ON"); c.executescript(dbm.SCHEMA_PATH.read_text())
    forward.ensure_autonomous_lineages(c, ArenaConfig())
    return c


def _fake(proposals: dict, votes: dict):
    """proposals: name -> proposal JSON; votes: (voter, rule substring) -> vote JSON"""
    def ask(model, system, payload):
        me = payload["you"]
        if "proposal" not in payload:
            return proposals.get(me, {"kind": "none"})
        for (voter, sub), v in votes.items():
            if voter == me and sub in payload["proposal"]:
                return v
        return {"vote": "disagree", "reason": "no"}
    return ask


def test_majority_of_whole_pool_and_proposer_counts_as_agree(monkeypatch):
    c = _conn()
    monkeypatch.setattr(g, "_universe_symbols", lambda: {"BTC-USD", "BTC"})
    monkeypatch.setattr(g, "_ask_json", _fake(
        {"RONIN": {"kind": "add", "practice": "good", "text": "Size down when RSI14 is above 75"},
         "VIPER": {"kind": "add", "practice": "good", "text": "Never hold through a 20d MA break"}},
        {("LYNX", "RSI14"): {"vote": "agree", "evidence_from_my_trades": "bought at RSI 80 and lost 2%"},
         ("VIPER", "RSI14"): {"vote": "disagree", "reason": "no"}}))
    res = {r["text"]: r for r in g.pool_constitution(c, ArenaConfig(), None)}
    assert res["Size down when RSI14 is above 75"]["accepted"]            # RONIN + LYNX = 2/3
    assert not res["hold through a 20d MA break"]["accepted"]             # only VIPER
    assert res["hold through a 20d MA break"]["practice"] == "bad"        # 'Never X' -> AVOID X
    assert g.active_texts(c) == ["DO: Size down when RSI14 is above 75"]


def test_ticker_specific_rules_are_dropped_and_bare_agree_counts_as_no(monkeypatch):
    c = _conn()
    monkeypatch.setattr(g, "_universe_symbols", lambda: {"BTC-USD", "BTC"})
    monkeypatch.setattr(g, "_ask_json", _fake(
        {"RONIN": {"kind": "add", "practice": "good", "text": "Buy BTC-USD whenever it dips"},
         "VIPER": {"kind": "add", "practice": "good", "text": "Take profit after a 5% gain in one wake-up"}},
        {("RONIN", "Take profit"): {"vote": "agree"},                     # no evidence -> no
         ("LYNX", "Take profit"): {"vote": "agree", "evidence_from_my_trades": ""}}))
    res = g.pool_constitution(c, ArenaConfig(), None)
    assert [r["text"] for r in res] == ["Take profit after a 5% gain in one wake-up"]
    assert not res[0]["accepted"] and res[0]["agree"] == 1
    assert g.active_texts(c) == []
