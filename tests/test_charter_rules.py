"""Charters: a buy must meet the agent's own entry conditions when
enforcement is on, and a rewrite that LOSES in backtests can't replace a
charter that didn't."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pit import charter

SPEC = {"entry": [["vsVWAP%", "<", -1.5], ["RSI14", "<", 40]], "exit": [], "max_hold_min": 30}


def test_entry_check_names_the_failing_condition():
    assert charter.entry_check(SPEC, {"vwap": -2.0, "rsi": 30}) is None
    why = charter.entry_check(SPEC, {"vwap": -0.2, "rsi": 30})
    assert "vsVWAP% is -0.20" in why and "< -1.5" in why
    assert "not available" in charter.entry_check(SPEC, {"vwap": -2.0})
    assert charter.entry_check(None, {}) is None                     # no spec -> nothing to enforce


def test_losing_rewrite_cannot_replace_a_neutral_charter(monkeypatch):
    neutral = {"train": {"trades": 40, "t": 0.5}}
    losing = {"train": {"trades": 40, "t": -3.0}}
    monkeypatch.setattr(charter, "backtest", lambda spec: losing)
    cfg = {"charter": "old style", "charter_spec": SPEC, "charter_card": neutral}
    assert charter.accept(cfg, {"charter": "new losing style", "spec": SPEC}) is False
    assert cfg["charter"] == "old style"
    monkeypatch.setattr(charter, "backtest", lambda spec: {"train": {"trades": 40, "t": 1.0}})
    assert charter.accept(cfg, {"charter": "better style", "spec": SPEC}) is True
    assert cfg["charter_last_good"]["charter"] == "better style"


def test_losing_charter_is_not_enforced(monkeypatch):
    from pit import charter
    monkeypatch.setenv("PIT_CHARTER_ENFORCE", "1")
    spec = {"entry": [("RSI14", "<", 30.0)], "exit": [], "max_hold_min": 60}
    assert charter.enforced({"charter_spec": spec, "charter_card": {"train": {"t": 0.5}}})
    assert not charter.enforced({"charter_spec": spec, "charter_card": {"train": {"t": -3.1}}})
    monkeypatch.setenv("PIT_CHARTER_ENFORCE", "0")
    assert not charter.enforced({"charter_spec": spec, "charter_card": None})
