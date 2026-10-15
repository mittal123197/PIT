"""Market-history research: rules are judged on many market snapshots
(day-clustered t), and a charter spec is replayed as round trips after costs."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pandas as pd
from pit import history, guidelines as g


def _panel(days=20, syms=8, edge=0.3, seed=0):
    """Synthetic panel: snapshots with vsVWAP% < -1 are followed by +edge% over
    the next 30 min; everything else is noise around 0."""
    rng = np.random.default_rng(seed)
    rows = []
    for d in range(days):
        for s in range(syms):
            for bar in range(0, 72):
                vw = rng.normal(0, 1)
                f30 = rng.normal(edge if vw < -1 else 0, 0.2)
                rows.append({"sym": f"S{s}", "day": d, "bar": bar, "px": 100 + bar * 0.01,
                             "vwap": vw, "f30": f30, "rsi": 50.0})
    return pd.DataFrame(rows)


def test_rule_with_real_edge_is_supported_and_noise_is_not():
    df = _panel()
    good = history.research_rule(g.parse_spec([["vsVWAP%", "<", -1]]), "good", df)
    assert good["verdict"] == "supported" and good["days"] == 20
    noise = history.research_rule(g.parse_spec([["RSI14", ">", 60]]), "good", df)
    assert noise is not None and noise["verdict"] != "supported"


def test_position_fields_cannot_be_researched_on_history():
    df = _panel(days=2)
    assert history.research_rule(g.parse_spec([["pos_pnl%", ">", 0]], "sell"), "bad", df) is None


def test_charter_that_never_triggers_says_so():
    df = _panel(days=3)
    card = history.backtest_charter({"entry": [["vsVWAP%", "<", -99]], "exit": [],
                                     "max_hold_min": 30}, df)
    assert card["trades"] == 0 and card["verdict"] == "never triggers"
