"""Mutation — how a losing lineage's next generation is born.

The loser studies the winner's trade log and produces a *mutated* strategy
config: it must change, not clone (deliberate: an exact copy would just re-lose
to a winner that has itself moved on — alpha decay). The winner never sees the
loser's log.

Phase 1 uses a deterministic mutation so the arena evolves with no LLM key. If
Groq is configured, `pit.llm.mutate_with_llm` is used instead (richer, but the
deterministic path stays the always-available fallback).
"""
from __future__ import annotations

import copy
import random


_NUMERIC_FIELDS = (
    "lookback", "buy_threshold_pct", "sell_threshold_pct",
    "position_frac", "band_pct", "heartbeat_minutes",
)


def mutate_config(
    winner_cfg: dict,
    loser_cfg: dict,
    winner_trades: list[dict],
    seed: int = 0,
) -> tuple[dict, str]:
    """Return (new_loser_config, human-readable note). Deterministic given seed."""
    rng = random.Random(seed)
    new = copy.deepcopy(loser_cfg)
    notes: list[str] = []

    # Clean transient flags that the policy may have stamped on.
    new.pop("_announced_heartbeat", None)

    # 1. Learn the winner's *type* — but only sometimes, and never as a verbatim
    #    clone: adopting the approach while keeping some of your own parameters.
    if winner_cfg.get("type") and winner_cfg["type"] != loser_cfg.get("type"):
        if rng.random() < 0.6:
            new["type"] = winner_cfg["type"]
            notes.append(f"adopted winner approach '{winner_cfg['type']}'")

    # 2. Blend numeric params toward the winner, then perturb so it's a mutation
    #    rather than a copy.
    for field in _NUMERIC_FIELDS:
        if field not in winner_cfg and field not in loser_cfg:
            continue
        w = float(winner_cfg.get(field, loser_cfg.get(field, 0.0)))
        l = float(loser_cfg.get(field, w))
        blended = (w + l) / 2.0
        perturb = 1.0 + rng.uniform(-0.25, 0.25)  # +/-25% mutation
        value = blended * perturb
        if field in ("lookback", "heartbeat_minutes"):
            value = max(1, int(round(value)))
        else:
            value = round(value, 3)
        new[field] = value

    # 3. React to how active the winner was: if it out-traded the loser, get
    #    a bit more active (shorter heartbeat); if it barely traded, calm down.
    trade_ct = len(winner_trades)
    hb = int(new.get("heartbeat_minutes", loser_cfg.get("heartbeat_minutes", 60)))
    if trade_ct >= 6:
        hb = int(hb * 0.75)
        notes.append(f"winner traded {trade_ct}x -> tightening cadence")
    elif trade_ct <= 1:
        hb = int(hb * 1.5)
        notes.append(f"winner traded {trade_ct}x -> relaxing cadence")
    # keep cadence in a sane band so it can't run away to hundreds of minutes
    # (which would silently stop the agent from ever acting)
    new["heartbeat_minutes"] = max(15, min(240, hb))

    if not notes:
        notes.append("perturbed parameters toward winner")
    return new, "; ".join(notes)
