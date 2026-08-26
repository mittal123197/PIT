"""The vocabulary an agent policy speaks.

A policy (deterministic or LLM-driven) returns a list of these per wake. They
map one-to-one to the tool-calls the LLM agent will expose in Phase 2:
place_order / set_watch / set_heartbeat / hold.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PlaceOrder:
    symbol: str
    side: str          # "buy" | "sell"
    qty: float
    reason: str | None = None


@dataclass
class SetWatch:
    symbol: str
    trigger_type: str  # "pct_move" | "price_above" | "price_below"
    threshold: float
    reason: str | None = None


@dataclass
class SetHeartbeat:
    minutes: int
    reason: str | None = None


@dataclass
class Hold:
    reason: str | None = None


Action = PlaceOrder | SetWatch | SetHeartbeat | Hold
