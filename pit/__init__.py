"""PIT — an agent-vs-agent trading arena.

Two (later a pool of) LLM-driven trading agents duel with shared rules:
blind during a round, the loser's next generation is seeded from the winner's
trade log, stakes and ELO shift on win/loss, rounds shrink over time, and ties
are mechanically impossible.

This package is the Phase 1 paper-trading core. It is designed to run
end-to-end offline (synthetic price feed + deterministic policy) and to
upgrade in place to Groq-driven agents and real historical/broker data.
"""

__version__ = "0.1.0"
