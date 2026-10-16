# PIT — Agent Trading Arena

*Named after the trading pit — the open-outcry floor at an exchange.*

Three autonomous LLM traders (RONIN, VIPER, LYNX), each on its own model, get
the same paper capital, the same market data and the same clock. They trade
the same round at the same time; at the deadline they're ranked by return.
The winner reinforces what worked; everyone else is rebuilt from its own
self-critique. Between rounds the pool writes, researches and debates a shared
rulebook. Paper trading only — no real orders anywhere.

## How a round works

1. **Charters.** Each agent trades by a *charter* it wrote itself — a style,
   plus a checkable version (entry/exit conditions over the data table's
   columns). Charters must differ from rivals'. Before trading, the arena
   backtests each charter on ~60 days of real 5-minute market data after
   costs: the agent tunes on the earlier 40 days and is scored on the later,
   unseen 20 (catches overfitting). A revision can't replace a non-losing
   charter with a losing one.
2. **Risk bands.** The portfolio stop, take-profit and each stock's default
   stop are sized from the round's length and real volatility
   (sigma × sqrt(time)), so a 30-minute round and a full session each get bands
   that matter.
3. **Decisions.** Every few minutes each agent gets ONE data table for the
   whole universe — today's intraday moves (open%, vsVWAP%, 15m/30m%, day
   range, volume pace), the daily view (returns, moving averages, RSI,
   52-week position, volatility, auto-stop) and fundamentals — plus its own
   book and records, and makes one decision. Rows are in random order; every
   agent sees the same table.
4. **Execution.** Orders fill at the current quote plus half the spread
   (3 bps for US stocks, 8 for crypto). Every buy carries a stop. Optionally
   (`PIT_CHARTER_ENFORCE=1`) a buy that doesn't meet the agent's own charter
   entry conditions is rejected.
5. **Rivals.** Agents see rivals' live returns and last-round lessons — not their holdings, trades or taunts.
   Taunts are for spectators only; stock names in lessons are hidden.
6. **Feedback.** Each decision shows the agent its own sell record (did its
   sells beat holding?), its on- vs off-charter buys, and stops that fired.
7. **Resolution.** Everything is closed at the deadline; the pool is ranked
   (ties break on fewer trades, then lower drawdown — never a draw). ELO
   moves pairwise, scaled by how much market the round covered (a 30-minute
   round counts ~0.28x a full session). Stakes are flat: every round starts
   everyone at the same capital.
8. **Reflection.** Each agent critiques its own trades with its own model
   (each of its sells scored against holding to the deadline) and keeps,
   refines or rewrites its charter.
9. **Rulebook.** Each agent may propose one machine-checkable rule ("AVOID
   buying when vsVWAP% ≥ 1.5"). The arena researches it on the 60-day
   history (day-clustered t, search days + unseen days) and on the pool's own
   trades; contradicted, no-edge, duplicate or evidence-free proposals are
   vetoed. Survivors are debated — each rival raises an objection and may ask
   for a split test the arena runs — then voted: a proven edge needs a pool
   majority; a weak or untested lean needs every other agent and enters on
   probation. Active rules are re-tested every round and retired when the data
   turns against them.

## Quickstart

```bash
cd pit
pip install -r requirements.txt
cp .env.example .env        # add at least one provider key (see below)
scripts/run_arena.sh --mode live --minutes 60 --interval 300 --battles 1
```

Dashboard: http://127.0.0.1:5001 (leaderboard, rulebook with debates) and
`/live` (the round in progress: race chart vs SPY, cards with charters and
backtests, positions, fills, messages, the full decision audit log).

Replay a recent trading day (any of the last ~60 sessions) compressed into a
few minutes — the agents get exactly the live table, rebuilt from history:

```bash
scripts/run_arena.sh --mode replay --date 2026-10-09 --compress 10 --battles 1
```

`--reset` wipes the database first. See [RUNBOOK.md](RUNBOOK.md) for what to
expect while it runs.

## Models

Each lineage's model is set in `.env` with a provider prefix; an existing agent
switches at the next round.

```
PIT_RONIN_MODEL=anthropic:claude-sonnet-5-5   # ANTHROPIC_API_KEY  (PIT_CLAUDE_EFFORT=medium|high)
PIT_VIPER_MODEL=deepseek:deepseek-reasoner    # DEEPSEEK_API_KEY
PIT_LYNX_MODEL=deepseek:deepseek-chat        # DEEPSEEK_API_KEY (gemini:… needs a billed key)
```

Also supported: `openrouter:<model>` (OPENROUTER_API_KEY), `ollama:<model>`
(local), or a bare model id for Groq (GROQ_API_KEY). An agent whose provider
key is missing simply holds. Claude usage and estimated cost are logged to
`data/claude_usage.jsonl`.

## Settings worth knowing

| Variable | Default | Meaning |
|---|---|---|
| `PIT_MARKET` / `PIT_TRADE_UNIVERSE` | `us` / `top100` | or `crypto` / `crypto` |
| `PIT_BASE_CAPITAL` | 100000 | paper capital per agent per round |
| `PIT_SPREAD_BPS` | 3 (US), 8 (crypto) | half-spread paid on every fill |
| `PIT_CHARTER_ENFORCE` | 0 | 1 = reject buys outside the agent's own charter |
| `PIT_CHARTER_ITERS` | 3 | pre-round backtest→refine passes per charter |
| `PIT_STOP_SIGMA` / `PIT_PORTFOLIO_STOP_SIGMA` | 2 / 3 | stop widths in sigmas over the round |
| `PIT_QUOTE_REFRESH_SEC` | 120 | price cache; 15 is sensible for live |

## Code map

```
cli.py ─ live.py            the round loop: ticks, marks, stops, resolution trigger
           ├─ autonomous.py one agent decision (prompt + table -> orders)
           ├─ brief.py      the live data table (daily + intraday + fundamentals)
           ├─ replay.py     a past session served from history (same table)
           ├─ history.py    60-day 5-minute market panel: rule research, charter backtests
           ├─ charter.py    agents' own strategies: write, backtest, search, enforce
           ├─ guidelines.py the rulebook: review, propose, research, debate, vote
           ├─ forward.py    money and agents: fills, stops, resolution, reflection
           ├─ risk.py       volatility-scaled bands      market.py  quotes, fundamentals
           ├─ resolve.py / rating.py   tie-break cascade, ELO
           └─ llm.py        providers behind llm_chat(model, messages)
db.py / schema.sql (SQLite)        web.py / queries.py / templates  — read-only dashboard
```

## Tests

```bash
python3 -m pytest -q        # ~100 tests, offline, a few seconds
```

Tests never call a model provider or download market data (keys are blanked
in `tests/conftest.py`).
