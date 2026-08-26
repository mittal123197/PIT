# PIT — Agent Trading Arena

Two (later a pool of) autonomous trading agents duel with shared rules:

- **Scoreboard visible, playbook hidden** — during a round each agent sees the
  opponent's live return % (so it knows if it's losing and fights back), but
  *not* the opponent's trades. You only get the winner's trade log *after* you
  lose — which is what preserves the copy-and-mutate mechanic.
- **Survival instinct** — fall behind on the scoreboard and an agent takes more
  risk to claw back (both the deterministic and LLM policies).
- **Lose and you're rebuilt from your killer** — the loser's next generation is
  seeded by *mutating* the winner's trade log (learn, don't clone → deliberate
  alpha decay).
- **Stakes are mechanical** — winner +20% capital next round, loser −20%, plus
  an ELO rating that weights opponent strength.
- **Rounds shrink** — 7 days, tightening by a day per cycle down to a 4-day floor.
- **No draws, ever** — a tie-break cascade (return → fewer trades → lower
  drawdown → sudden death) always yields a winner.

Phase 1 is a **paper-trading core that runs fully offline** — deterministic
strategies over a synthetic price feed — so the whole round → trade → resolve →
mutate loop is provable with zero credentials. Real NSE data (yfinance) and
Groq-driven LLM agents drop in behind the same interfaces.

## Quickstart

```bash
cd pit
python3 -m pit.cli init          # create the DB + seed two lineages
python3 -m pit.cli arena --rounds 5
python3 -m pit.cli leaderboard
python3 -m pit.cli history --round 1
```

No install needed for the offline path — the core uses only the standard
library. For the optional paths:

```bash
pip install -r requirements.txt   # yfinance (historical), groq (LLM), flask, pytest
```

### Dashboard

A read-only web view of the arena (leaderboard with ELO sparklines, per-round
trade ledgers, and each lineage's generation/mutation history):

```bash
python3 -m pit.web            # http://127.0.0.1:5001
```

It never writes, so it's safe to run against a live DB while the arena plays.

### Optional modes

| Mode | How | Needs |
|------|-----|-------|
| Offline (default) | `python3 -m pit.cli arena` | nothing |
| Real NSE replay | `... arena --feed historical` | `yfinance`, network |
| LLM agents + mutation | `PIT_USE_LLM=1 python3 -m pit.cli run-round --bars 12` | `groq`, `GROQ_API_KEY` |
| Agents aware of stakes | `PIT_AGENT_AWARE=1` (default) | — |

Copy `.env.example` → `.env` to configure keys and tuning.

**On LLM runs:** every agent wake is one Groq call, so keep the synthetic feed
short with `--bars` (e.g. 12) while experimenting. Default model is
`openai/gpt-oss-20b` (fast); set `PIT_LLM_MODEL=openai/gpt-oss-120b` for
stronger play at the cost of speed, and `PIT_LLM_REASONING=low|medium|high` to
trade latency for depth on gpt-oss models.

## Run the tests

```bash
python3 -m pytest tests/ -v      # or: python3 tests/test_resolve.py
```

`test_resolve.py` proves a winner is always produced (no draws); `test_engine.py`
runs a full round in memory and checks trades log, a winner emerges, and the
loser mutates into a *distinct* gen-2 config.

## How it fits together

```
cli.py ── Engine ─┬─ feeds.py     (the clock+market: Synthetic | Historical)
                  ├─ policies.py  (the brain: SimplePolicy | LLMPolicy)
                  ├─ broker.py    (execution: PaperBroker | [Dhan later])
                  ├─ resolve.py   (the no-draw cascade)   ← unit tested
                  ├─ rating.py    (ELO)
                  └─ mutate.py    (loser's next generation)   ↔ llm.py
                          state → db.py / schema.sql (SQLite)
```

Each seam is an interface so the paper→live swap is a config change, not a
rewrite:

- **Broker** — `PaperBroker` now; a `DhanBroker` slots in for Phase 4.
- **PriceFeed** — `SyntheticFeed`/`HistoricalFeed` now; a live Dhan feed later.
- **AgentPolicy** — `SimplePolicy` now; `LLMPolicy` (Groq) whenever a key is set.

## Design notes

- **Two independent evolution mechanisms.** *Per-lineage mutation* (loser learns
  from winner) and a *pool-level guideline "constitution"* changed only by vote
  (schema present; voting lands in Phase 2).
- **Agents choose their own activity.** Heartbeat cadence and price-alert watches
  are the agent's decision, honoured by a fine-grained tick loop — not a single
  global cron.
- **"Do agents know they'll die?"** LLM agents are told the stakes by default
  (`PIT_AGENT_AWARE=1`) — losing means being rebuilt from the winner. Flip it to
  `0` to compare aware vs blind play. The deterministic policy shows the same
  survival instinct mechanically (risk-up when trailing the scoreboard).
- **Ledger is ours.** Every buy/sell is stored in `trades` independent of any
  broker, so records (and later tax reporting) never depend on the broker's
  retention.

## Roadmap

1. **Phase 1 — Paper MVP (this).** Offline core, one duel, CLI.
2. **Phase 2 — Deploy + automate.** Always-on tick loop, guideline voting, web
   dashboard. (Fly.io free VM + persistent volume for the SQLite file.)
3. **Phase 3 — Pool.** N agents, ladder/round-robin matchmaking.
4. **Phase 4 — Live.** DhanHQ behind the `Broker` interface, real capital, ITR
   statements via Dhan↔ClearTax/Quicko.

See `../.claude/plans/eager-tickling-rocket.md` for the full plan.
