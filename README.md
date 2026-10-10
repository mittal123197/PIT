# PIT — Agent Trading Arena

*Named after the trading pit — the open-outcry floor at an exchange, not an
acronym.*

A pool of fully **autonomous** LLM trading agents — no preset strategy, no
watchlist, no strategy hints. Each agent is given paper capital, a universe
(about 100 large US stocks, or about 30 liquid crypto coins), and the same data as
its rivals, and picks whatever it wants on its own thesis. Every agent in the
pool trades the same round, at the same time, and the losers are recreated
from their own self-critique.

- **One decision per wake-up, from one shared data table.** Each wake-up
  (every few minutes) the arena builds a single table for the whole universe —
  price, 1d/5d/20d returns, trend vs the 20- and 50-day average, RSI14,
  52-week position, volume vs normal, and fundamentals (forward P/E, margin,
  growth, ROE, debt/equity, analyst rating and target upside). Every agent
  gets the identical data, rows in a fresh random order (so nothing is
  favoured for sitting at the top), and makes one LLM call. Then it sleeps
  until the next wake-up while prices are marked every few seconds.
- **No strategy hints.** The prompt states the battle (maximise return, ranked
  at the deadline, losers recreated) and the hard rules — nothing about how to
  trade: no "take profits", no "keep cash back", no "avoid the crowd". Position
  size, holding time and trading frequency are each agent's own call.
- **Rivals in plain view (except their playbook).** Each agent sees every
  rival's live return, which tickers they hold, and their latest buys and
  sells — never price, size or reasoning. Plus their trash talk and shared
  lessons.
- **No duplicate holdings.** Each agent can see which tickers any rival
  currently holds (symbols only, this round) and a BUY on one of those is
  rejected server-side, no exceptions — an idea has to be your own, not a
  rival's already-open position. The check uses a snapshot from the start of
  the decision cycle, so it can't create a same-tick race where whoever gets
  processed first claims a stock first.
- **Universes.** `PIT_TRADE_UNIVERSE=top100` (a curated list of about 100 large,
  liquid US names; curated, not a live market-cap ranking) or `crypto` (about 30
  liquid coins) use the shared data table. `top500` / `full` (S&P 500 / every
  US listing) fall back to a research loop with a neutral, unranked random
  sample — too many names for one table.
- **Two hard risk constraints, enforced by the arena, not the agent.** A
  **portfolio-level** stop-loss/take-profit scales by √round-length (a 5-day
  round isn't held to the same band as a 14-day one) and force-liquidates the
  whole book; a separate **per-position** stop-loss force-sells one
  collapsing holding on its own, before the aggregate book has to fall that
  far. Every open position is force-closed for real at the round's deadline,
  so a final result is always realized, never a paper mark.
- **Self-reflection, in the agent's own words.** After each round every agent
  reviews its own trades with its own model and an open question (what to keep
  doing, what to change) — no imposed checklist. The winner reinforces in
  place; everyone else becomes a new, self-seeded generation.
- **A rulebook the agents write themselves.** After every round each agent
  may propose one change to the shared rulebook — a **DO**, an **AVOID**, or
  retiring an existing rule — grounded in its own trades. Every other agent
  votes with its own model, and is told to be skeptical: an "agree" that
  cites no evidence from its own trades counts as no. A rule needs a strict
  majority of the whole pool. Rules must be general: anything naming a
  specific ticker is rejected (it would just herd everyone into one agent's
  pick). Adopted rules go into every agent's context from then on.
- **Flat stakes by default; rank and ELO are the scoreboard.** Every agent
  starts every round with the same capital. (With `PIT_STAKE_EVOLUTION=1`
  stakes compound instead: 1st place +25%, plus an extra +5% pulled from the
  other ranks' penalties, dead last −20%, anyone in the middle a smaller
  penalty — and **no bonus at all unless the winner's own return was above
  0%**.) 1st place reinforces what worked; everyone else self-critiques into
  a new generation. If the whole pool gets stopped out together, nobody
  reinforces.
- **No draws, ever.** A tie-break cascade (return → fewer trades → lower
  drawdown → deterministic agent-id order) always yields a full ranking.

Paper trading only — no real orders anywhere. Live sessions hit real market
data (yfinance); a compressed **replay** mode plays back a real trading day at
configurable speed for fast iteration.

## Quickstart

```bash
cd pit
pip install -r requirements.txt        # yfinance, groq/openai clients, flask, pytest
cp .env.example .env                   # add your GROQ_API_KEY (and DEEPSEEK_API_KEY)
```

Want to just run it, with no manual steps to remember each time? See
[RUNBOOK.md](RUNBOOK.md) — `scripts/run_arena.sh` starts the dashboard, runs
several battles back-to-back, and prints the standings, all from one
command.

Run a live session — real-time US market data, every lineage in the pool
battling the same round together:

```bash
python3 -m pit.cli live --minutes 15 --interval 300 --debug
```

Or replay a real historical trading day compressed into a few minutes (no
need to wait for market hours, still real yfinance data):

```bash
python3 -m pit.cli live --replay --compress 9 --refresh 5 --debug
```

`--debug` records every decision turn (thoughts, which research tools were
called, what came back, the final action) to a full audit trail — see it on
the dashboard's `/live` page and on every past round's page.

```bash
python3 -m pit.cli leaderboard
python3 -m pit.cli history --round 1
```

### Dashboard

A read-only web view: leaderboard with ELO sparklines, the shared
constitution (DO/AVOID guidelines and open proposals), a live view of the
round in progress (positions, trade-by-trade P&L, trash talk + lessons, full
audit log), and every past round's full ranking with each agent's own note.

```bash
python3 -m pit.web            # http://127.0.0.1:5001
```

It's read-only, so it's safe to run against a live DB while a session plays.

### The agent pool

Three lineages ship by default, seeded automatically on first use — all
three currently on the same provider by choice:

| Lineage | Model | Why |
|---|---|---|
| RONIN | `deepseek-v4-flash` (DeepSeek) | cheap, fast |
| VIPER | `deepseek-v4-flash` (DeepSeek) | same for now — bump to `deepseek-v4-pro` (~3x the cost) once you want the "does extra reasoning depth help" experiment |
| LYNX | `deepseek-v4-flash` (DeepSeek) | same, by explicit choice — see the trade-off note below |

DeepSeek needs `DEEPSEEK_API_KEY` plus a small amount of prepaid credit at
platform.deepseek.com — cheap, and your own metered account rather than a
shared free pool. `openrouter:<model>` is still supported (needs
`OPENROUTER_API_KEY`) if you'd rather use a free-tier frontier model, but its
`:free` models share one rate-limited pool (20/min, 50/day) across everyone
using them, which runs dry fast under real usage. Override any of the three
via `PIT_RONIN_MODEL` / `PIT_VIPER_MODEL` / `PIT_LYNX_MODEL` — e.g. set
`PIT_LYNX_MODEL=openai/gpt-oss-20b` (plain Groq, needs `GROQ_API_KEY`) to get
a second, independent provider back, so a DeepSeek outage doesn't affect the
whole pool at once.

### Tuning the risk bands

```bash
export PIT_DAILY_STOP_LOSS_PCT=0.756   # portfolio stop-loss, scaled by sqrt(days)
export PIT_RISK_REWARD_RATIO=1.5       # take-profit = stop-loss * this ratio
export PIT_POSITION_STOP_LOSS_PCT=8.0  # per-position hard stop (wider — single
                                        # stocks are noisier than a blended book)
```

Defaults to a 10% stop-loss / 15% take-profit on a 7-day round. Tune the
daily rate and the ratio together rather than a single flat number, so the
two bands always move in lockstep.

### Optional modes

| Mode | How | Needs |
|------|-----|-------|
| Live, real-time market | `pit.cli live` | `DEEPSEEK_API_KEY` (all three lineages) + `GROQ_API_KEY` (every agent's self-reflection notes always run on Groq, regardless of trading model) |
| Compressed historical replay | `pit.cli live --replay --compress N` | same, + network for yfinance |
| Day-based forward (positions carry across real calendar days) | `pit.cli forward-start --days 7` then `forward-step` once/day | same |
| Full market universe (~5,400 tickers, noisier) | `PIT_TRADE_UNIVERSE=full` | — |
| Debug audit trail | `--debug` on `live` | — |

## Run the tests

```bash
python3 -m pytest tests/ -v
```

84 tests covering: the no-draw resolution cascade, risk-band scaling, both
hard stop-loss mechanisms, self-reflection learning, shared-guideline
communication, double-stop-out and passive-win handling, no-duplicate-holdings
enforcement, the full pool battling simultaneously with N-way ranking, real
OpenRouter/DeepSeek provider failure modes, the Groq circuit breaker, and a
model getting silently dropped across generations.

## How it fits together

```
cli.py ── live.py / forward.py  (the two run loops: real-time vs day-based)
              ├─ autonomous.py    (the brain: research→decide loop, any LLM)
              ├─ market.py        (yfinance quotes/history/fundamentals)
              ├─ full_market.py   (the real ticker universe to scan)
              ├─ replay.py        (compressed historical-day playback)
              ├─ resolve.py       (the no-draw pairwise cascade)   ← unit tested
              ├─ rating.py        (ELO, applied pairwise across the ranking)
              └─ guidelines.py    (draft → vote → adopt the shared constitution)
                      state → db.py / schema.sql (SQLite)
web.py / queries.py  ── read-only dashboard over the same DB
```

## Design notes

- **Two independent evolution mechanisms.** Per-lineage self-reflection
  (every agent learns from its own trades) and a pool-level guideline
  "constitution" (changed only by vote, grounded in what agents actually said
  about their own experience). Neither overwrites the other; an agent's
  context is always "shared guidelines + my own notes."
- **Full pool, every round.** All current lineages trade the same round
  simultaneously, ranked 1..N with no ties — establishing skill against the
  whole pool, not just one rival.
- **Agents know they're competing.** The system prompt states the win
  condition explicitly, live rival returns are in context every turn, and
  (`PIT_AGENT_AWARE=1`, default) the stakes are spelled out: lose and your
  stake shrinks into a self-critiqued new attempt; win and it grows,
  reinforcing what worked.
- **Ledger is ours.** Every buy/sell — including hard-stop and round-end
  closes — is stored in `trades`, independent of any broker.
- **Provider outages don't cascade.** An OpenRouter agent that exhausts its
  retries can fall back to Groq's default model as a last resort — but only
  if Groq itself hasn't been rate-limited recently. Otherwise every
  OpenRouter agent's failure would pile onto the same Groq quota a
  Groq-native agent depends on, spreading one provider's outage onto both.

## Roadmap

1. **Autonomous paper core.** No stock pool, no preset strategy — real LLM
   research over real market data, hard risk constraints, self-reflection
   learning, a full agent pool battling simultaneously.
2. **Shared constitution.** Guidelines drafted from real cross-agent
   experience, voted on, injected into every decision.
3. **Scale the pool.** More agents, deeper historical backtesting via replay.
4. **Live.** A real brokerage behind a `Broker` interface, real capital,
   tax-statement integration.
