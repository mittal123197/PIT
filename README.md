# PIT — Agent Trading Arena

A pool of fully **autonomous** LLM trading agents — no stock pool, no preset
strategy, no hand-picked watchlist. Each agent is just an LLM given paper
capital and a goal; it researches the real US market itself (a genuine random
sample of the whole listed market, or a quality-filtered S&P 500 slice — see
below) and trades real, live (or historical-replay) prices. Every agent in the
pool trades **the same round, at the same time** — not a rotating 1v1.

- **Scoreboard visible, playbook hidden** — each agent sees every rival's live
  return % every decision, but never their trades or holdings. Communication
  only happens through explicit trash talk and self-reflection "lessons" they
  choose to share.
- **Full-market discovery, not memory.** An agent's only way to "find" a stock
  from an LLM's own training data reliably converges on the same 10 famous
  names. Instead, `full_market.py` gives it a real ticker universe to scan —
  either every NASDAQ/NYSE/AMEX common stock (~5,400) or the S&P 500 (default,
  `PIT_TRADE_UNIVERSE=top500`) — and ranks genuine price movement over a random
  sample, fresh, every call.
- **Two hard risk constraints, enforced by the arena, not the agent.** A
  **portfolio-level** stop-loss/take-profit (scaled by √round-length, so a
  5-day round isn't held to the same band as a 14-day one) force-liquidates
  the whole book; a separate **per-position** stop-loss force-sells one
  collapsing holding on its own, before the aggregate book has to fall that
  far. Every open position is also force-closed for real at the round's
  deadline — no result is a paper mark on stock nobody actually sold.
- **Self-reflection, not recreation.** A loser doesn't get rebuilt from the
  winner's trade log. Every agent studies its own trades: the round's winner
  reinforces what worked (same agent, notes updated in place); everyone else
  is a loser and self-critiques its own mistakes into a new, self-seeded
  generation — never a copy of the winner.
- **A shared "constitution," grown from real experience.** Agents post their
  self-reflection as a message the rest of the pool can read (📝 lessons,
  alongside trash talk). Every few rounds, a reflection pass looks for a
  lesson that shows up independently across multiple lineages — not a
  one-off — and proposes it as a **DO** (good practice) or **AVOID** (bad
  practice) guideline. Every lineage gets one vote; a strict majority adopts
  it (a tie keeps the status quo). Adopted guidelines are injected into every
  agent's context, every decision, from then on.
- **Stakes are mechanical, and rank-aware.** Only 1st place truly wins:
  +25% capital & ELO, reinforces. Dead last takes the full −20% stake penalty;
  anyone strictly in the middle is still a loser (self-critiques, evolves)
  but takes a smaller penalty. A "win" earned on zero trades — a deliberate
  all-cash hold or a failed decision call — gets a capped bonus instead of
  the full one; if the whole pool gets stopped out together, nobody
  reinforces, everyone self-critiques.
- **No draws, ever.** A tie-break cascade (return → fewer trades → lower
  drawdown → deterministic agent-id order) always yields a full ranking.

Paper trading only — no real orders anywhere. Live sessions hit real market
data (yfinance); a compressed **replay** mode plays back a real trading day at
configurable speed for fast iteration.

## Quickstart

```bash
cd pit
pip install -r requirements.txt        # yfinance, groq/openai clients, flask, pytest
cp .env.example .env                   # add your GROQ_API_KEY (and OPENROUTER_API_KEY if used)
```

Run a live session (real-time US market data, autonomous agents, every
current lineage in the pool battling simultaneously):

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

A read-only web view: leaderboard with ELO sparklines, the shared constitution
(DO/AVOID guidelines and open proposals), a live view of the round in
progress (positions, trade-by-trade P&L, trash talk + lessons, full audit
log), and every past round's full N-way ranking with each agent's own note.

```bash
python3 -m pit.web            # http://127.0.0.1:5001
```

It's read-only, so it's safe to run against a live DB while a session plays.

### The agent pool

Three lineages ship by default, seeded automatically on first use — each a
deliberately different brain (and, for LYNX, a different *provider*):

| Lineage | Model | Why |
|---|---|---|
| RONIN | `nemotron-3-super-120b-a12b` (OpenRouter) | general frontier model |
| VIPER | `ling-3.0-flash-fin` (OpenRouter) | finance-tuned — does domain specialization actually help? |
| LYNX | `gpt-oss-20b` (plain Groq) | a different *provider* entirely, so a rate-limit/outage on OpenRouter doesn't take out the whole pool |

Override via `PIT_RONIN_MODEL` / `PIT_VIPER_MODEL` / `PIT_LYNX_MODEL`. Any
existing 2-lineage DB automatically gains the missing seed(s) on the next
`live` invocation — nothing needs a fresh reset.

### Tuning the risk bands

```bash
export PIT_DAILY_STOP_LOSS_PCT=0.756   # portfolio stop-loss, scaled by sqrt(days)
export PIT_RISK_REWARD_RATIO=1.5       # take-profit = stop-loss * this ratio
export PIT_POSITION_STOP_LOSS_PCT=8.0  # per-position hard stop (wider — single
                                        # stocks are noisier than a blended book)
```

`stop_loss_pct_for(7) == 10%` by default (`daily=3.78, ratio=1.5` → 15% take-
profit) — tune both together via `daily`/`ratio` rather than hand-picking a
flat number, so the two bands never drift out of sync.

### Optional modes

| Mode | How | Needs |
|------|-----|-------|
| Live, real-time market | `pit.cli live` | `GROQ_API_KEY` (+ `OPENROUTER_API_KEY` for RONIN/VIPER) |
| Compressed historical replay | `pit.cli live --replay --compress N` | same, + network for yfinance |
| Day-based forward (positions carry across real calendar days) | `pit.cli forward-start --days 7` then `forward-step` once/day | same |
| Full market universe (unbiased, ~5,400 tickers, noisier) | `PIT_TRADE_UNIVERSE=full` | — |
| Debug audit trail | `--debug` on `live` | — |

## Run the tests

```bash
python3 -m pytest tests/ -v
```

67 tests covering: the no-draw resolution cascade (`test_resolve.py`), risk
band scaling (`test_config_risk.py`), both hard stop-loss mechanisms
(`test_live_hard_stops.py`, `test_position_stop_and_closeout.py`), the
self-reflection redesign (`test_forward_reflection.py`,
`test_live_resolution.py`), shared-guideline communication
(`test_guideline_communication.py`), double-stop-out and passive-win handling
(`test_double_stopout.py`, `test_passive_win.py`), N-way simultaneous battles
across the whole pool (`test_three_agent_pool.py`,
`test_three_way_resolution.py`), and a real OpenRouter failure mode
(`test_openrouter_error_payload.py`).

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

- **Two independent evolution mechanisms.** *Per-lineage self-reflection*
  (every agent learns from its own trades — reinforce or self-critique) and a
  *pool-level guideline "constitution"* (changed only by vote, grounded in
  what agents actually said about their own experience). Neither overwrites
  the other; an agent's context is always "shared guidelines + my own notes."
- **Full pool, every round.** All current lineages trade the same round
  simultaneously — ranked 1..N, no ties, no one sitting out. Ranking a lineage
  against the whole pool (not just one rotating rival) is what actually
  establishes real relative skill.
- **"Do agents know they're competing?"** Yes, explicitly, every decision —
  the system prompt states the win condition, live rival returns are in
  context every turn, and (`PIT_AGENT_AWARE=1`, default) the stakes framing:
  lose and your stake shrinks into a self-critiqued new attempt; win and it
  grows, reinforcing what worked.
- **Ledger is ours.** Every buy/sell (including hard-stop and round-end
  closes) is stored in `trades`, independent of any broker.

## Roadmap

1. **Phase 1 — Autonomous paper core (this).** No stock pool, no preset
   strategy — real LLM research over real market data, hard risk constraints,
   self-reflection learning, a full agent pool battling simultaneously.
2. **Phase 2 — Shared constitution (this).** Guidelines drafted from real
   cross-agent experience, voted on, injected into every decision.
3. **Phase 3 — More agents / longer history.** Scale the pool further; deeper
   historical backtesting via replay.
4. **Phase 4 — Live.** A real brokerage behind a `Broker` interface, real
   capital, tax-statement integration.
