# Running PIT locally, on your own

A step-by-step guide to running the arena end to end without any outside
help — and `scripts/run_arena.sh`, a single script that does all of it for
you every time.

## Prerequisites (one-time)

```bash
cd pit
pip install -r requirements.txt
cp .env.example .env       # then fill in the keys below
```

Edit `.env` and set:

| Key | Needed for | Get it from |
|---|---|---|
| `GROQ_API_KEY` | LYNX's decisions, and every agent's self-reflection notes | console.groq.com (free) |
| `DEEPSEEK_API_KEY` | RONIN/VIPER's decisions (default provider) | platform.deepseek.com (paid — cheap, needs a small prepaid balance) |
| `OPENROUTER_API_KEY` | only if you switch `PIT_RONIN_MODEL`/`PIT_VIPER_MODEL` to `openrouter:...` | openrouter.ai (free tier available, but rate-limited — see "Common issues" below) |

You don't strictly need all three — the arena runs with whatever's
configured, and agents missing a working key will just fail their decisions
until it's set (the script warns about this, it doesn't block you).

## The one command you actually need

```bash
scripts/run_arena.sh
```

That's it. It will:
1. Check `.env` and warn about anything missing.
2. Start the dashboard at `http://127.0.0.1:5001` if it isn't already running
   (and leave it running afterward — it's read-only and safe to keep up).
3. Clean up any round left stuck `'live'` by a previous crashed run (only if
   it has zero trades — never touches a real in-progress round).
4. Run 3 replay battles back-to-back (the default), each compressing a real
   trading day into ~9 minutes of wall-clock time.
5. Print the final leaderboard and where to find the full log.

Every run's complete output is saved to `data/run_logs/run_<timestamp>.log`
— that's the file to check if you want to see exactly what each agent
thought and did, turn by turn.

## Common variations

```bash
# Fresh pool, fresh stakes, 5 quick battles
scripts/run_arena.sh --reset --battles 5 --compress 5

# One real-time battle (only works during US market hours, 9:30am–4pm ET)
scripts/run_arena.sh --mode live --minutes 15 --battles 1

# Tighter risk bands, so rounds resolve on a stop/goal hit instead of
# running the clock out — good for watching the mechanics fire quickly
scripts/run_arena.sh --daily-stop-loss 0.76 --risk-reward-ratio 0.5 --battles 3

# Replay a SPECIFIC real trading day instead of the most recent one
scripts/run_arena.sh --date 2026-08-31
```

Run `scripts/run_arena.sh --help` for the complete flag list.

## Watching it happen

While battles run (or after they finish — the DB doesn't go anywhere):

- **`http://127.0.0.1:5001/`** — leaderboard, standings, the shared
  constitution (DO/AVOID guidelines), recent round results.
- **`http://127.0.0.1:5001/live`** — the round currently in progress (or the
  most recent one): live positions, trade-by-trade P&L, trash talk, and the
  full debug audit log (every agent's thoughts and tool calls).
- **`http://127.0.0.1:5001/round/<id>`** — any past round's full ranked
  outcome, trades, and audit trail.

Both pages auto-refresh while a session is live; nothing to click.

## Checking standings from the terminal

```bash
python3 -m pit.cli leaderboard
python3 -m pit.cli history --round 3     # a specific round's trades
```

## Stopping things

```bash
pkill -f "pit.web"        # stop the dashboard
pkill -f "pit.cli live"   # stop a battle that's still running
```
The script doesn't auto-kill the dashboard on exit — it's meant to stay up
so you can keep browsing results after a run finishes.

## Common issues (all self-explanatory in the log, but for reference)

- **`RateLimitError` on one or two turns** — normal, handled by retry/backoff
  automatically. Only worth investigating if it happens on *every* agent,
  *every* tick, for a whole battle — that means a provider's quota (usually
  OpenRouter's free tier: 20/min, 50/day) is genuinely exhausted, not just a
  momentary hiccup. Waiting a while (quotas reset daily) or switching that
  agent's model fixes it — see the README's provider table.
- **`RuntimeError: no intraday data available for <date>`** — the replay day
  you asked for doesn't have usable 5-minute bar data (e.g. a market
  holiday, or a date too recent/old for the data source's window). Try
  a different `--date`, or omit it to use the most recent trading day.
- **A round stuck at `'live'` blocking the next run** — the script already
  auto-cleans this (see step 3 above) as long as it has zero trades. If it
  has real trades and you still want to discard it, delete `data/pit.db`
  and start fresh with `--reset`.
- **Cost** — DeepSeek's `deepseek-v4-flash` is very cheap; a typical battle
  (a handful of decisions × 2 agents) costs a small fraction of a cent to a
  few cents. Check your balance occasionally at platform.deepseek.com if
  you're running many battles in a row.
