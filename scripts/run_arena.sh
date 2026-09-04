#!/usr/bin/env bash
# Run N PIT battles + the dashboard, with no manual babysitting.
#
#   scripts/run_arena.sh [options]
#
# Run with --help for the full flag list. See RUNBOOK.md for what to expect
# while it's running and how to read the results afterward.
set -uo pipefail
# (deliberately NOT `set -e` — one battle crashing must not abort the rest
# of the loop; each battle's exit code is checked explicitly below instead.)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$SCRIPT_DIR"

# ---- defaults ----
MODE="replay"          # replay | live
BATTLES=3
COMPRESS=9             # replay: compress one real trading day into this many minutes
REPLAY_DATE=""         # replay: which real day (default: pit.cli's own "last trading day")
MINUTES=15             # live: session length in minutes
INTERVAL=300           # live: seconds between agent decisions
REFRESH=5              # seconds between cheap mark-to-market refreshes
DEBUG=1                # 1 = record the full audit trail (thoughts/tools/actions)
RESET=0                # 1 = wipe the local DB before running (fresh pool, fresh stakes)
PORT="${PIT_WEB_PORT:-5001}"
DAILY_STOP_LOSS=""     # optional: override PIT_DAILY_STOP_LOSS_PCT
RISK_REWARD_RATIO=""   # optional: override PIT_RISK_REWARD_RATIO

usage() {
  cat <<'EOF'
Usage: scripts/run_arena.sh [options]

  --mode live|replay          Trading mode (default: replay)
  --battles N                 How many battles to run back-to-back (default: 3)
  --compress N                 Replay: compress one real day into N minutes (default: 9)
  --date YYYY-MM-DD           Replay: which real trading day to replay
                               (default: pit.cli's own "most recent weekday")
  --minutes N                 Live: session length in minutes (default: 15)
  --interval N                 Live: seconds between agent decisions (default: 300)
  --refresh N                 Seconds between cheap price refreshes (default: 5)
  --no-debug                  Disable the full audit trail (default: on)
  --reset                     Wipe the local DB first (fresh pool, fresh stakes)
  --port N                     Dashboard port (default: 5001)
  --daily-stop-loss N          Override PIT_DAILY_STOP_LOSS_PCT for this run
  --risk-reward-ratio N         Override PIT_RISK_REWARD_RATIO for this run
  -h, --help                   Show this help

Examples:
  scripts/run_arena.sh                                        # 3 replay battles, defaults
  scripts/run_arena.sh --battles 5 --compress 5               # 5 quick replay battles
  scripts/run_arena.sh --mode live --minutes 15 --battles 1   # one real-time battle
  scripts/run_arena.sh --reset --battles 2                    # fresh pool, 2 battles
  scripts/run_arena.sh --daily-stop-loss 0.76 --risk-reward-ratio 0.5   # ~2%/1% bands
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode) MODE="$2"; shift 2 ;;
    --battles) BATTLES="$2"; shift 2 ;;
    --compress) COMPRESS="$2"; shift 2 ;;
    --date) REPLAY_DATE="$2"; shift 2 ;;
    --minutes) MINUTES="$2"; shift 2 ;;
    --interval) INTERVAL="$2"; shift 2 ;;
    --refresh) REFRESH="$2"; shift 2 ;;
    --no-debug) DEBUG=0; shift ;;
    --reset) RESET=1; shift ;;
    --port) PORT="$2"; shift 2 ;;
    --daily-stop-loss) DAILY_STOP_LOSS="$2"; shift 2 ;;
    --risk-reward-ratio) RISK_REWARD_RATIO="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage; exit 1 ;;
  esac
done

PYTHON="${PIT_PYTHON:-python3}"
LOG_DIR="$SCRIPT_DIR/data/run_logs"
mkdir -p "$LOG_DIR"
RUN_LOG="$LOG_DIR/run_$(date +%Y%m%d_%H%M%S).log"
PID_FILE="$SCRIPT_DIR/data/.dashboard.pid"

log() { echo "$@" | tee -a "$RUN_LOG"; }

# ---- refuse to run two battle-runners at once against the same DB ----
# Two `pit.cli live` processes deciding trades for the same active round
# simultaneously isn't just a SQLite-contention risk (both writing the same
# tables) — it's a logic error: the round model assumes one writer. Caught
# for real running two of these scripts at once, which surfaced as an
# `OperationalError('unable to open database file')` mid-battle in one of
# them once both were hammering data/pit.db concurrently.
LOCK_FILE="$SCRIPT_DIR/data/.run_arena.lock"
mkdir -p "$SCRIPT_DIR/data"
if [[ -f "$LOCK_FILE" ]]; then
  other_pid="$(cat "$LOCK_FILE" 2>/dev/null || true)"
  if [[ -n "$other_pid" ]] && kill -0 "$other_pid" 2>/dev/null; then
    echo "⚠️  Another run_arena.sh is already running (pid $other_pid)." >&2
    echo "   Running two at once against the same data/pit.db corrupts the active round." >&2
    echo "   Wait for it to finish, or if it's actually dead, remove $LOCK_FILE and retry." >&2
    exit 1
  fi
  # stale lock from a killed/crashed run — safe to take over
fi
echo $$ > "$LOCK_FILE"
trap 'rm -f "$LOCK_FILE"' EXIT

log "PIT arena runner — mode=$MODE battles=$BATTLES $(date '+%Y-%m-%d %H:%M:%S')"
log "Full log: $RUN_LOG"

# ---- .env sanity check (warn, don't hard-fail — Groq-only still runs LYNX) ----
if [[ ! -f .env ]]; then
  log "⚠️  No .env found — copy .env.example to .env and add your API keys first."
  exit 1
fi
set -a
# shellcheck disable=SC1091
source .env
set +a

missing=()
[[ -z "${GROQ_API_KEY:-}" ]] && missing+=("GROQ_API_KEY (needed by LYNX, and by every agent's self-reflection notes)")
[[ -z "${DEEPSEEK_API_KEY:-}" ]] && missing+=("DEEPSEEK_API_KEY (needed by RONIN/VIPER by default)")
if [[ ${#missing[@]} -gt 0 ]]; then
  log "⚠️  Missing from .env:"
  for m in "${missing[@]}"; do log "   - $m"; done
  log "   Continuing anyway — those agents will fail their decisions until it's set."
fi

# ---- optional risk-band overrides for this run only ----
[[ -n "$DAILY_STOP_LOSS" ]] && export PIT_DAILY_STOP_LOSS_PCT="$DAILY_STOP_LOSS"
[[ -n "$RISK_REWARD_RATIO" ]] && export PIT_RISK_REWARD_RATIO="$RISK_REWARD_RATIO"

# ---- optional reset ----
if [[ "$RESET" == "1" ]]; then
  log "Resetting local state (data/pit.db)..."
  rm -f data/pit.db data/pit.db-shm data/pit.db-wal
fi

# ---- clear a round left stuck 'live' by a previous crashed/killed run ----
# (only removed if it has ZERO trades — a real in-progress round with actual
# trades is never touched)
"$PYTHON" - <<'PYEOF' 2>>"$RUN_LOG"
import sqlite3, os
db = os.getenv("PIT_DB_PATH") or "data/pit.db"
if not os.path.exists(db):
    raise SystemExit(0)
conn = sqlite3.connect(db)
conn.row_factory = sqlite3.Row
row = conn.execute("SELECT id FROM rounds WHERE status='live'").fetchone()
if row:
    rid = row["id"]
    trades = conn.execute("SELECT COUNT(*) c FROM trades WHERE round_id=?", (rid,)).fetchone()["c"]
    if trades == 0:
        for t in ("round_states", "agent_messages", "agent_audit"):
            conn.execute(f"DELETE FROM {t} WHERE round_id=?", (rid,))
        conn.execute("DELETE FROM rounds WHERE id=?", (rid,))
        conn.commit()
        print(f"  (cleaned up an empty stuck round #{rid} from a previous interrupted run)")
PYEOF

# ---- start (or reuse) the dashboard — left running after this script exits ----
if curl -sS -o /dev/null -w "%{http_code}" "http://127.0.0.1:$PORT/" 2>/dev/null | grep -q "200"; then
  log "Dashboard already running at http://127.0.0.1:$PORT"
else
  log "Starting dashboard at http://127.0.0.1:$PORT ..."
  PIT_WEB_PORT="$PORT" nohup "$PYTHON" -m pit.web >> "$LOG_DIR/web.log" 2>&1 &
  echo $! > "$PID_FILE"
  sleep 2
fi

# ---- run the battles ----
FAILURES=0
for i in $(seq 1 "$BATTLES"); do
  log ""
  log "=== BATTLE $i/$BATTLES starting $(date '+%H:%M:%S') ==="

  ARGS=(live)
  if [[ "$MODE" == "replay" ]]; then
    ARGS+=(--replay --compress "$COMPRESS" --refresh "$REFRESH")
    [[ -n "$REPLAY_DATE" ]] && ARGS+=(--date "$REPLAY_DATE")
  else
    ARGS+=(--minutes "$MINUTES" --interval "$INTERVAL")
  fi
  [[ "$DEBUG" == "1" ]] && ARGS+=(--debug)

  if "$PYTHON" -m pit.cli "${ARGS[@]}" >>"$RUN_LOG" 2>&1; then
    log "=== BATTLE $i/$BATTLES done $(date '+%H:%M:%S') ==="
  else
    code=$?
    FAILURES=$((FAILURES + 1))
    log "=== BATTLE $i/$BATTLES FAILED (exit $code) — see $RUN_LOG for the traceback ==="
    # clean up whatever it left stuck before the next attempt
    "$PYTHON" - <<'PYEOF' 2>>"$RUN_LOG"
import sqlite3, os
db = os.getenv("PIT_DB_PATH") or "data/pit.db"
conn = sqlite3.connect(db)
conn.row_factory = sqlite3.Row
row = conn.execute("SELECT id FROM rounds WHERE status='live'").fetchone()
if row:
    rid = row["id"]
    trades = conn.execute("SELECT COUNT(*) c FROM trades WHERE round_id=?", (rid,)).fetchone()["c"]
    if trades == 0:
        for t in ("round_states", "agent_messages", "agent_audit"):
            conn.execute(f"DELETE FROM {t} WHERE round_id=?", (rid,))
        conn.execute("DELETE FROM rounds WHERE id=?", (rid,))
        conn.commit()
PYEOF
  fi
done

log ""
log "=== FINAL STANDINGS ==="
"$PYTHON" -m pit.cli leaderboard 2>&1 | tee -a "$RUN_LOG"

log ""
if [[ "$FAILURES" -gt 0 ]]; then
  log "$FAILURES of $BATTLES battle(s) failed — see $RUN_LOG for details."
fi
log "Dashboard: http://127.0.0.1:$PORT/live"
log "Full log:  $RUN_LOG"
