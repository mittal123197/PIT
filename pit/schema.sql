-- PIT arena schema (SQLite).
--
-- Two independent evolution mechanisms are modelled here:
--   1. Per-lineage mutation: a loser's next `agents` row is seeded from the
--      winner's `trades` (seeded_from_trade_agent_id).
--   2. Pool-level guidelines: a shared "constitution" injected into every
--      agent, changed only by vote (guidelines / guideline_proposals /
--      guideline_votes). Tables exist from Phase 1; the vote mechanism is
--      wired in Phase 2.

PRAGMA foreign_keys = ON;

-- A persistent "agent slot". Survives across generations; a mutation swaps the
-- current generation but keeps the lineage (and its vote history, rating, and
-- cumulative return).
CREATE TABLE IF NOT EXISTS lineages (
    id                     INTEGER PRIMARY KEY AUTOINCREMENT,
    name                   TEXT NOT NULL UNIQUE,
    cumulative_return_pct  REAL NOT NULL DEFAULT 0.0,
    current_stake          REAL NOT NULL DEFAULT 100000.0,  -- carried between rounds
    wins                   INTEGER NOT NULL DEFAULT 0,
    losses                 INTEGER NOT NULL DEFAULT 0,
    current_agent_id       INTEGER,          -- FK to agents.id (nullable at seed time)
    created_at             TEXT NOT NULL
);

-- One concrete generation of a lineage: a strategy config + activity profile +
-- rating. A new row is created every time a lineage loses and mutates.
CREATE TABLE IF NOT EXISTS agents (
    id                      INTEGER PRIMARY KEY AUTOINCREMENT,
    lineage_id              INTEGER NOT NULL REFERENCES lineages(id),
    generation              INTEGER NOT NULL DEFAULT 1,
    strategy_config         TEXT NOT NULL,    -- JSON: the mutable "brain"
    activity_profile        TEXT NOT NULL,    -- JSON: heartbeat_minutes, watch_rules
    rating                  REAL NOT NULL,
    -- The winning agent this generation mutated from (NULL for gen 1).
    seeded_from_trade_agent_id INTEGER REFERENCES agents(id),
    mutation_note           TEXT,
    created_at              TEXT NOT NULL
);

-- A single duel between two agent-generations.
CREATE TABLE IF NOT EXISTS rounds (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    round_number   INTEGER NOT NULL,
    agent_a_id     INTEGER NOT NULL REFERENCES agents(id),
    agent_b_id     INTEGER NOT NULL REFERENCES agents(id),
    start_at       TEXT NOT NULL,
    deadline       TEXT NOT NULL,
    length_days    INTEGER NOT NULL,
    goal_pct       REAL NOT NULL,
    status         TEXT NOT NULL DEFAULT 'running',  -- running | resolved
    created_at     TEXT NOT NULL
);

-- Per-agent-per-round mutable state (capital, holdings, risk tracking).
CREATE TABLE IF NOT EXISTS round_states (
    id                 INTEGER PRIMARY KEY AUTOINCREMENT,
    round_id           INTEGER NOT NULL REFERENCES rounds(id),
    agent_id           INTEGER NOT NULL REFERENCES agents(id),
    starting_capital   REAL NOT NULL,
    current_capital    REAL NOT NULL,     -- cash only; total value = cash + holdings
    holdings           TEXT NOT NULL DEFAULT '{}',  -- JSON {symbol: qty}
    status             TEXT NOT NULL DEFAULT 'active',  -- active | liquidated
    liquidated_at      TEXT,
    max_drawdown_pct   REAL NOT NULL DEFAULT 0.0,
    trade_count        INTEGER NOT NULL DEFAULT 0,
    final_return_pct   REAL,              -- filled at resolution
    UNIQUE (round_id, agent_id)
);

-- PIT's own permanent transaction ledger: every buy/sell with date + stock
-- info, independent of the broker's retention.
CREATE TABLE IF NOT EXISTS trades (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    round_id      INTEGER NOT NULL REFERENCES rounds(id),
    agent_id      INTEGER NOT NULL REFERENCES agents(id),
    ts            TEXT NOT NULL,
    symbol        TEXT NOT NULL,
    side          TEXT NOT NULL,          -- buy | sell
    qty           REAL NOT NULL,
    price         REAL NOT NULL,
    fee           REAL NOT NULL DEFAULT 0.0,
    capital_after REAL NOT NULL,          -- cash after the fill
    reason        TEXT                    -- optional rationale from the agent
);

-- Agent-registered price alerts. In Phase 1 the orchestrator evaluates these
-- each tick; in Phase 4 they migrate to broker-side triggers/webhooks.
CREATE TABLE IF NOT EXISTS price_watches (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    round_id      INTEGER NOT NULL REFERENCES rounds(id),
    agent_id      INTEGER NOT NULL REFERENCES agents(id),
    symbol        TEXT NOT NULL,
    trigger_type  TEXT NOT NULL,          -- pct_move | price_above | price_below
    threshold     REAL NOT NULL,
    reference     REAL,                   -- e.g. price at time the watch was set
    active        INTEGER NOT NULL DEFAULT 1,
    fired_at      TEXT,
    created_at    TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS round_results (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    round_id          INTEGER NOT NULL UNIQUE REFERENCES rounds(id),
    winner_agent_id   INTEGER REFERENCES agents(id),
    loser_agent_id    INTEGER REFERENCES agents(id),
    resolution_reason TEXT NOT NULL,      -- return | trade_count | drawdown | sudden_death
    winner_return_pct REAL,
    loser_return_pct  REAL,
    rating_delta      REAL,
    created_at        TEXT NOT NULL
);

-- ---- Pool-level "constitution" (schema now, vote mechanism in Phase 2) ----

CREATE TABLE IF NOT EXISTS guidelines (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    text        TEXT NOT NULL,
    status      TEXT NOT NULL DEFAULT 'active',  -- active | retired
    version     INTEGER NOT NULL DEFAULT 1,
    created_at  TEXT NOT NULL,
    retired_at  TEXT
);

CREATE TABLE IF NOT EXISTS guideline_proposals (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    kind           TEXT NOT NULL,         -- add | remove
    guideline_id   INTEGER REFERENCES guidelines(id),  -- NULL for a brand-new 'add'
    proposed_text  TEXT,
    source_round_id INTEGER REFERENCES rounds(id),
    resolution     TEXT,                  -- NULL (open) | accepted | rejected
    resolved_at    TEXT,
    created_at     TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS guideline_votes (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    proposal_id  INTEGER NOT NULL REFERENCES guideline_proposals(id),
    lineage_id   INTEGER NOT NULL REFERENCES lineages(id),  -- vote belongs to the slot
    vote         TEXT NOT NULL,           -- agree | disagree
    reasoning    TEXT,
    created_at   TEXT NOT NULL,
    UNIQUE (proposal_id, lineage_id)
);

-- Full audit trail of every LLM turn an agent takes, behind debug mode
-- (PIT_DEBUG=1 / `pit live --debug`). One row per research-or-decide turn, so
-- a single fill can be traced back through every tool call and thought that
-- led to it, not just its final "reason" string.
CREATE TABLE IF NOT EXISTS agent_audit (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    round_id            INTEGER NOT NULL REFERENCES rounds(id),
    agent_id            INTEGER NOT NULL REFERENCES agents(id),
    ts                  TEXT NOT NULL,
    turn                INTEGER NOT NULL,       -- 1, 2, 3... within one decision
    model               TEXT,
    thoughts            TEXT,                   -- the agent's own "thoughts" field
    research_requested  TEXT,                   -- JSON: {"scan":true,"history":[...],...}
    research_results    TEXT,                   -- JSON: what came back
    done                INTEGER NOT NULL DEFAULT 0,  -- 1 = this turn's action was final
    orders              TEXT,                   -- JSON: orders, if done
    message             TEXT,                   -- taunt message, if done
    raw_response        TEXT                    -- the full raw LLM JSON for this turn
);
CREATE INDEX IF NOT EXISTS idx_audit_round ON agent_audit(round_id, id);

-- Agent-to-agent messages (banter / mocking) during a live session. Each agent
-- sees the rival's recent messages in its next decision, so a rivalry develops.
CREATE TABLE IF NOT EXISTS agent_messages (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    round_id   INTEGER NOT NULL REFERENCES rounds(id),
    agent_id   INTEGER NOT NULL REFERENCES agents(id),
    ts         TEXT NOT NULL,
    message    TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_messages_round ON agent_messages(round_id, id);

-- Which real trading days a live (forward) round has already processed, so a
-- daily step is idempotent and the round advances one real day at a time.
CREATE TABLE IF NOT EXISTS live_days (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    round_id     INTEGER NOT NULL REFERENCES rounds(id),
    trade_date   TEXT NOT NULL,
    processed_at TEXT NOT NULL,
    UNIQUE (round_id, trade_date)
);

CREATE INDEX IF NOT EXISTS idx_trades_round_agent ON trades(round_id, agent_id);
CREATE INDEX IF NOT EXISTS idx_watches_round_active ON price_watches(round_id, active);
CREATE INDEX IF NOT EXISTS idx_agents_lineage ON agents(lineage_id);
