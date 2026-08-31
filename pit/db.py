"""SQLite access layer: connection, schema init, and a thin row helper.

Deliberately lightweight — raw SQL over `sqlite3`, matching the "lightweight,
not enterprise" instruction and every sibling project's storage choice. The
whole DB is one file, so the Phase 2 deploy story is just "put this file on a
persistent volume".
"""
from __future__ import annotations

import os
import sqlite3
from pathlib import Path

try:  # dotenv is optional at import time; CLI loads it explicitly too.
    from dotenv import load_dotenv

    load_dotenv()
except Exception:  # pragma: no cover
    pass

PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = PACKAGE_DIR.parent
SCHEMA_PATH = PACKAGE_DIR / "schema.sql"

DEFAULT_DB_PATH = os.getenv("PIT_DB_PATH", str(PROJECT_DIR / "data" / "pit.db"))


def connect(db_path: str | None = None) -> sqlite3.Connection:
    """Open a connection with rows accessible by column name and FKs enforced."""
    path = db_path or DEFAULT_DB_PATH
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    # WAL lets the dashboard (reader) and the arena daemon (writer) share the
    # file concurrently — needed once both run in the deployed container.
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA busy_timeout = 30000")
    return conn


def init_db(conn: sqlite3.Connection) -> None:
    """Create all tables if they don't exist (idempotent)."""
    conn.executescript(SCHEMA_PATH.read_text())
    _migrate_columns(conn)
    conn.commit()


# No formal migration framework — this project is early enough that most
# schema changes have just meant starting a fresh data/*.db. `CREATE TABLE IF
# NOT EXISTS` alone doesn't add a column to an EXISTING table, so a new column
# on a table that already existed needs one of these, best-effort (ignored if
# the column is already there or the table doesn't exist yet).
_COLUMN_ADDITIONS = [
    ("guidelines", "practice", "TEXT NOT NULL DEFAULT 'good'"),
    ("guideline_proposals", "practice", "TEXT NOT NULL DEFAULT 'good'"),
    ("agent_messages", "kind", "TEXT NOT NULL DEFAULT 'banter'"),
    ("round_results", "winner_note", "TEXT"),
    ("round_results", "loser_note", "TEXT"),
    ("round_results", "both_stopped_out", "INTEGER NOT NULL DEFAULT 0"),
    ("round_results", "passive_win", "INTEGER NOT NULL DEFAULT 0"),
    ("round_states", "cost_basis", "TEXT NOT NULL DEFAULT '{}'"),
]


def _migrate_columns(conn: sqlite3.Connection) -> None:
    for table, column, decl in _COLUMN_ADDITIONS:
        try:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {decl}")
        except sqlite3.OperationalError:
            pass  # column already exists (or table doesn't exist yet)


def reset_db(db_path: str | None = None) -> None:
    """Delete the DB file entirely. Destructive — used by `cli init --fresh`."""
    path = Path(db_path or DEFAULT_DB_PATH)
    if path.exists():
        path.unlink()
