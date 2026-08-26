"""Command-line entrypoint for the arena.

    python -m pit.cli init                 # create DB + seed two lineages
    python -m pit.cli run-round            # run one duel, print the result
    python -m pit.cli arena --rounds 5     # run several duels back to back
    python -m pit.cli leaderboard          # standings
    python -m pit.cli history [--round N]  # trades + results

By default everything runs offline: deterministic policy + synthetic feed. Add
--feed historical for yfinance NSE replay, or set PIT_USE_LLM=1 (with a
GROQ_API_KEY) to let Groq drive the agents and the mutation step.
"""
from __future__ import annotations

import argparse
import os

from . import db as dbm
from .config import DEFAULT
from .engine import Engine
from .feeds import HistoricalFeed, SyntheticFeed

SEED_LINEAGES = [
    ("CONTANGO", {"type": "momentum", "lookback": 5, "buy_threshold_pct": 1.0,
                  "sell_threshold_pct": -1.5, "position_frac": 0.25,
                  "heartbeat_minutes": 60}),
    ("BACKWARDATION", {"type": "mean_reversion", "lookback": 10, "band_pct": 2.0,
                       "position_frac": 0.25, "heartbeat_minutes": 120}),
]


def _policy_factory(use_llm: bool):
    if use_llm:
        from .llm import LLMPolicy, groq_available
        if groq_available():
            return lambda agent_row: LLMPolicy()
        print("  (PIT_USE_LLM set but Groq unavailable — using deterministic policy)")
    from .policies import SimplePolicy
    return lambda agent_row: SimplePolicy()


def _engine(conn, use_llm: bool) -> Engine:
    from .llm import mutate_with_llm
    return Engine(conn, config=DEFAULT, policy_factory=_policy_factory(use_llm),
                  mutate_fn=mutate_with_llm, use_llm=use_llm)


def _feed_factory(kind: str, round_number: int, bars_override: int | None = None):
    """Return a callable `(length_days) -> PriceFeed`, sized to the round.

    `--bars` overrides the day-based sizing (handy to keep LLM runs short).
    """
    def make(length_days: int):
        bars = bars_override or DEFAULT.bars_for_days(length_days)
        if kind == "historical":
            try:
                return HistoricalFeed(DEFAULT.universe, bars=bars)
            except Exception as exc:
                print(f"  historical feed unavailable ({exc}); using synthetic")
        return SyntheticFeed(DEFAULT.universe, seed=1000 + round_number,
                             bars=bars, bar_minutes=DEFAULT.bar_minutes)
    return make


def _lineage_ids(conn) -> list[int]:
    return [r["id"] for r in conn.execute(
        "SELECT id FROM lineages ORDER BY id").fetchall()]


# ---- commands ---------------------------------------------------------

def cmd_init(args):
    if args.fresh:
        dbm.reset_db()
        print("Wiped existing DB.")
    conn = dbm.connect()
    dbm.init_db(conn)
    existing = conn.execute("SELECT COUNT(*) c FROM lineages").fetchone()["c"]
    if existing:
        print(f"DB ready — {existing} lineages already present.")
        return
    eng = _engine(conn, use_llm=False)
    for name, cfg in SEED_LINEAGES:
        eng.create_lineage(name, cfg)
        print(f"  seeded lineage: {name} ({cfg['type']})")
    print(f"Initialised at {dbm.DEFAULT_DB_PATH}")


def cmd_run_round(args):
    conn = dbm.connect()
    dbm.init_db(conn)
    use_llm = os.getenv("PIT_USE_LLM", "0") not in ("0", "", "false")
    eng = _engine(conn, use_llm)
    ids = _lineage_ids(conn)
    if len(ids) < 2:
        print("Need at least 2 lineages. Run `init` first.")
        return
    factory = _feed_factory(args.feed, eng._next_round_number(), args.bars)
    out = eng.run_round(ids[0], ids[1], feed_factory=factory)
    _print_outcome(out)
    _print_reflection(eng.last_reflection)


def cmd_arena(args):
    conn = dbm.connect()
    dbm.init_db(conn)
    use_llm = os.getenv("PIT_USE_LLM", "0") not in ("0", "", "false")
    eng = _engine(conn, use_llm)
    ids = _lineage_ids(conn)
    if len(ids) < 2:
        print("Need at least 2 lineages. Run `init` first.")
        return
    for _ in range(args.rounds):
        factory = _feed_factory(args.feed, eng._next_round_number(), args.bars)
        out = eng.run_round(ids[0], ids[1], feed_factory=factory)
        _print_outcome(out)
        _print_reflection(eng.last_reflection)
    print()
    _print_leaderboard(conn)


def cmd_forward_start(args):
    from . import forward
    conn = dbm.connect()
    dbm.init_db(conn)
    try:
        rid = forward.start_round(conn, days=args.days)
    except RuntimeError as exc:
        print(f"  {exc}")
        return
    st = forward.status(conn)
    print(f"Live round #{st['round_number']} started — {st['length_days']} "
          f"trading days, goal {st['goal_pct']}%, two autonomous agents from ₹"
          f"{DEFAULT.base_capital:,.0f} paper each.")
    print("  Run `pit forward-step` once per day after the NSE close (~15:30 IST).")


def cmd_forward_step(args):
    from . import forward
    from .llm import groq_available
    if not groq_available():
        print("  Autonomous agents need Groq. Set GROQ_API_KEY (and PIT_USE_LLM is\n"
              "  not required for forward mode).")
        return
    conn = dbm.connect()
    dbm.init_db(conn)
    print("Stepping the live round (agents researching real market)...")
    res = forward.step_round(conn, trade_date=args.date)
    status = res.get("status")
    if status in ("no_active_round", "already_processed", "not_a_trading_day"):
        print(f"  {status.replace('_', ' ')}" + (f" ({res.get('date')})" if res.get('date') else ""))
        return
    if status == "resolved":
        o = res["outcome"]
        print(f"\n  Day {res['day']} — ROUND RESOLVED")
        print(f"  WINNER {o['winner']} {o['winner_return']:+.2f}%  "
              f"beat {o['loser']} {o['loser_return']:+.2f}% ({o['reason']})")
        print(f"  mutation → {o['loser']}: {o['mutation_note']}")
    else:
        print(f"\n  Day {res['day']} of {res['of']} processed ({res['date']}).")


def cmd_forward_status(args):
    from . import forward
    conn = dbm.connect()
    dbm.init_db(conn)
    st = forward.status(conn)
    if not st:
        print("No live round in progress. Start one with `pit forward-start`.")
        return
    print(f"Live round #{st['round_number']}: day {st['days_done']}/"
          f"{st['length_days']}, goal {st['goal_pct']}%, started {st['started']}.")


def cmd_guidelines(args):
    from . import guidelines as gmod
    conn = dbm.connect()
    dbm.init_db(conn)
    active = gmod.active_guidelines(conn)
    print("\nActive guidelines (the pool's constitution)")
    print("-" * 68)
    if not active:
        print("  (none yet — a reflection pass proposes one every "
              f"{DEFAULT.reflection_interval} rounds if a pattern recurs)")
    for g in active:
        print(f"  #{g['id']} (v{g['version']}): {g['text']}")

    props = conn.execute(
        """SELECT p.id, p.kind, p.proposed_text, p.resolution,
                  SUM(v.vote='agree') agree, COUNT(v.id) total
           FROM guideline_proposals p
           LEFT JOIN guideline_votes v ON v.proposal_id = p.id
           GROUP BY p.id ORDER BY p.id DESC LIMIT 10""",
    ).fetchall()
    if props:
        print("\nRecent proposals")
        print("-" * 68)
        for p in props:
            res = p["resolution"] or "open"
            print(f"  #{p['id']} [{p['kind']}] {res} "
                  f"({p['agree'] or 0}/{p['total'] or 0} agreed): {p['proposed_text']}")


def cmd_leaderboard(args):
    conn = dbm.connect()
    dbm.init_db(conn)
    _print_leaderboard(conn)


def cmd_history(args):
    conn = dbm.connect()
    dbm.init_db(conn)
    where = "WHERE round_id=?" if args.round else ""
    params = (args.round,) if args.round else ()
    results = conn.execute(
        f"""SELECT r.round_number, rr.resolution_reason, rr.winner_return_pct,
                   rr.loser_return_pct, wl.name win, ll.name lose
            FROM round_results rr
            JOIN rounds r ON r.id = rr.round_id
            JOIN agents wa ON wa.id = rr.winner_agent_id
            JOIN agents la ON la.id = rr.loser_agent_id
            JOIN lineages wl ON wl.id = wa.lineage_id
            JOIN lineages ll ON ll.id = la.lineage_id
            {where.replace('round_id', 'r.id')}
            ORDER BY r.round_number""",
        params,
    ).fetchall()
    print("\nRound results")
    print("-" * 68)
    for row in results:
        print(f"  R{row['round_number']:>2}  {row['win']:<14} beat {row['lose']:<14} "
              f"[{row['resolution_reason']}]  {row['winner_return_pct']:+.2f}% vs "
              f"{row['loser_return_pct']:+.2f}%")
    if args.round:
        trades = conn.execute(
            """SELECT t.ts, l.name, t.symbol, t.side, t.qty, t.price, t.reason
               FROM trades t JOIN agents a ON a.id=t.agent_id
               JOIN lineages l ON l.id=a.lineage_id
               WHERE t.round_id=? ORDER BY t.id""",
            (args.round,),
        ).fetchall()
        print(f"\nTrades in round {args.round}")
        print("-" * 68)
        for t in trades:
            print(f"  {t['name']:<14} {t['side']:<4} {t['qty']:>6.0f} {t['symbol']:<12} "
                  f"@ {t['price']:>8.2f}  {t['reason'] or ''}")


# ---- printing ---------------------------------------------------------

def _print_outcome(out):
    print(f"\nRound {out.round_number}  ({out.length_days}d)  "
          f"— resolved by {out.reason}")
    print(f"  WINNER  {out.winner_lineage:<14} {out.winner_return_pct:+.2f}%   "
          f"(ELO +{out.rating_delta})")
    print(f"  loser   {out.loser_lineage:<14} {out.loser_return_pct:+.2f}%")
    print(f"  mutation → {out.loser_lineage}: {out.mutation_note}")


def _print_reflection(ref):
    if not ref:
        return
    verdict = "ADOPTED" if ref["accepted"] else "rejected"
    print(f"  reflection → proposal to {ref['kind']} a guideline "
          f"[{verdict}, {ref['agree']}/{ref['total']} agreed]")
    print(f"             \"{ref['text']}\"")


def _print_leaderboard(conn):
    rows = conn.execute(
        """SELECT l.name, l.wins, l.losses, l.cumulative_return_pct,
                  l.current_stake, a.generation, a.rating
           FROM lineages l JOIN agents a ON a.id = l.current_agent_id
           ORDER BY a.rating DESC""",
    ).fetchall()
    print("Leaderboard")
    print("-" * 74)
    print(f"  {'#':<3}{'Lineage':<15}{'Rating':>8}{'W-L':>8}{'Gen':>5}"
          f"{'Cum%':>9}{'Stake':>12}")
    print("-" * 74)
    for i, r in enumerate(rows, 1):
        wl = f"{r['wins']}-{r['losses']}"
        print(f"  {i:<3}{r['name']:<15}{r['rating']:>8.0f}{wl:>8}"
              f"{r['generation']:>5}{r['cumulative_return_pct']:>+9.2f}"
              f"{r['current_stake']:>12,.0f}")


def main(argv=None):
    p = argparse.ArgumentParser(prog="pit", description="Agent trading arena")
    sub = p.add_subparsers(dest="cmd", required=True)

    pi = sub.add_parser("init", help="create DB and seed lineages")
    pi.add_argument("--fresh", action="store_true", help="wipe existing DB first")
    pi.set_defaults(func=cmd_init)

    pr = sub.add_parser("run-round", help="run a single duel")
    pr.add_argument("--feed", choices=["synthetic", "historical"], default="synthetic")
    pr.add_argument("--bars", type=int, help="synthetic feed length (LLM runs: keep small)")
    pr.set_defaults(func=cmd_run_round)

    pa = sub.add_parser("arena", help="run several duels")
    pa.add_argument("--rounds", type=int, default=5)
    pa.add_argument("--feed", choices=["synthetic", "historical"], default="synthetic")
    pa.add_argument("--bars", type=int, help="synthetic feed length (LLM runs: keep small)")
    pa.set_defaults(func=cmd_arena)

    pl = sub.add_parser("leaderboard", help="show standings")
    pl.set_defaults(func=cmd_leaderboard)

    pg = sub.add_parser("guidelines", help="show the pool's shared guidelines")
    pg.set_defaults(func=cmd_guidelines)

    fs = sub.add_parser("forward-start", help="begin a live forward paper round")
    fs.add_argument("--days", type=int, help="trading days (default: config)")
    fs.set_defaults(func=cmd_forward_start)

    fp = sub.add_parser("forward-step", help="run one real trading day (autonomous agents)")
    fp.add_argument("--date", help="override the trade date (YYYY-MM-DD)")
    fp.set_defaults(func=cmd_forward_step)

    fst = sub.add_parser("forward-status", help="show the live round's progress")
    fst.set_defaults(func=cmd_forward_status)

    ph = sub.add_parser("history", help="show round results / trades")
    ph.add_argument("--round", type=int, help="show trades for this round id")
    ph.set_defaults(func=cmd_history)

    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
