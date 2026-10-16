"""Command-line entrypoint for the arena.

    python -m pit.cli live --minutes 60 --interval 300     # a live US/crypto battle
    python -m pit.cli live --replay --date 2026-10-09 --compress 10
    python -m pit.cli leaderboard                          # standings
    python -m pit.cli guidelines                           # the pool's rulebook
    python -m pit.cli history [--round N]                  # results / trades

scripts/run_arena.sh wraps `live` (several battles, dashboard, keep-awake).
The database and the three lineages are created on first use.
"""
from __future__ import annotations

import argparse

from . import db as dbm


# ---- commands ---------------------------------------------------------











def cmd_live(args):
    from . import live, replay
    conn = dbm.connect()
    dbm.init_db(conn)
    if args.replay:
        from datetime import date
        day = date.fromisoformat(args.date) if args.date else None
        replay.start(day=day, compress_minutes=args.compress)
        # Compressed replay -> proportionally shorter decision interval so we
        # get multiple decisions during the compressed session.
        interval = args.interval if args.interval != 900 else max(60, args.compress * 60 // 5)
        try:
            live.run_live(conn, minutes=args.compress + 1,
                          interval=interval, refresh=args.refresh,
                          debug=args.debug)
        finally:
            replay.stop()
    else:
        live.run_live(conn, minutes=args.minutes, interval=args.interval,
                      refresh=args.refresh, debug=args.debug)




def cmd_guidelines(args):
    from . import guidelines as gmod
    conn = dbm.connect()
    dbm.init_db(conn)
    active = gmod.active_guidelines(conn)
    print("\nActive guidelines (the pool's constitution)")
    print("-" * 68)
    if not active:
        print("  (none yet — agents propose rules after each round; the arena "
              "researches and the pool debates and votes)")
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
    # --round takes the user-facing round NUMBER (the "R<n>" this same command
    # prints below, and what the leaderboard/dashboard show) — resolve it to
    # the internal rounds.id ourselves. Filtering directly on the id used to
    # silently return nothing whenever the two diverged (they're only equal
    # by coincidence; ids are a plain autoincrement, round_number is the
    # sequential label), which reads as "no trades" rather than "wrong id".
    round_id = None
    if args.round:
        row = conn.execute("SELECT id FROM rounds WHERE round_number=?",
                           (args.round,)).fetchone()
        if not row:
            print(f"No round numbered {args.round} found.")
            return
        round_id = row["id"]
    where = "WHERE r.id=?" if round_id else ""
    params = (round_id,) if round_id else ()
    # Full placement (1st/2nd/3rd + return%), not winner-vs-loser — round_results
    # only ever records the top and bottom finisher, silently dropping anyone
    # in the middle once a round has 3+ participants (which every round does
    # now: the whole pool trades together, not a rotating 1v1).
    rounds = conn.execute(
        f"""SELECT r.id, r.round_number, rr.resolution_reason
            FROM rounds r LEFT JOIN round_results rr ON rr.round_id = r.id
            {where}
            ORDER BY r.round_number""",
        params,
    ).fetchall()
    round_ids = [r["id"] for r in rounds]
    rankings_by_round: dict[int, list] = {}
    if round_ids:
        placeholders = ",".join("?" * len(round_ids))
        for row in conn.execute(
            f"""SELECT rr.round_id, rr.rank, rr.return_pct, l.name
                FROM round_rankings rr
                JOIN agents a ON a.id = rr.agent_id
                JOIN lineages l ON l.id = a.lineage_id
                WHERE rr.round_id IN ({placeholders})
                ORDER BY rr.round_id, rr.rank""",
            round_ids,
        ).fetchall():
            rankings_by_round.setdefault(row["round_id"], []).append(row)
    print("\nRound results")
    print("-" * 68)
    for row in rounds:
        places = rankings_by_round.get(row["id"]) or []
        if not places:
            print(f"  R{row['round_number']:>2}  (unresolved)")
            continue
        placed = "  ".join(f"{p['rank']}. {p['name']} {p['return_pct']:+.2f}%"
                           for p in places)
        reason = f" [{row['resolution_reason']}]" if row["resolution_reason"] else ""
        print(f"  R{row['round_number']:>2}  {placed}{reason}")
    if round_id:
        trades = conn.execute(
            """SELECT t.ts, l.name, t.symbol, t.side, t.qty, t.price, t.reason
               FROM trades t JOIN agents a ON a.id=t.agent_id
               JOIN lineages l ON l.id=a.lineage_id
               WHERE t.round_id=? ORDER BY t.id""",
            (round_id,),
        ).fetchall()
        print(f"\nTrades in round {args.round}")
        print("-" * 68)
        for t in trades:
            ts = t["ts"][11:19] if t["ts"] and "T" in t["ts"] else (t["ts"] or "")
            print(f"  {ts:<19} {t['name']:<14} {t['side']:<4} {t['qty']:>6.0f} {t['symbol']:<12} "
                  f"@ {t['price']:>8.2f}  {t['reason'] or ''}")


# ---- printing ---------------------------------------------------------





def _print_leaderboard(conn):
    rows = conn.execute(
        """SELECT l.name, l.wins, l.losses, l.cumulative_return_pct,
                  a.generation, a.rating
           FROM lineages l JOIN agents a ON a.id = l.current_agent_id
           ORDER BY a.rating DESC""",
    ).fetchall()
    print("Leaderboard")
    print("-" * 74)
    print(f"  {'#':<3}{'Lineage':<15}{'Rating':>8}{'W-L':>8}{'Gen':>5}"
          f"{'Cum%':>9}")
    print("-" * 74)
    for i, r in enumerate(rows, 1):
        wl = f"{r['wins']}-{r['losses']}"
        print(f"  {i:<3}{r['name']:<15}{r['rating']:>8.0f}{wl:>8}"
              f"{r['generation']:>5}{r['cumulative_return_pct']:>+9.2f}")


def main(argv=None):
    p = argparse.ArgumentParser(prog="pit", description="Agent trading arena")
    sub = p.add_subparsers(dest="cmd", required=True)




    pl = sub.add_parser("leaderboard", help="show standings")
    pl.set_defaults(func=cmd_leaderboard)

    pg = sub.add_parser("guidelines", help="show the pool's shared guidelines")
    pg.set_defaults(func=cmd_guidelines)




    lv = sub.add_parser("live", help="run a live session: agents trade + trash-talk")
    lv.add_argument("--minutes", type=int, default=180, help="session length")
    lv.add_argument("--interval", type=int, default=900,
                    help="seconds between ticks (default 15 min — swing pace)")
    lv.add_argument("--replay", action="store_true",
                    help="replay a past trading day at compressed speed")
    lv.add_argument("--date", help="replay day (YYYY-MM-DD); default = last trading day")
    lv.add_argument("--compress", type=int, default=20,
                    help="replay: wall-clock minutes for the full session (default 20)")
    lv.add_argument("--refresh", type=int, default=10,
                    help="seconds between price/P&L marks (default 10)")
    lv.add_argument("--debug", action="store_true",
                    help="record every thought, tool call, and research result "
                         "per decision to the audit log (visible on /live)")
    lv.set_defaults(func=cmd_live)

    ph = sub.add_parser("history", help="show round results / trades")
    ph.add_argument("--round", type=int,
                    help="show trades for this round number (the R<n> shown "
                         "in the results list, not the internal db id)")
    ph.set_defaults(func=cmd_history)

    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
