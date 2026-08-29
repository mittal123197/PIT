"""The arena engine: run a round end-to-end, then resolve, rate, and mutate.

This is the orchestrator. It owns the tick loop (the arena's clock), honours
each agent's own wake cadence (heartbeat + price watches), enforces the single
hard constraint (stop-loss), and after the deadline applies the return cascade,
stake shift, ELO update, and the loser's mutation into a new generation.

Designed so the same `_simulate` loop backs both a real (persisted) round and an
in-memory sudden-death tiebreak.
"""
from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone

from . import rating
from .actions import Hold, PlaceOrder, SetHeartbeat, SetWatch
from .broker import Broker, InsufficientFunds, InvalidOrder, PaperBroker
from .config import DEFAULT, ArenaConfig
from .feeds import PriceFeed, SyntheticFeed
from .policies import AgentContext, AgentPolicy, SimplePolicy
from .portfolio import Portfolio
from .resolve import AgentSummary, resolve


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass
class _AgentRun:
    """Live bookkeeping for one agent while a round simulates."""
    agent_id: int
    lineage_id: int
    policy: AgentPolicy
    config: dict
    portfolio: Portfolio
    heartbeat_minutes: int
    last_wake: datetime | None = None
    watches: list[dict] | None = None       # in-memory active watches

    def __post_init__(self):
        if self.watches is None:
            self.watches = []


@dataclass
class RoundOutcome:
    round_id: int
    round_number: int
    length_days: int
    winner_agent_id: int
    loser_agent_id: int
    winner_lineage: str
    loser_lineage: str
    reason: str
    winner_return_pct: float
    loser_return_pct: float
    rating_delta: float
    mutation_note: str


class Engine:
    def __init__(
        self,
        conn: sqlite3.Connection,
        config: ArenaConfig = DEFAULT,
        broker: Broker | None = None,
        policy_factory=None,
        mutate_fn=None,
        use_llm: bool = False,
    ) -> None:
        self.conn = conn
        self.config = config
        self.use_llm = use_llm
        self.last_reflection: dict | None = None
        self.broker = broker or PaperBroker(
            per_order_fee=config.per_order_fee, allow_short=config.allow_short
        )
        # by default: deterministic policy + deterministic mutation (offline).
        self.policy_factory = policy_factory or (lambda agent_row: SimplePolicy())
        if mutate_fn is None:
            from .mutate import mutate_config
            mutate_fn = mutate_config
        self.mutate_fn = mutate_fn

    # ---- lineage / agent creation -------------------------------------

    def create_lineage(self, name: str, strategy_config: dict) -> int:
        hb = int(strategy_config.get("heartbeat_minutes", 60))
        activity = {"heartbeat_minutes": hb}
        cur = self.conn.execute(
            "INSERT INTO lineages (name, current_stake, created_at) VALUES (?,?,?)",
            (name, self.config.base_capital, _now_iso()),
        )
        lineage_id = cur.lastrowid
        agent_id = self._insert_agent(
            lineage_id, generation=1, strategy_config=strategy_config,
            activity_profile=activity, rating=self.config.elo_base,
            seeded_from=None, note=None,
        )
        self.conn.execute(
            "UPDATE lineages SET current_agent_id=? WHERE id=?", (agent_id, lineage_id)
        )
        self.conn.commit()
        return lineage_id

    def _insert_agent(self, lineage_id, generation, strategy_config,
                      activity_profile, rating, seeded_from, note) -> int:
        cur = self.conn.execute(
            """INSERT INTO agents
               (lineage_id, generation, strategy_config, activity_profile,
                rating, seeded_from_trade_agent_id, mutation_note, created_at)
               VALUES (?,?,?,?,?,?,?,?)""",
            (lineage_id, generation, json.dumps(strategy_config),
             json.dumps(activity_profile), rating, seeded_from, note, _now_iso()),
        )
        return cur.lastrowid

    # ---- running a round ----------------------------------------------

    def run_round(
        self, lineage_a_id: int, lineage_b_id: int,
        feed: PriceFeed | None = None, feed_factory=None,
    ) -> RoundOutcome:
        """Run one duel. `feed_factory(length_days) -> PriceFeed` lets the caller
        build a feed sized to the (only-now-known) round length; an explicit
        `feed` overrides it (used by tests)."""
        la = self._lineage(lineage_a_id)
        lb = self._lineage(lineage_b_id)
        agent_a = self._agent(la["current_agent_id"])
        agent_b = self._agent(lb["current_agent_id"])

        round_number = self._next_round_number()
        length_days = self._length_for(round_number)
        goal_pct = self.config.goal_pct_for(length_days)
        if feed is None:
            feed = (feed_factory(length_days) if feed_factory
                    else SyntheticFeed(
                        self.config.universe, seed=1000 + round_number,
                        bars=self.config.bars_for_days(length_days),
                        bar_minutes=self.config.bar_minutes))

        start_ts = feed.now().isoformat()
        deadline_ts = feed._timestamps[-1].isoformat() if hasattr(feed, "_timestamps") else start_ts
        cur = self.conn.execute(
            """INSERT INTO rounds
               (round_number, agent_a_id, agent_b_id, start_at, deadline,
                length_days, goal_pct, status, created_at)
               VALUES (?,?,?,?,?,?,?, 'running', ?)""",
            (round_number, agent_a["id"], agent_b["id"], start_ts, deadline_ts,
             length_days, goal_pct, _now_iso()),
        )
        round_id = cur.lastrowid

        runs = {
            agent_a["id"]: self._make_run(agent_a, la["current_stake"]),
            agent_b["id"]: self._make_run(agent_b, lb["current_stake"]),
        }
        for r in runs.values():
            self.conn.execute(
                """INSERT INTO round_states
                   (round_id, agent_id, starting_capital, current_capital, holdings)
                   VALUES (?,?,?,?, '{}')""",
                (round_id, r.agent_id, r.portfolio.starting_capital,
                 r.portfolio.cash),
            )
        self.conn.commit()

        from . import guidelines as gmod
        active = gmod.active_texts(self.conn)

        self._simulate(feed, runs, goal_pct, round_id=round_id, persist=True,
                       guidelines=active)
        final_prices = feed.prices()
        self._persist_final_states(round_id, runs, final_prices)

        outcome = self._resolve_round(round_id, round_number, length_days,
                                      la, lb, agent_a, agent_b, runs, final_prices)

        # reflection pass: every N rounds the pool may vote on a shared guideline
        self.last_reflection = None
        if (self.config.reflection_enabled
                and round_number % self.config.reflection_interval == 0):
            self.last_reflection = gmod.open_and_resolve(
                self.conn, self.config, round_id, self.use_llm)
        return outcome

    def _make_run(self, agent_row, stake: float) -> _AgentRun:
        cfg = json.loads(agent_row["strategy_config"])
        cfg.pop("_announced_heartbeat", None)
        hb = int(cfg.get("heartbeat_minutes", 60))
        return _AgentRun(
            agent_id=agent_row["id"],
            lineage_id=agent_row["lineage_id"],
            policy=self.policy_factory(agent_row),
            config=cfg,
            portfolio=Portfolio.open(stake),
            heartbeat_minutes=hb,
        )

    def _simulate(self, feed: PriceFeed, runs: dict[int, _AgentRun],
                  goal_pct: float, round_id: int | None, persist: bool,
                  guidelines: list[str] | None = None) -> None:
        """Step the feed bar-by-bar, waking + executing agents. The core loop."""
        guidelines = guidelines or []
        history: dict[str, list[float]] = {s: [] for s in feed.symbols}
        total_bars = len(feed)
        bar_index = 0

        while True:
            prices = feed.prices()
            now = feed.now()
            for sym, px in prices.items():
                history[sym].append(px)
            bars_remaining = total_bars - bar_index - 1

            # start-of-tick scoreboard: every agent sees each other's live
            # return % (but never their trades). Snapshotted before anyone acts
            # so ordering doesn't advantage whoever moves first.
            snapshot = {aid: r.portfolio.return_pct(prices) for aid, r in runs.items()}
            liq = {aid: r.portfolio.status == "liquidated" for aid, r in runs.items()}

            for aid, run in runs.items():
                if run.portfolio.status != "active":
                    continue
                run.portfolio.mark(prices)

                # 1. hard constraint: stop-loss
                if run.portfolio.return_pct(prices) <= -self.config.stop_loss_pct:
                    self._liquidate(run, prices, now, round_id, persist)
                    continue

                # 2. decide whether this agent wakes this bar
                fired = self._fire_watches(run, prices, now)
                due = (run.last_wake is None or
                       (now - run.last_wake).total_seconds() / 60.0
                       >= run.heartbeat_minutes)
                if not (due or fired):
                    continue
                run.last_wake = now

                # 3. act — with the opponent scoreboard (best of the others)
                others = [v for k, v in snapshot.items() if k != aid]
                opp_return = max(others) if others else 0.0
                opp_liq = all(liq[k] for k in runs if k != aid) if len(runs) > 1 else False
                ctx = AgentContext(
                    now=now, prices=prices, history=history,
                    cash=run.portfolio.cash, positions=dict(run.portfolio.positions),
                    return_pct=snapshot[aid],
                    universe=list(feed.symbols), strategy_config=run.config,
                    goal_pct=goal_pct, bars_remaining=bars_remaining,
                    opponent_return_pct=opp_return, opponent_liquidated=opp_liq,
                    guidelines=guidelines,
                )
                self._execute(run, run.policy.decide(ctx), prices, now, round_id, persist)

            if not feed.advance():
                break
            bar_index += 1
        if persist:
            self.conn.commit()

    def _execute(self, run, actions, prices, now, round_id, persist) -> None:
        for act in actions:
            if isinstance(act, PlaceOrder):
                px = prices.get(act.symbol)
                if px is None:
                    continue
                try:
                    fill = self.broker.execute(
                        run.portfolio, act.symbol, act.side, act.qty, px
                    )
                except (InsufficientFunds, InvalidOrder):
                    continue
                if persist:
                    self._log_trade(round_id, run.agent_id, now, fill, act.reason)
            elif isinstance(act, SetWatch):
                self._add_watch(run, act, prices.get(act.symbol), now, round_id, persist)
            elif isinstance(act, SetHeartbeat):
                run.heartbeat_minutes = max(1, int(act.minutes))
                run.config["heartbeat_minutes"] = run.heartbeat_minutes
            elif isinstance(act, Hold):
                continue

    def _liquidate(self, run, prices, now, round_id, persist) -> None:
        fills = self.broker.liquidate(run.portfolio, prices)
        run.portfolio.status = "liquidated"
        if persist:
            for fill in fills:
                self._log_trade(round_id, run.agent_id, now, fill, "stop-loss")
            self.conn.execute(
                "UPDATE round_states SET status='liquidated', liquidated_at=? "
                "WHERE round_id=? AND agent_id=?",
                (now.isoformat(), round_id, run.agent_id),
            )

    # ---- watches ------------------------------------------------------

    def _add_watch(self, run, act: SetWatch, ref_price, now, round_id, persist) -> None:
        watch = {
            "symbol": act.symbol, "trigger_type": act.trigger_type,
            "threshold": act.threshold, "reference": ref_price, "active": True,
        }
        run.watches.append(watch)
        if persist:
            self.conn.execute(
                """INSERT INTO price_watches
                   (round_id, agent_id, symbol, trigger_type, threshold,
                    reference, created_at)
                   VALUES (?,?,?,?,?,?,?)""",
                (round_id, run.agent_id, act.symbol, act.trigger_type,
                 act.threshold, ref_price, now.isoformat()),
            )

    @staticmethod
    def _fire_watches(run, prices, now) -> bool:
        fired = False
        for w in run.watches:
            if not w["active"]:
                continue
            px = prices.get(w["symbol"])
            if px is None:
                continue
            hit = False
            if w["trigger_type"] == "price_above":
                hit = px >= w["threshold"]
            elif w["trigger_type"] == "price_below":
                hit = px <= w["threshold"]
            elif w["trigger_type"] == "pct_move" and w["reference"]:
                hit = abs(px / w["reference"] - 1.0) * 100.0 >= w["threshold"]
            if hit:
                w["active"] = False
                fired = True
        return fired

    # ---- persistence helpers ------------------------------------------

    def _log_trade(self, round_id, agent_id, now, fill, reason) -> None:
        self.conn.execute(
            """INSERT INTO trades
               (round_id, agent_id, ts, symbol, side, qty, price, fee,
                capital_after, reason)
               VALUES (?,?,?,?,?,?,?,?,?,?)""",
            (round_id, agent_id, now.isoformat(), fill.symbol, fill.side,
             fill.qty, fill.price, fill.fee, fill.cash_after, reason),
        )

    def _persist_final_states(self, round_id, runs, prices) -> None:
        for run in runs.values():
            pf = run.portfolio
            self.conn.execute(
                """UPDATE round_states SET current_capital=?, holdings=?,
                   max_drawdown_pct=?, trade_count=?, final_return_pct=?
                   WHERE round_id=? AND agent_id=?""",
                (pf.cash, json.dumps(pf.positions), pf.max_drawdown_pct,
                 pf.trade_count, pf.return_pct(prices), round_id, run.agent_id),
            )
        self.conn.commit()

    # ---- resolution ---------------------------------------------------

    def _resolve_round(self, round_id, round_number, length_days,
                       la, lb, agent_a, agent_b, runs, prices) -> RoundOutcome:
        sa = self._summary(agent_a["id"], runs[agent_a["id"]], prices)
        sb = self._summary(agent_b["id"], runs[agent_b["id"]], prices)
        res = resolve(sa, sb, self.config.tie_epsilon_pct)

        if res.reason == "sudden_death":
            winner_id, loser_id, res_reason = self._sudden_death(agent_a, agent_b)
        else:
            winner_id, loser_id, res_reason = res.winner_id, res.loser_id, res.reason

        summ = {sa.agent_id: sa, sb.agent_id: sb}
        winner_return = summ[winner_id].return_pct
        loser_return = summ[loser_id].return_pct

        # map agent -> lineage rows
        by_agent = {agent_a["id"]: (la, agent_a), agent_b["id"]: (lb, agent_b)}
        win_lin, win_agent = by_agent[winner_id]
        lose_lin, lose_agent = by_agent[loser_id]

        # ELO (rating lives on the agent, carried into the next generation)
        nw, nl, delta = rating.update(
            win_agent["rating"], lose_agent["rating"], self.config.elo_k
        )
        self.conn.execute("UPDATE agents SET rating=? WHERE id=?", (nw, winner_id))
        self.conn.execute("UPDATE agents SET rating=? WHERE id=?", (nl, loser_id))

        # stake shift + cumulative return + W/L (asymmetric: winning pays more
        # than losing costs)
        wd = self.config.win_stake_bonus_pct / 100.0
        ld = self.config.loss_stake_penalty_pct / 100.0
        self._apply_lineage_result(win_lin, winner_return, won=True, stake_mult=1 + wd)
        self._apply_lineage_result(lose_lin, loser_return, won=False, stake_mult=1 - ld)

        self.conn.execute(
            """INSERT INTO round_results
               (round_id, winner_agent_id, loser_agent_id, resolution_reason,
                winner_return_pct, loser_return_pct, rating_delta, created_at)
               VALUES (?,?,?,?,?,?,?,?)""",
            (round_id, winner_id, loser_id, res_reason, winner_return,
             loser_return, delta, _now_iso()),
        )

        # mutation: loser's next generation, seeded from the winner's log
        note = self._mutate_loser(round_id, lose_lin, lose_agent, win_agent, nl)

        self.conn.execute("UPDATE rounds SET status='resolved' WHERE id=?", (round_id,))
        self.conn.commit()

        return RoundOutcome(
            round_id=round_id, round_number=round_number, length_days=length_days,
            winner_agent_id=winner_id, loser_agent_id=loser_id,
            winner_lineage=win_lin["name"], loser_lineage=lose_lin["name"],
            reason=res_reason, winner_return_pct=round(winner_return, 3),
            loser_return_pct=round(loser_return, 3), rating_delta=delta,
            mutation_note=note,
        )

    def _mutate_loser(self, round_id, lose_lin, lose_agent, win_agent, carried_rating) -> str:
        winner_trades = [
            dict(row) for row in self.conn.execute(
                "SELECT ts, symbol, side, qty, price, reason FROM trades "
                "WHERE round_id=? AND agent_id=? ORDER BY id",
                (round_id, win_agent["id"]),
            ).fetchall()
        ]
        winner_cfg = json.loads(win_agent["strategy_config"])
        loser_cfg = json.loads(lose_agent["strategy_config"])
        new_cfg, note = self.mutate_fn(winner_cfg, loser_cfg, winner_trades, round_id)

        new_agent_id = self._insert_agent(
            lineage_id=lose_lin["id"],
            generation=lose_agent["generation"] + 1,
            strategy_config=new_cfg,
            activity_profile={"heartbeat_minutes": int(new_cfg.get("heartbeat_minutes", 60))},
            rating=carried_rating,               # rating follows the lineage
            seeded_from=win_agent["id"],
            note=note,
        )
        self.conn.execute(
            "UPDATE lineages SET current_agent_id=? WHERE id=?",
            (new_agent_id, lose_lin["id"]),
        )
        return note

    def _apply_lineage_result(self, lin, round_return, won: bool, stake_mult: float) -> None:
        new_stake = max(1.0, lin["current_stake"] * stake_mult)
        cum = lin["cumulative_return_pct"]
        new_cum = ((1 + cum / 100.0) * (1 + round_return / 100.0) - 1.0) * 100.0
        self.conn.execute(
            """UPDATE lineages SET current_stake=?, cumulative_return_pct=?,
               wins=wins+?, losses=losses+? WHERE id=?""",
            (round(new_stake, 2), round(new_cum, 4), 1 if won else 0,
             0 if won else 1, lin["id"]),
        )

    def _summary(self, agent_id, run: _AgentRun, prices) -> AgentSummary:
        return AgentSummary(
            agent_id=agent_id,
            return_pct=run.portfolio.return_pct(prices),
            trade_count=run.portfolio.trade_count,
            max_drawdown_pct=run.portfolio.max_drawdown_pct,
            liquidated=run.portfolio.status == "liquidated",
        )

    def _sudden_death(self, agent_a, agent_b):
        """Rare tiebreak: a fresh, unpersisted 1-day mini-round."""
        feed = SyntheticFeed(self.config.universe, bars=26, seed=99991,
                             bar_minutes=15)
        runs = {
            agent_a["id"]: self._make_run(agent_a, self.config.base_capital),
            agent_b["id"]: self._make_run(agent_b, self.config.base_capital),
        }
        self._simulate(feed, runs, goal_pct=0.0, round_id=None, persist=False)
        prices = feed.prices()
        ra = runs[agent_a["id"]].portfolio.return_pct(prices)
        rb = runs[agent_b["id"]].portfolio.return_pct(prices)
        if abs(ra - rb) <= self.config.tie_epsilon_pct:
            # deterministic last resort
            win, lose = sorted([agent_a["id"], agent_b["id"]])
            return win, lose, "sudden_death"
        if ra > rb:
            return agent_a["id"], agent_b["id"], "sudden_death"
        return agent_b["id"], agent_a["id"], "sudden_death"

    # ---- small queries ------------------------------------------------

    def _lineage(self, lineage_id):
        return self.conn.execute(
            "SELECT * FROM lineages WHERE id=?", (lineage_id,)
        ).fetchone()

    def _agent(self, agent_id):
        return self.conn.execute(
            "SELECT * FROM agents WHERE id=?", (agent_id,)
        ).fetchone()

    def _next_round_number(self) -> int:
        row = self.conn.execute("SELECT COALESCE(MAX(round_number),0) n FROM rounds").fetchone()
        return int(row["n"]) + 1

    def _length_for(self, round_number: int) -> int:
        days = self.config.start_round_days - self.config.round_shrink_days * (round_number - 1)
        return max(self.config.min_round_days, days)
