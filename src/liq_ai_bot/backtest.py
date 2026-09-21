"""Backtest harness + prop-challenge Monte-Carlo simulator (DESIGN.md §4).

Event-driven, on-closed-bar. Wires:  Strategy -> RiskEngine -> paper fills
-> journal. This is where an edge is proven or killed.

Two entry points:
  * `run_backtest(bars, ...)` — single deterministic pass over a bar series,
    returns per-trade results + summary metrics.
  * `monte_carlo_challenge(daily_r_samples, ...)` — resamples daily outcomes
    many times through the risk/compliance rules to estimate P(pass) and
    P(breach). This is the metric that actually matters for a prop challenge.

Pure standard library. Fills are modeled simply (stop/target intrabar with a
conservative "stop checked first" assumption). Costs (fees/funding/slippage)
are parameters — do NOT report edge without them (DESIGN.md §4 gate 4).
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable, Dict, List, Optional, Sequence

from .config import Settings
from .risk_engine import AccountState, Decision, ProposedTrade, RiskEngine
from .strategy import Strategy
from .types import Bar, Setup, Side


@dataclass
class TradeResult:
    setup: Setup
    size: float
    exit_price: float
    r_multiple: float
    pnl: float
    reason: str            # "target" | "stop" | "eod" | "blocked"


@dataclass
class BacktestReport:
    trades: List[TradeResult] = field(default_factory=list)
    blocked: int = 0
    final_balance: float = 0.0
    start_balance: float = 0.0
    max_drawdown_abs: float = 0.0
    worst_day: float = 0.0

    def wins(self) -> int:
        return sum(1 for t in self.trades if t.pnl > 0)

    def win_rate(self) -> float:
        return self.wins() / len(self.trades) if self.trades else 0.0

    def expectancy_r(self) -> float:
        return (sum(t.r_multiple for t in self.trades) / len(self.trades)
                if self.trades else 0.0)

    def total_return(self) -> float:
        return (self.final_balance - self.start_balance) / self.start_balance \
            if self.start_balance else 0.0

    def summary(self) -> Dict[str, float]:
        return {
            "trades": len(self.trades),
            "blocked": self.blocked,
            "win_rate": round(self.win_rate(), 4),
            "expectancy_r": round(self.expectancy_r(), 4),
            "total_return": round(self.total_return(), 4),
            "final_balance": round(self.final_balance, 2),
            "max_drawdown_abs": round(self.max_drawdown_abs, 2),
            "worst_day": round(self.worst_day, 2),
        }


@dataclass
class Costs:
    taker_fee: float = 0.0005     # 5 bps per side
    slippage_frac: float = 0.0002  # 2 bps
    # funding is per-bar-held; left 0 in the stub, wire in M2.
    funding_per_bar: float = 0.0

    def entry_exit_cost(self, notional: float) -> float:
        return notional * (self.taker_fee + self.slippage_frac) * 2.0


def _exit_on_bar(setup: Setup, bar: Bar) -> Optional[tuple]:
    """Conservative intrabar fill: if both stop and target are touched in the
    same bar, assume the STOP hit first. Returns (exit_price, reason) or None."""
    if setup.side is Side.LONG:
        hit_stop = bar.low <= setup.stop
        hit_tgt = setup.target is not None and bar.high >= setup.target
        if hit_stop:
            return (setup.stop, "stop")
        if hit_tgt:
            return (setup.target, "target")
    else:
        hit_stop = bar.high >= setup.stop
        hit_tgt = setup.target is not None and bar.low <= setup.target
        if hit_stop:
            return (setup.stop, "stop")
        if hit_tgt:
            return (setup.target, "target")
    return None


def run_backtest(
    bars: Sequence[Bar],
    strategy: Strategy,
    settings: Settings,
    costs: Optional[Costs] = None,
    journal: Optional[Callable[[dict], None]] = None,
) -> BacktestReport:
    """Single deterministic pass. One position at a time (matches
    max_concurrent_positions=1 default)."""
    costs = costs or Costs()
    start = settings.rules.account_size
    state = AccountState.new(start, reset_hour_utc=settings.rules.daily_reset_hour_utc)
    engine = RiskEngine(settings, state)
    report = BacktestReport(start_balance=start, final_balance=start)

    open_trade: Optional[TradeResult] = None
    trough = start
    day_open_balance = start
    prev_day_key = state.day_key

    for bar in bars:
        now = datetime.fromtimestamp(bar.ts, tz=timezone.utc)
        engine.on_time(now)

        # worst-day tracking on day roll
        if state.day_key != prev_day_key:
            report.worst_day = min(report.worst_day, state.balance - day_open_balance)
            day_open_balance = state.balance
            prev_day_key = state.day_key

        # 1) manage an open position first
        if open_trade is not None:
            ex = _exit_on_bar(open_trade.setup, bar)
            if ex is not None:
                exit_price, reason = ex
                s = open_trade.setup
                gross = ((exit_price - s.entry) if s.side is Side.LONG
                         else (s.entry - exit_price)) * open_trade.size
                notional = s.entry * open_trade.size
                pnl = gross - costs.entry_exit_cost(notional)
                open_trade.exit_price = exit_price
                open_trade.pnl = pnl
                open_trade.r_multiple = s.r_multiple_to(exit_price)
                open_trade.reason = reason
                engine.register_close(s.symbol, pnl)
                report.trades.append(open_trade)
                if journal:
                    journal({"event": "close", "reason": reason, "pnl": pnl,
                             "features": s.features})
                open_trade = None

        # equity + drawdown bookkeeping
        engine.on_equity(state.balance)
        trough = min(trough, state.equity)
        report.max_drawdown_abs = max(report.max_drawdown_abs,
                                      state.peak_equity - state.equity)

        # 2) look for a new setup only if flat
        if open_trade is None:
            setup = strategy.on_bar(bar)
            if setup is not None:
                proposed = ProposedTrade(
                    symbol=setup.symbol, side=setup.side.value,
                    entry=setup.entry, stop=setup.stop, target=setup.target,
                )
                verdict = engine.vet(proposed, now=now)
                if verdict.approved:
                    engine.register_open(proposed, verdict.approved_size)
                    open_trade = TradeResult(
                        setup=setup, size=verdict.approved_size,
                        exit_price=0.0, r_multiple=0.0, pnl=0.0, reason="open")
                    if journal:
                        journal({"event": "open", "side": setup.side.value,
                                 "size": verdict.approved_size,
                                 "decision": verdict.decision.value,
                                 "reasons": verdict.reasons})
                else:
                    report.blocked += 1
                    if journal:
                        journal({"event": "blocked", "reasons": verdict.reasons})
        else:
            # already in a trade -> strategy still updates its internal state
            strategy.on_bar(bar)

    report.final_balance = state.balance
    return report


# ---- Monte-Carlo challenge simulator ----------------------------------------
@dataclass
class ChallengeOutcome:
    passed: int = 0
    breached: int = 0
    timed_out: int = 0
    runs: int = 0

    def p_pass(self) -> float:
        return self.passed / self.runs if self.runs else 0.0

    def p_breach(self) -> float:
        return self.breached / self.runs if self.runs else 0.0

    def summary(self) -> Dict[str, float]:
        return {
            "runs": self.runs,
            "p_pass": round(self.p_pass(), 4),
            "p_breach": round(self.p_breach(), 4),
            "p_timeout": round(self.timed_out / self.runs, 4) if self.runs else 0.0,
        }


def monte_carlo_challenge(
    daily_r_samples: Sequence[float],
    settings: Settings,
    n_runs: int = 5000,
    trades_per_day: int = 2,
    max_days: int = 60,
    seed: Optional[int] = 42,
) -> ChallengeOutcome:
    """Resample daily P&L (in R multiples) through the risk/compliance rules to
    estimate P(pass phase 1) and P(breach).

    `daily_r_samples` is your empirical per-trade R distribution (from
    `run_backtest`). Each simulated day draws `trades_per_day` R outcomes,
    sizes each at the configured per-trade risk, applies daily-loss + max-DD
    breach checks, and stops at the profit target.

    This is intentionally simple and honest: it models the *account curve under
    the rules*, which is what determines pass/fail — not price microstructure.
    """
    if not daily_r_samples:
        raise ValueError("need a non-empty R-multiple sample from a backtest")
    rng = random.Random(seed)
    rules = settings.rules
    risk = settings.risk
    start = rules.account_size
    target_abs = rules.phase1_target_abs()
    daily_limit = rules.daily_loss_abs() * risk.daily_stop_fraction_of_firm  # internal stop
    max_dd_abs = rules.max_drawdown_abs() * risk.internal_dd_fraction_of_firm

    out = ChallengeOutcome()
    samples = list(daily_r_samples)

    for _ in range(n_runs):
        out.runs += 1
        balance = start
        peak = start
        result = "timeout"
        for _day in range(max_days):
            day_start = balance
            day_pnl = 0.0
            for _t in range(trades_per_day):
                r = rng.choice(samples)
                per_trade_risk_abs = balance * risk.per_trade_risk
                day_pnl += r * per_trade_risk_abs
                cur = balance + day_pnl
                peak = max(peak, cur)
                # trailing/unknown DD anchors on peak; static on start
                floor = (peak - max_dd_abs
                         if rules.drawdown_type.value in ("trailing", "unknown")
                         else start - max_dd_abs)
                if cur <= floor:
                    result = "breach"; break
                if (day_start - cur) >= daily_limit:
                    result = "breach"; break   # internal daily stop treated as hard
            balance += day_pnl
            if result == "breach":
                break
            if balance - start >= target_abs:
                result = "pass"; break
        if result == "pass":
            out.passed += 1
        elif result == "breach":
            out.breached += 1
        else:
            out.timed_out += 1
    return out
