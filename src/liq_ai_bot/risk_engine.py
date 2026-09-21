"""Risk / compliance layer — THE MOST IMPORTANT PART (DESIGN.md §3).

Invariant: this layer can only ever REDUCE or BLOCK risk, never add it.
A proposed trade is either:
  * approved as-is,
  * approved with a SMALLER size, or
  * rejected (blocked) with a reason.

It never increases size, never overrides a stop, never relaxes a limit.

Two independent drawdown trackers run at once:
  1. the FIRM's official calc (what actually fails your account), and
  2. a STRICTER internal calc (personal daily stop = half the firm's daily
     limit; internal max-DD buffer well inside the firm's floor).
The bot stops for the day / halts at the *internal* limit, leaving margin for
slippage and latency.

Pure standard library.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import List, Optional

from .config import DrawdownType, PropRules, RiskConfig, Settings


class Decision(str, Enum):
    APPROVE = "approve"
    REDUCE = "reduce"
    BLOCK = "block"


@dataclass
class ProposedTrade:
    """What the strategy wants to do, before risk vetting."""
    symbol: str
    side: str                 # "long" | "short"
    entry: float
    stop: float               # price; distance to entry defines 1R
    target: Optional[float] = None
    intended_risk_frac: Optional[float] = None  # override per-trade risk; else RiskConfig

    def stop_distance(self) -> float:
        d = abs(self.entry - self.stop)
        if d <= 0:
            raise ValueError("stop must differ from entry (zero-distance stop)")
        return d


@dataclass
class RiskVerdict:
    decision: Decision
    approved_size: float          # units/contracts (0.0 if blocked)
    risk_abs: float               # dollars at risk if stopped (0.0 if blocked)
    reasons: List[str] = field(default_factory=list)

    @property
    def approved(self) -> bool:
        return self.decision in (Decision.APPROVE, Decision.REDUCE) and self.approved_size > 0


@dataclass
class OpenPosition:
    symbol: str
    side: str
    size: float
    entry: float
    stop: float

    def open_risk_abs(self) -> float:
        return self.size * abs(self.entry - self.stop)


@dataclass
class AccountState:
    """Live equity/drawdown bookkeeping. `day_key` detects the firm reset."""
    start_balance: float
    balance: float
    equity: float                 # balance + unrealized
    peak_equity: float
    day_start_balance: float      # balance at the start of the current trading day
    day_key: str                  # e.g. "2026-09-20" in the firm's reset tz
    realized_today: float = 0.0   # signed P&L booked today
    open_positions: List[OpenPosition] = field(default_factory=list)
    trading_days: int = 0
    daily_pnls: List[float] = field(default_factory=list)  # realized P&L per completed day
    halted_today: bool = False

    @classmethod
    def new(cls, start_balance: float, now: Optional[datetime] = None,
            reset_hour_utc: int = 0) -> "AccountState":
        now = now or datetime.now(timezone.utc)
        return cls(
            start_balance=start_balance,
            balance=start_balance,
            equity=start_balance,
            peak_equity=start_balance,
            day_start_balance=start_balance,
            day_key=_day_key(now, reset_hour_utc),
        )

    def total_open_risk_abs(self) -> float:
        return sum(p.open_risk_abs() for p in self.open_positions)


def _day_key(ts: datetime, reset_hour_utc: int) -> str:
    """Trading-day bucket. A day rolls over at reset_hour_utc (00:00 by default)."""
    ts = ts.astimezone(timezone.utc)
    shifted = ts.timestamp() - reset_hour_utc * 3600
    d = datetime.fromtimestamp(shifted, tz=timezone.utc)
    return d.strftime("%Y-%m-%d")


class RiskEngine:
    """Stateful hard-constraint guard. Feed it market/account updates, then ask
    it to `vet()` each proposed trade."""

    def __init__(self, settings: Settings, state: AccountState):
        self.settings = settings
        self.rules: PropRules = settings.rules
        self.risk: RiskConfig = settings.risk
        self.state = state

    # -- clock / equity updates ------------------------------------------------
    def on_time(self, now: datetime) -> None:
        """Roll the trading day at the firm's reset hour."""
        key = _day_key(now, self.rules.daily_reset_hour_utc)
        if key != self.state.day_key:
            # close out the day
            self.state.daily_pnls.append(self.state.realized_today)
            if self.state.realized_today != 0.0 or self.state.trading_days == 0:
                self.state.trading_days += 1
            self.state.day_key = key
            self.state.day_start_balance = self.state.balance
            self.state.realized_today = 0.0
            self.state.halted_today = False

    def on_equity(self, balance: float, unrealized: float = 0.0) -> None:
        self.state.balance = balance
        self.state.equity = balance + unrealized
        if self.state.equity > self.state.peak_equity:
            self.state.peak_equity = self.state.equity

    def on_fill_closed(self, realized_pnl: float) -> None:
        self.state.balance += realized_pnl
        self.state.realized_today += realized_pnl
        self.on_equity(self.state.balance)

    # -- budget calculations ---------------------------------------------------
    def firm_daily_loss_used(self) -> float:
        """Dollars lost today by the firm's calc (loss is positive)."""
        return max(0.0, self.state.day_start_balance - self.state.equity)

    def firm_daily_budget(self) -> float:
        return self.rules.daily_loss_abs()

    def internal_daily_budget(self) -> float:
        return self.firm_daily_budget() * self.risk.daily_stop_fraction_of_firm

    def firm_drawdown_floor(self) -> float:
        """Equity floor at which the firm's max-DD is breached."""
        if self.rules.drawdown_type in (DrawdownType.TRAILING, DrawdownType.UNKNOWN):
            # stricter interpretation when unknown: trail the peak
            return self.state.peak_equity - self.rules.max_drawdown_abs()
        return self.state.start_balance - self.rules.max_drawdown_abs()

    def internal_drawdown_floor(self) -> float:
        """Our stricter floor: only consume a fraction of the allowed DD."""
        allowed = self.rules.max_drawdown_abs() * self.risk.internal_dd_fraction_of_firm
        anchor = (self.state.peak_equity
                  if self.rules.drawdown_type in (DrawdownType.TRAILING, DrawdownType.UNKNOWN)
                  else self.state.start_balance)
        return anchor - allowed

    def remaining_daily_budget(self) -> float:
        return max(0.0, self.internal_daily_budget() - self.firm_daily_loss_used())

    def remaining_dd_budget(self) -> float:
        return max(0.0, self.state.equity - self.internal_drawdown_floor())

    # -- consistency guard -----------------------------------------------------
    def best_day_cap_abs(self) -> Optional[float]:
        """Max profit allowed in a single day so no day exceeds the firm's
        best-day cap as a share of TOTAL profit. Returns None if no such rule."""
        cap = self.rules.best_day_cap
        if cap is None:
            return None
        total_profit = max(0.0, self.state.balance - self.state.start_balance)
        realized_today = max(0.0, self.state.realized_today)
        best_prior = max([0.0] + [p for p in self.state.daily_pnls if p > 0])
        # If today already leads, the future total must be large enough that
        # today/total <= cap  ->  total >= today/cap. Head-room today:
        current_total = total_profit
        leading = max(best_prior, realized_today)
        if current_total <= 0:
            return None
        # allowed best day given current total:
        allowed_best = cap * (current_total)
        # headroom is how much more we can bank today before today dominates.
        return max(0.0, allowed_best - realized_today)

    # -- the core: vet a proposed trade ---------------------------------------
    def vet(self, trade: ProposedTrade, now: Optional[datetime] = None) -> RiskVerdict:
        if now is not None:
            self.on_time(now)

        reasons: List[str] = []

        # Hard pre-conditions that BLOCK outright.
        if self.state.halted_today:
            return RiskVerdict(Decision.BLOCK, 0.0, 0.0, ["day halted (internal stop hit)"])

        if self.rules.automation_allowed is False:
            return RiskVerdict(Decision.BLOCK, 0.0, 0.0,
                               ["automation not permitted by firm rules"])

        if len(self.state.open_positions) >= self.risk.max_concurrent_positions:
            return RiskVerdict(Decision.BLOCK, 0.0, 0.0,
                               [f"max concurrent positions "
                                f"({self.risk.max_concurrent_positions}) reached"])

        # Budgets remaining (internal, stricter).
        rem_daily = self.remaining_daily_budget()
        rem_dd = self.remaining_dd_budget()
        governing_budget = min(rem_daily, rem_dd)

        # Kill-switch: if either budget is nearly exhausted, stop for the day.
        firm_daily = self.firm_daily_budget()
        if firm_daily > 0 and rem_daily <= self.risk.kill_switch_budget_fraction * (
            self.internal_daily_budget()
        ):
            self.state.halted_today = True
            return RiskVerdict(Decision.BLOCK, 0.0, 0.0,
                               ["kill-switch: daily budget nearly exhausted -> halt day"])
        if rem_dd <= self.risk.kill_switch_budget_fraction * (
            self.rules.max_drawdown_abs() * self.risk.internal_dd_fraction_of_firm
        ):
            self.state.halted_today = True
            return RiskVerdict(Decision.BLOCK, 0.0, 0.0,
                               ["kill-switch: drawdown budget nearly exhausted -> halt"])

        if governing_budget <= 0:
            return RiskVerdict(Decision.BLOCK, 0.0, 0.0, ["no risk budget remaining"])

        # Desired per-trade risk (dollars).
        risk_frac = trade.intended_risk_frac or self.risk.per_trade_risk
        risk_frac = min(risk_frac, self.risk.per_trade_risk)  # never exceed configured max
        desired_risk = self.state.balance * risk_frac

        # Cap by remaining open-risk allowance.
        open_risk_room = max(
            0.0, self.state.balance * self.risk.max_total_open_risk
            - self.state.total_open_risk_abs()
        )
        if open_risk_room <= 0:
            return RiskVerdict(Decision.BLOCK, 0.0, 0.0,
                               ["total open-risk allowance already used"])

        # Cap by firm's max single-trade loss (funded), if any.
        allowed_risk = desired_risk
        if self.rules.max_single_trade_loss is not None:
            allowed_risk = min(allowed_risk,
                               self.state.balance * self.rules.max_single_trade_loss)

        # Cap by remaining daily + DD budget and open-risk room.
        allowed_risk = min(allowed_risk, governing_budget, open_risk_room)

        # Consistency guard: don't let a single winning day dominate. This caps
        # the *upside* we pursue, implemented by shrinking target ambition, but
        # here we surface it as a note (sizing itself is loss-based).
        headroom = self.best_day_cap_abs()
        if headroom is not None and headroom <= 0:
            reasons.append("consistency: best-day cap reached -> prefer to bank/stand down")

        if allowed_risk <= 0:
            return RiskVerdict(Decision.BLOCK, 0.0, 0.0, ["risk capped to zero by guards"])

        # Convert dollar risk -> position size via stop distance.
        stop_dist = trade.stop_distance()
        size = allowed_risk / stop_dist
        size = _round_down(size, step=1e-6)
        if size <= 0:
            return RiskVerdict(Decision.BLOCK, 0.0, 0.0,
                               ["computed size rounds to zero"])

        final_risk = size * stop_dist
        decision = Decision.APPROVE
        if final_risk < desired_risk - 1e-9:
            decision = Decision.REDUCE
            reasons.append(
                f"reduced: risk ${final_risk:,.2f} < desired ${desired_risk:,.2f} "
                f"(governing budget ${governing_budget:,.2f})"
            )
        return RiskVerdict(decision, size, final_risk, reasons)

    # -- position lifecycle helpers -------------------------------------------
    def register_open(self, trade: ProposedTrade, size: float) -> None:
        self.state.open_positions.append(
            OpenPosition(trade.symbol, trade.side, size, trade.entry, trade.stop)
        )

    def register_close(self, symbol: str, realized_pnl: float) -> None:
        self.state.open_positions = [
            p for p in self.state.open_positions if p.symbol != symbol
        ]
        self.on_fill_closed(realized_pnl)


def _round_down(x: float, step: float) -> float:
    if step <= 0:
        return x
    return math.floor(x / step) * step
