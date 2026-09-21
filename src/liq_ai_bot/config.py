"""Configuration + prop-firm RULES model.

Everything the risk/compliance layer needs to enforce lives here. Values are
CONSERVATIVE PLACEHOLDERS that you MUST replace with MyFundedPerps' official,
current numbers (see DESIGN.md §0 RULES table). Until then every rule below is
marked ``confirmed=False`` and live/eval profiles will refuse to arm.

Pure standard library (dataclasses) so it runs offline. If you later add
pydantic, this maps 1:1 onto a BaseModel.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Optional


class DrawdownType(str, Enum):
    STATIC = "static"        # floor is fixed at (start_balance - max_dd)
    TRAILING = "trailing"    # floor trails peak equity (intraday or end-of-day)
    UNKNOWN = "unknown"      # not yet confirmed from docs -> treat as trailing (stricter)


class Profile(str, Enum):
    SANDBOX = "sandbox"      # offline/backtest; placeholders allowed
    EVAL = "eval"            # challenge; requires confirmed rules
    FUNDED = "funded"        # funded; requires confirmed rules


# Sentinel: a numeric rule we have NOT verified from the firm's docs.
# Kept as a real float so math works in backtests, but flagged unconfirmed.
UNCONFIRMED = None


@dataclass(frozen=True)
class PropRules:
    """The firm's hard rules. Fill `confirmed_*` flags as you verify each.

    All percentages are FRACTIONS of the starting balance (0.05 == 5%), unless
    a field says otherwise. Defaults are deliberately conservative guesses drawn
    from common perp-prop shapes (DESIGN.md §0) — NOT confirmed MyFundedPerps
    values.
    """

    account_size: float = 50_000.0

    # Profit targets (fraction of start balance)
    phase1_target: float = 0.10          # ~8-10% common
    phase2_target: Optional[float] = 0.05  # None if single-phase

    # Drawdown
    max_drawdown: float = 0.10           # overall
    drawdown_type: DrawdownType = DrawdownType.UNKNOWN
    daily_loss_limit: float = 0.05       # resets at reset hour
    daily_reset_hour_utc: int = 0        # 00:00 UTC common

    # Funded-only guards
    max_single_trade_loss: Optional[float] = None  # fraction; None = no explicit rule

    # Pacing / payout
    min_trading_days: int = 0
    time_limit_days: Optional[int] = None          # None = no limit
    best_day_cap: Optional[float] = 0.20  # best day <= 20% of total profit (consistency)
    payout_split: float = 0.90

    # Policy
    automation_allowed: Optional[bool] = None      # MUST confirm; None = unknown
    instruments: tuple = ("BTC/USDT:USDT", "ETH/USDT:USDT")

    # --- confirmation flags: flip to True only when verified from live docs ---
    confirmed_targets: bool = False
    confirmed_drawdown: bool = False
    confirmed_daily: bool = False
    confirmed_consistency: bool = False
    confirmed_automation: bool = False

    def all_confirmed(self) -> bool:
        return all(
            (
                self.confirmed_targets,
                self.confirmed_drawdown,
                self.confirmed_daily,
                self.confirmed_consistency,
                self.confirmed_automation,
            )
        )

    def unconfirmed_fields(self) -> list:
        missing = []
        if not self.confirmed_targets:
            missing.append("profit targets")
        if not self.confirmed_drawdown:
            missing.append("max drawdown (+ static/trailing type)")
        if not self.confirmed_daily:
            missing.append("daily loss limit (+ reset hour)")
        if not self.confirmed_consistency:
            missing.append("consistency / best-day cap")
        if not self.confirmed_automation:
            missing.append("automation / API policy")
        if self.drawdown_type is DrawdownType.UNKNOWN:
            missing.append("drawdown type is UNKNOWN (defaulting to trailing = stricter)")
        if self.automation_allowed is None:
            missing.append("automation_allowed is None (unknown)")
        return missing

    # absolute-dollar helpers -------------------------------------------------
    def max_drawdown_abs(self) -> float:
        return self.account_size * self.max_drawdown

    def daily_loss_abs(self) -> float:
        return self.account_size * self.daily_loss_limit

    def phase1_target_abs(self) -> float:
        return self.account_size * self.phase1_target


@dataclass(frozen=True)
class RiskConfig:
    """Our OWN discipline — always stricter than the firm's. The risk engine
    enforces the *tighter* of (firm rule, our rule)."""

    per_trade_risk: float = 0.0035        # 0.35% of balance per trade (DESIGN.md §5: 0.25-0.5%)
    # Personal daily stop = fraction of the FIRM's daily limit we allow ourselves to lose.
    daily_stop_fraction_of_firm: float = 0.50   # stop the day at half the firm's daily limit
    # Internal max-DD buffer: stop well inside the firm's floor (fraction of firm max DD).
    internal_dd_fraction_of_firm: float = 0.60   # use only 60% of the allowed drawdown
    max_concurrent_positions: int = 1
    max_total_open_risk: float = 0.01     # sum of open-position risk <= 1% of balance
    # Kill-switch: if remaining daily/DD budget drops below this fraction, stop trading.
    kill_switch_budget_fraction: float = 0.15

    def per_trade_risk_abs(self, balance: float) -> float:
        return balance * self.per_trade_risk


@dataclass(frozen=True)
class Settings:
    profile: Profile = Profile.SANDBOX
    rules: PropRules = field(default_factory=PropRules)
    risk: RiskConfig = field(default_factory=RiskConfig)
    execution_timeframe: str = "15m"      # chosen execution TF (open question #4)
    session_filter: bool = True           # prefer London/NY killzones

    def require_ready_for_live(self) -> None:
        """Raise if this config is not safe to arm against a real evaluation.

        The scaffold's fail-closed gate: you cannot accidentally run the EVAL or
        FUNDED profile on unverified placeholder rules, or with automation whose
        legality you haven't confirmed.
        """
        if self.profile is Profile.SANDBOX:
            return
        problems = self.rules.unconfirmed_fields()
        if self.rules.automation_allowed is not True:
            problems.append(
                "automation is not confirmed ALLOWED — do not run a bot until verified"
            )
        if problems:
            raise ConfigNotConfirmedError(
                f"Profile {self.profile.value!r} refuses to arm. Confirm from "
                f"MyFundedPerps docs and update config: {problems}"
            )


class ConfigNotConfirmedError(RuntimeError):
    """Raised when a live/eval profile is used with unverified placeholder rules."""


def sandbox_settings(**overrides) -> Settings:
    """Convenience: a safe offline profile for backtests/prototyping."""
    base = Settings(profile=Profile.SANDBOX)
    return replace(base, **overrides) if overrides else base
