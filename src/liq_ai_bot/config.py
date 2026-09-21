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
from typing import List, Optional


class DrawdownType(str, Enum):
    STATIC = "static"        # floor is fixed at (start_balance - max_dd)
    TRAILING = "trailing"    # floor trails peak equity (intraday or end-of-day)
    UNKNOWN = "unknown"      # not yet confirmed from docs -> treat as trailing (stricter)


class Profile(str, Enum):
    SANDBOX = "sandbox"      # offline/backtest; placeholders allowed
    EVAL = "eval"            # challenge; requires confirmed rules
    FUNDED = "funded"        # funded; requires confirmed rules


@dataclass(frozen=True)
class AutomationPolicy:
    """MyFundedPerps' confirmed automation rules, encoded as machine-checkable
    guardrails rather than prose. Automation IS permitted, but ONLY through the
    documented public API and only for ordinary (non-prohibited) trading.

    The clauses map directly onto this bot's design (DESIGN.md §0, §3):
      * documented API only — never touch the website's internal / browser-network
        endpoints (i.e. no scraping the UI's private XHR calls);
      * automation never exempts any ordinary trading rule (the risk/compliance
        layer still applies in full);
      * prohibited strategy classes stay prohibited when automated.

    These are enforced by ``require_ready_for_live``: a config whose behavior
    could fall into a prohibited class is refused before it can arm.
    """

    # Must route orders through the official/documented API surface only.
    documented_api_only: bool = True
    # Hard bans (all True == this bot must NOT do these).
    forbid_internal_browser_endpoints: bool = True   # no UI/dev-tools/network XHR endpoints
    forbid_hft: bool = True
    forbid_latency_arbitrage: bool = True
    forbid_quote_stuffing: bool = True
    forbid_platform_exploitation: bool = True         # e.g. gaming simulated fills
    forbid_cross_account_coordination: bool = True
    forbid_risk_limit_circumvention: bool = True
    # The firm may throttle strategies with excessive infra usage or execution
    # behavior materially different from ordinary trading.
    respect_infra_and_ordinary_execution: bool = True

    def violations(self, settings: "Settings") -> list:
        """Return reasons this configuration would breach the automation policy.

        This is a *design-time* screen against the prohibited classes, using what
        the config can reveal. It is necessarily conservative and NOT a substitute
        for the firm's own monitoring — but it makes the obvious footguns fail
        closed. Empty list == no detectable violation.
        """
        problems: List[str] = []
        # HFT / quote-stuffing / latency-arb proxy: this is an on-closed-bar,
        # one-position-at-a-time system. A sub-minute execution timeframe or
        # many concurrent positions would drift toward the prohibited HFT class.
        if self.forbid_hft:
            try:
                tf_s = _timeframe_seconds(settings.execution_timeframe)
                if tf_s < 60:
                    problems.append(
                        f"execution_timeframe {settings.execution_timeframe!r} is sub-minute "
                        f"-> risks the prohibited HFT/latency class; use >= 1m bars")
            except ValueError:
                problems.append(
                    f"execution_timeframe {settings.execution_timeframe!r} is unparseable")
        if self.forbid_hft and settings.risk.max_concurrent_positions > 5:
            problems.append(
                "max_concurrent_positions > 5 drifts toward HFT-like behavior")
        return problems


def _timeframe_seconds(tf: str) -> int:
    """Minimal timeframe parser (kept here to avoid importing datafeed in config)."""
    tf = tf.strip().lower()
    units = {"s": 1, "m": 60, "h": 3600, "d": 86_400, "w": 604_800}
    if not tf or not tf[-1].isalpha() or not tf[:-1].isdigit() or tf[-1] not in units:
        raise ValueError(f"invalid timeframe {tf!r}")
    qty = int(tf[:-1])
    if qty <= 0:
        raise ValueError(f"invalid timeframe {tf!r}")
    return qty * units[tf[-1]]


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

    # Profit targets (fraction of start balance).
    # Confirmed: $3,000 on a $50k account == 6%. Scales to other sizes.
    phase1_target: float = 0.06          # $3,000 / $50,000
    phase2_target: Optional[float] = None  # single-phase unless confirmed otherwise

    # Drawdown.
    # Confirmed: MLL $2,000 on $50k == 4%, STATIC (floor = start_balance - 4%).
    max_drawdown: float = 0.04           # $2,000 / $50,000
    drawdown_type: DrawdownType = DrawdownType.STATIC
    # Confirmed: DLL $1,000 on $50k == 2%. Resets at reset hour.
    daily_loss_limit: float = 0.02       # $1,000 / $50,000
    daily_reset_hour_utc: int = 0        # 00:00 UTC common

    # Funded-only guards
    max_single_trade_loss: Optional[float] = None  # fraction; None = no explicit rule

    # Pacing / payout
    min_trading_days: int = 0
    time_limit_days: Optional[int] = None          # None = no limit
    best_day_cap: Optional[float] = None  # confirmed: NO consistency / best-day rule
    payout_split: float = 0.90

    # Policy
    # Confirmed: automation IS permitted, but ONLY via the documented public API
    # (see AutomationPolicy below). Calling the website's internal/browser-network
    # endpoints is prohibited, as are HFT/latency-arb/quote-stuffing/exploitation/
    # cross-account/risk-limit-circumvention strategies. Automation never exempts
    # any ordinary trading rule.
    automation_allowed: bool = True
    instruments: tuple = ("BTC/USDT:USDT", "ETH/USDT:USDT")

    # --- confirmation flags: flip to True only when verified from live docs ---
    confirmed_targets: bool = True       # phase-1 target confirmed ($3k/$50k = 6%)
    confirmed_drawdown: bool = True      # MLL confirmed ($2k/$50k = 4%, static)
    confirmed_daily: bool = True         # DLL confirmed ($1k/$50k = 2%)
    confirmed_consistency: bool = True   # confirmed: no consistency/best-day rule
    confirmed_automation: bool = True    # confirmed: allowed via documented API only

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

    # Tightened for the confirmed 2% DLL / 4% static MLL box (a much smaller
    # room than the old 5%/10% placeholders — see DESIGN.md §5).
    per_trade_risk: float = 0.002         # 0.2% of balance per trade
    # Personal daily stop = fraction of the FIRM's daily limit we allow ourselves to lose.
    daily_stop_fraction_of_firm: float = 0.40   # stop the day at 40% of DLL (~0.8% of acct)
    # Internal max-DD buffer: stop well inside the firm's floor (fraction of firm max DD).
    internal_dd_fraction_of_firm: float = 0.50   # halt at 50% of MLL (~2% of acct), half the floor as buffer
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
    automation: AutomationPolicy = field(default_factory=AutomationPolicy)
    execution_timeframe: str = "15m"      # chosen execution TF (open question #4)
    session_filter: bool = True           # prefer London/NY killzones

    def require_ready_for_live(self) -> None:
        """Raise if this config is not safe to arm against a real evaluation.

        The scaffold's fail-closed gate: you cannot accidentally run the EVAL or
        FUNDED profile on unverified placeholder rules, with automation whose
        legality you haven't confirmed, or with a configuration that would fall
        into a prohibited automation class.
        """
        if self.profile is Profile.SANDBOX:
            return
        problems = self.rules.unconfirmed_fields()
        if self.rules.automation_allowed is not True:
            problems.append(
                "automation is not confirmed ALLOWED — do not run a bot until verified"
            )
        # Screen the configuration against the firm's prohibited automation classes.
        problems.extend(self.automation.violations(self))
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
