"""M3/M4 walk-forward validation harness (DESIGN.md §4 gates 2-4).

This is where the base strategy earns the right to exist — or gets retired. It
answers the only question that matters before risking a challenge: **does the edge
survive out-of-sample and after costs?**

Pipeline (all deterministic, pure standard library):

  1. `walk_forward_splits`  — carve the bar series into rolling train/test folds
     with an embargo gap so no in-sample bar leaks into its test window.
  2. `tune_params`         — grid-search `SignalParams` on the TRAIN window, pick
     the params with the best in-sample objective (expectancy penalized for
     thin trade counts). This is the only place params are fit.
  3. `run_walk_forward`    — for each fold: tune on train, then evaluate those
     frozen params on the untouched TEST window. Concatenate the out-of-sample
     trades across folds — that pooled OOS record is the honest edge estimate.
  4. `cost_sensitivity`    — re-run the OOS backtest across a fee/slippage grid;
     an edge that only exists at zero cost is not an edge (gate 4).
  5. `go_no_go`            — combine OOS expectancy, in-sample-vs-OOS decay, cost
     survival, and the Monte-Carlo P(pass)/P(breach) under the *confirmed* prop
     rules into a single, honest verdict.

Nothing here promises profit. A NO-GO is the expected — and correct — output for a
strategy without a real edge; the harness is built to say so plainly.
"""
from __future__ import annotations

import itertools
import statistics
from dataclasses import dataclass, field, replace
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from .backtest import Costs, monte_carlo_challenge, run_backtest
from .config import Settings, sandbox_settings
from .strategy import LiquiditySFPStrategy, SignalParams
from .levels import LevelParams
from .types import Bar


# ============================================================================
# 1) WALK-FORWARD SPLITTER
# ============================================================================
@dataclass(frozen=True)
class Fold:
    index: int
    train: List[Bar]
    test: List[Bar]

    def train_span(self) -> Tuple[int, int]:
        return (self.train[0].ts, self.train[-1].ts) if self.train else (0, 0)

    def test_span(self) -> Tuple[int, int]:
        return (self.test[0].ts, self.test[-1].ts) if self.test else (0, 0)


def walk_forward_splits(
    bars: Sequence[Bar],
    train_size: int,
    test_size: int,
    *,
    step: Optional[int] = None,
    embargo: int = 0,
    anchored: bool = False,
) -> List[Fold]:
    """Generate rolling (or anchored) train/test folds over an ordered bar series.

    Args:
        train_size: bars in each in-sample (training) window.
        test_size:  bars in each out-of-sample (test) window.
        step:       bars to advance the window each fold (default = test_size, so
                    test windows tile the series without overlap).
        embargo:    bars skipped between the end of train and the start of test.
                    Guards against leakage from indicators that look back across
                    the boundary (ATR/pivots warm up on train's tail).
        anchored:   if True, every train window starts at bar 0 and grows
                    (expanding-window walk-forward); otherwise it rolls.

    Returns folds in chronological order. A fold is emitted only if a full
    test_size window fits after train + embargo.
    """
    if train_size <= 0 or test_size <= 0:
        raise ValueError("train_size and test_size must be positive")
    step = step or test_size
    n = len(bars)
    folds: List[Fold] = []
    train_start = 0
    fold_i = 0
    # the first test window can start once train + embargo bars exist
    cursor = train_size
    while cursor + embargo + test_size <= n:
        te_start = cursor + embargo
        te_end = te_start + test_size
        tr_start = 0 if anchored else train_start
        tr = list(bars[tr_start:cursor])
        te = list(bars[te_start:te_end])
        folds.append(Fold(fold_i, tr, te))
        fold_i += 1
        cursor += step
        if not anchored:
            train_start += step
    return folds


# ============================================================================
# 2) PARAMETER TUNER (in-sample only)
# ============================================================================
@dataclass(frozen=True)
class ParamGrid:
    """Search space over SignalParams. Keep it small: a big grid on one window is
    just curve-fitting (DESIGN.md §4 gate 3)."""
    rvol_min: Tuple[float, ...] = (1.2, 1.5, 1.8)
    confluence_min: Tuple[float, ...] = (30.0, 45.0, 60.0)
    conviction_min: Tuple[float, ...] = (0.2, 0.3, 0.4)
    stop_pad_atr: Tuple[float, ...] = (0.25,)
    fallback_r: Tuple[float, ...] = (2.0,)

    def combinations(self) -> List[SignalParams]:
        out = []
        for rv, cf, cv, sp, fr in itertools.product(
            self.rvol_min, self.confluence_min, self.conviction_min,
            self.stop_pad_atr, self.fallback_r
        ):
            out.append(SignalParams(rvol_min=rv, confluence_min=cf,
                                    conviction_min=cv, stop_pad_atr=sp, fallback_r=fr))
        return out


def _make_strategy(symbol: str, params: SignalParams,
                   level_params: Optional[LevelParams]) -> LiquiditySFPStrategy:
    # fresh strategy => fresh LevelEngine state per evaluation (no leakage)
    return LiquiditySFPStrategy(symbol, params=params, level_params=level_params)


def _objective(expectancy_r: float, n_trades: int, min_trades: int) -> float:
    """In-sample score. Expectancy in R, penalized when the sample is too thin to
    trust (a 3-trade 5R average is noise, not edge)."""
    if n_trades < min_trades:
        # scale the score down toward zero as trades fall below the floor
        return expectancy_r * (n_trades / min_trades) - (min_trades - n_trades) * 0.05
    return expectancy_r


@dataclass
class TuneResult:
    best_params: SignalParams
    best_score: float
    best_expectancy_r: float
    best_trades: int
    evaluated: int


def tune_params(
    train_bars: Sequence[Bar],
    settings: Settings,
    *,
    symbol: str = "BTC/USDT:USDT",
    grid: Optional[ParamGrid] = None,
    costs: Optional[Costs] = None,
    level_params: Optional[LevelParams] = None,
    min_trades: int = 15,
) -> TuneResult:
    """Grid-search SignalParams on ONE training window; return the best set.

    Ties break toward MORE trades (more robust), then toward the default-ish
    middle of the grid. If no combo trades at all, returns the first grid point
    with a score of -inf so the caller can detect a dead window.
    """
    grid = grid or ParamGrid()
    combos = grid.combinations()
    best: Optional[TuneResult] = None
    for p in combos:
        strat = _make_strategy(symbol, p, level_params)
        report = run_backtest(train_bars, strat, settings, costs=costs)
        n = len(report.trades)
        exp = report.expectancy_r()
        score = _objective(exp, n, min_trades)
        if (best is None or score > best.best_score
                or (score == best.best_score and n > best.best_trades)):
            best = TuneResult(p, score, exp, n, len(combos))
    return best


# ============================================================================
# 3) WALK-FORWARD DRIVER
# ============================================================================
@dataclass
class FoldResult:
    index: int
    train_span: Tuple[int, int]
    test_span: Tuple[int, int]
    params: SignalParams
    in_sample_expectancy_r: float
    in_sample_trades: int
    oos_expectancy_r: float
    oos_trades: int
    oos_r_multiples: List[float] = field(default_factory=list)


@dataclass
class WalkForwardResult:
    folds: List[FoldResult]
    oos_r_multiples: List[float]            # pooled OOS trade R across all folds

    # ---- pooled OOS metrics ----
    def oos_trades(self) -> int:
        return len(self.oos_r_multiples)

    def oos_expectancy_r(self) -> float:
        return statistics.fmean(self.oos_r_multiples) if self.oos_r_multiples else 0.0

    def oos_win_rate(self) -> float:
        if not self.oos_r_multiples:
            return 0.0
        return sum(1 for r in self.oos_r_multiples if r > 0) / len(self.oos_r_multiples)

    def mean_in_sample_expectancy_r(self) -> float:
        vals = [f.in_sample_expectancy_r for f in self.folds if f.in_sample_trades > 0]
        return statistics.fmean(vals) if vals else 0.0

    def is_oos_decay(self) -> float:
        """In-sample minus out-of-sample expectancy. Large positive == overfit
        (looked great on train, collapsed on unseen data)."""
        return self.mean_in_sample_expectancy_r() - self.oos_expectancy_r()

    def summary(self) -> Dict[str, float]:
        return {
            "folds": len(self.folds),
            "oos_trades": self.oos_trades(),
            "oos_expectancy_r": round(self.oos_expectancy_r(), 4),
            "oos_win_rate": round(self.oos_win_rate(), 4),
            "mean_in_sample_expectancy_r": round(self.mean_in_sample_expectancy_r(), 4),
            "is_minus_oos_decay": round(self.is_oos_decay(), 4),
        }


def run_walk_forward(
    bars: Sequence[Bar],
    *,
    train_size: int,
    test_size: int,
    settings: Optional[Settings] = None,
    symbol: str = "BTC/USDT:USDT",
    grid: Optional[ParamGrid] = None,
    costs: Optional[Costs] = None,
    level_params: Optional[LevelParams] = None,
    embargo: int = 0,
    anchored: bool = False,
    min_trades: int = 15,
) -> WalkForwardResult:
    """Full walk-forward: tune on each train window, evaluate frozen params on the
    following (untouched) test window, pool the OOS trades.

    The pooled OOS R-distribution is the number that matters — it's the strategy's
    performance on data it was never tuned against.
    """
    settings = settings or sandbox_settings()
    costs = costs or Costs()
    folds = walk_forward_splits(bars, train_size, test_size,
                                embargo=embargo, anchored=anchored)
    fold_results: List[FoldResult] = []
    pooled_oos: List[float] = []
    for fold in folds:
        tuned = tune_params(fold.train, settings, symbol=symbol, grid=grid,
                            costs=costs, level_params=level_params, min_trades=min_trades)
        # evaluate frozen params on the untouched test window
        strat = _make_strategy(symbol, tuned.best_params, level_params)
        rep = run_backtest(fold.test, strat, settings, costs=costs)
        oos_r = [t.r_multiple for t in rep.trades]
        pooled_oos.extend(oos_r)
        fold_results.append(FoldResult(
            index=fold.index,
            train_span=fold.train_span(),
            test_span=fold.test_span(),
            params=tuned.best_params,
            in_sample_expectancy_r=tuned.best_expectancy_r,
            in_sample_trades=tuned.best_trades,
            oos_expectancy_r=(statistics.fmean(oos_r) if oos_r else 0.0),
            oos_trades=len(oos_r),
            oos_r_multiples=oos_r,
        ))
    return WalkForwardResult(fold_results, pooled_oos)


# ============================================================================
# 4) COST SENSITIVITY SWEEP
# ============================================================================
@dataclass
class CostPoint:
    taker_fee: float
    slippage_frac: float
    oos_expectancy_r: float
    oos_trades: int


def cost_sensitivity(
    bars: Sequence[Bar],
    params: SignalParams,
    settings: Settings,
    *,
    symbol: str = "BTC/USDT:USDT",
    level_params: Optional[LevelParams] = None,
    fee_grid: Sequence[float] = (0.0, 0.0002, 0.0005, 0.001),
    slippage_grid: Sequence[float] = (0.0, 0.0002, 0.0005),
) -> List[CostPoint]:
    """Re-run a single backtest of `params` on `bars` across a fee/slippage grid.

    Used on the pooled OOS window (or a held-out slice) to confirm the edge does
    not evaporate once realistic frictions are applied (DESIGN.md §4 gate 4).
    """
    points: List[CostPoint] = []
    for fee in fee_grid:
        for slip in slippage_grid:
            strat = _make_strategy(symbol, params, level_params)
            rep = run_backtest(bars, strat, settings,
                               costs=Costs(taker_fee=fee, slippage_frac=slip))
            points.append(CostPoint(fee, slip, round(rep.expectancy_r(), 4),
                                    len(rep.trades)))
    return points


# ============================================================================
# 5) GO / NO-GO REPORT
# ============================================================================
@dataclass(frozen=True)
class GoNoGoThresholds:
    """Honest, conservative bars a strategy must clear before real deployment.
    Tune with eyes open — loosening these to force a GO defeats the point."""
    min_oos_trades: int = 40          # need a real sample, not a handful
    min_oos_expectancy_r: float = 0.05  # positive edge after costs, in R
    max_is_oos_decay_r: float = 0.30   # in-sample must not vastly exceed OOS
    min_cost_survival_expectancy_r: float = 0.0  # still positive at realistic cost
    max_breach_prob: float = 0.30      # P(breach) ceiling under confirmed rules
    min_pass_prob: float = 0.40        # P(pass phase 1) floor


@dataclass
class GoNoGoReport:
    verdict: str                       # "GO" | "NO-GO"
    reasons: List[str]
    checks: Dict[str, bool]
    metrics: Dict[str, float]

    def render(self) -> str:
        lines = [f"VERDICT: {self.verdict}", ""]
        lines.append("Checks:")
        for k, ok in self.checks.items():
            lines.append(f"  [{'PASS' if ok else 'FAIL'}] {k}")
        lines.append("")
        lines.append("Key metrics:")
        for k, v in self.metrics.items():
            lines.append(f"  {k:>28}: {v}")
        if self.reasons:
            lines.append("")
            lines.append("Why:")
            for r in self.reasons:
                lines.append(f"  - {r}")
        return "\n".join(lines)


def go_no_go(
    wf: WalkForwardResult,
    settings: Settings,
    *,
    cost_points: Optional[Sequence[CostPoint]] = None,
    realistic_fee: float = 0.0005,
    realistic_slippage: float = 0.0002,
    thresholds: Optional[GoNoGoThresholds] = None,
    mc_runs: int = 5000,
    mc_seed: int = 42,
) -> GoNoGoReport:
    """Combine OOS edge + overfit decay + cost survival + Monte-Carlo P(pass)/
    P(breach) under the CONFIRMED prop rules into a single verdict.

    Every check must pass for a GO. The default is NO-GO — the strategy must earn
    a GO, and a strategy with no edge will (correctly) fail here.
    """
    t = thresholds or GoNoGoThresholds()
    reasons: List[str] = []
    checks: Dict[str, bool] = {}

    oos_n = wf.oos_trades()
    oos_exp = wf.oos_expectancy_r()
    decay = wf.is_oos_decay()

    # 1) enough OOS trades
    checks["sufficient OOS sample"] = oos_n >= t.min_oos_trades
    if oos_n < t.min_oos_trades:
        reasons.append(f"only {oos_n} OOS trades (< {t.min_oos_trades}); result is noise")

    # 2) positive OOS expectancy
    checks["positive OOS expectancy"] = oos_exp >= t.min_oos_expectancy_r
    if oos_exp < t.min_oos_expectancy_r:
        reasons.append(f"OOS expectancy {oos_exp:.3f}R < {t.min_oos_expectancy_r}R")

    # 3) no severe in-sample -> OOS collapse (overfit guard)
    checks["no overfit collapse"] = decay <= t.max_is_oos_decay_r
    if decay > t.max_is_oos_decay_r:
        reasons.append(f"in-sample beat OOS by {decay:.3f}R (> {t.max_is_oos_decay_r}); overfit")

    # 4) edge survives realistic costs
    cost_ok = True
    cost_exp_at_realistic = None
    if cost_points:
        match = [c for c in cost_points
                 if abs(c.taker_fee - realistic_fee) < 1e-9
                 and abs(c.slippage_frac - realistic_slippage) < 1e-9]
        if match:
            cost_exp_at_realistic = match[0].oos_expectancy_r
            cost_ok = cost_exp_at_realistic >= t.min_cost_survival_expectancy_r
            if not cost_ok:
                reasons.append(
                    f"at realistic cost expectancy {cost_exp_at_realistic:.3f}R "
                    f"< {t.min_cost_survival_expectancy_r}R")
    checks["survives realistic costs"] = cost_ok

    # 5) Monte-Carlo pass/breach under the confirmed prop rules
    p_pass = p_breach = None
    if wf.oos_r_multiples:
        outcome = monte_carlo_challenge(wf.oos_r_multiples, settings,
                                        n_runs=mc_runs, seed=mc_seed)
        p_pass, p_breach = outcome.p_pass(), outcome.p_breach()
        checks["P(breach) acceptable"] = p_breach <= t.max_breach_prob
        checks["P(pass) sufficient"] = p_pass >= t.min_pass_prob
        if p_breach > t.max_breach_prob:
            reasons.append(f"P(breach) {p_breach:.3f} > {t.max_breach_prob}")
        if p_pass < t.min_pass_prob:
            reasons.append(f"P(pass) {p_pass:.3f} < {t.min_pass_prob}")
    else:
        checks["P(breach) acceptable"] = False
        checks["P(pass) sufficient"] = False
        reasons.append("no OOS trades -> cannot simulate the challenge")

    verdict = "GO" if all(checks.values()) else "NO-GO"
    metrics = {
        "oos_trades": oos_n,
        "oos_expectancy_r": round(oos_exp, 4),
        "oos_win_rate": round(wf.oos_win_rate(), 4),
        "in_sample_minus_oos_decay_r": round(decay, 4),
        "expectancy_at_realistic_cost_r": (round(cost_exp_at_realistic, 4)
                                           if cost_exp_at_realistic is not None else "n/a"),
        "p_pass": round(p_pass, 4) if p_pass is not None else "n/a",
        "p_breach": round(p_breach, 4) if p_breach is not None else "n/a",
    }
    if verdict == "NO-GO" and not reasons:
        reasons.append("one or more checks failed")
    return GoNoGoReport(verdict, reasons, checks, metrics)
