"""Tests for the M3/M4 walk-forward validation harness (stdlib unittest, no deps).

Run:  python3 -m unittest discover -s tests -v   (from project root)

Covers the splitter (no leakage, embargo, anchored), the tuner, the walk-forward
driver, cost sensitivity, and the GO/NO-GO verdict logic (both branches driven by
synthetic OOS R-distributions so the verdict math is tested independently of
strategy behavior).
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from liq_ai_bot.config import sandbox_settings  # noqa: E402
from liq_ai_bot.datafeed import synthetic_bars  # noqa: E402
from liq_ai_bot.levels import LevelParams  # noqa: E402
from liq_ai_bot.strategy import SignalParams  # noqa: E402
from liq_ai_bot.walkforward import (  # noqa: E402
    Fold, FoldResult, GoNoGoThresholds, ParamGrid, WalkForwardResult,
    cost_sensitivity, go_no_go, run_walk_forward, tune_params, walk_forward_splits,
)


BARS = synthetic_bars(n=2400, timeframe="1h")


class TestSplitter(unittest.TestCase):
    def test_window_sizes_and_ordering(self):
        folds = walk_forward_splits(BARS, train_size=600, test_size=200)
        self.assertTrue(folds)
        for f in folds:
            self.assertEqual(len(f.train), 600)
            self.assertEqual(len(f.test), 200)
        # test windows advance chronologically
        starts = [f.test[0].ts for f in folds]
        self.assertEqual(starts, sorted(starts))

    def test_no_leakage_train_before_test(self):
        folds = walk_forward_splits(BARS, train_size=500, test_size=200, embargo=10)
        for f in folds:
            self.assertLess(f.train[-1].ts, f.test[0].ts)  # train strictly before test

    def test_embargo_gap_enforced(self):
        tf = BARS[1].ts - BARS[0].ts   # bar spacing (seconds)
        embargo = 15
        folds = walk_forward_splits(BARS, train_size=500, test_size=200, embargo=embargo)
        for f in folds:
            gap_bars = round((f.test[0].ts - f.train[-1].ts) / tf) - 1
            self.assertGreaterEqual(gap_bars, embargo)

    def test_anchored_train_grows(self):
        folds = walk_forward_splits(BARS, train_size=400, test_size=200,
                                    step=200, anchored=True)
        self.assertGreater(len(folds), 1)
        # every anchored train starts at bar 0 and later folds have >= bars
        self.assertTrue(all(f.train[0].ts == BARS[0].ts for f in folds))
        self.assertGreaterEqual(len(folds[-1].train), len(folds[0].train))

    def test_rolling_train_moves(self):
        folds = walk_forward_splits(BARS, train_size=400, test_size=200, step=200)
        self.assertGreater(folds[1].train[0].ts, folds[0].train[0].ts)

    def test_invalid_sizes_raise(self):
        with self.assertRaises(ValueError):
            walk_forward_splits(BARS, train_size=0, test_size=100)

    def test_no_folds_when_series_too_short(self):
        short = BARS[:100]
        self.assertEqual(walk_forward_splits(short, train_size=90, test_size=50), [])


class TestParamGrid(unittest.TestCase):
    def test_combinations_cover_product(self):
        g = ParamGrid(rvol_min=(1.2, 1.5), confluence_min=(30.0, 45.0),
                      conviction_min=(0.2,), stop_pad_atr=(0.25,), fallback_r=(2.0,))
        combos = g.combinations()
        self.assertEqual(len(combos), 4)             # 2 * 2 * 1 * 1 * 1
        self.assertTrue(all(isinstance(c, SignalParams) for c in combos))


class TestTuner(unittest.TestCase):
    def test_returns_a_param_set_from_the_grid(self):
        settings = sandbox_settings()
        grid = ParamGrid(rvol_min=(1.2, 1.5), confluence_min=(30.0, 45.0),
                         conviction_min=(0.2,))
        res = tune_params(BARS[:800], settings, grid=grid,
                          level_params=LevelParams(swing_len=5), min_trades=3)
        self.assertIn(res.best_params.rvol_min, (1.2, 1.5))
        self.assertEqual(res.evaluated, 4)


class TestWalkForwardDriver(unittest.TestCase):
    def test_runs_and_pools_oos(self):
        settings = sandbox_settings()
        grid = ParamGrid(rvol_min=(1.2,), confluence_min=(30.0,), conviction_min=(0.2,))
        wf = run_walk_forward(
            BARS, train_size=600, test_size=200, settings=settings, grid=grid,
            level_params=LevelParams(swing_len=5), embargo=10, min_trades=3,
        )
        self.assertTrue(wf.folds)
        # pooled OOS equals the concatenation of per-fold OOS trades
        total = sum(f.oos_trades for f in wf.folds)
        self.assertEqual(wf.oos_trades(), total)
        s = wf.summary()
        self.assertIn("oos_expectancy_r", s)
        self.assertIn("is_minus_oos_decay", s)


class TestCostSensitivity(unittest.TestCase):
    def test_grid_shape_and_zero_cost_ge_high_cost(self):
        settings = sandbox_settings()
        pts = cost_sensitivity(
            BARS[:800], SignalParams(rvol_min=1.2, confluence_min=30.0, conviction_min=0.2),
            settings, level_params=LevelParams(swing_len=5),
            fee_grid=(0.0, 0.001), slippage_grid=(0.0, 0.001),
        )
        self.assertEqual(len(pts), 4)
        zero = next(p for p in pts if p.taker_fee == 0.0 and p.slippage_frac == 0.0)
        high = next(p for p in pts if p.taker_fee == 0.001 and p.slippage_frac == 0.001)
        # more cost never improves expectancy (given identical trades)
        self.assertGreaterEqual(zero.oos_expectancy_r, high.oos_expectancy_r)


def _wf_from_r(r_multiples):
    """Build a WalkForwardResult directly from a synthetic OOS R-distribution so the
    verdict logic can be tested without running the strategy."""
    fold = FoldResult(
        index=0, train_span=(0, 1), test_span=(2, 3),
        params=SignalParams(), in_sample_expectancy_r=0.1,
        in_sample_trades=50, oos_expectancy_r=(sum(r_multiples) / len(r_multiples)),
        oos_trades=len(r_multiples), oos_r_multiples=list(r_multiples),
    )
    return WalkForwardResult([fold], list(r_multiples))


class TestGoNoGoVerdict(unittest.TestCase):
    def test_no_go_when_too_few_trades(self):
        wf = _wf_from_r([1.0, -1.0, 2.0])   # only 3 trades
        rep = go_no_go(wf, sandbox_settings(), mc_runs=500)
        self.assertEqual(rep.verdict, "NO-GO")
        self.assertFalse(rep.checks["sufficient OOS sample"])

    def test_no_go_when_negative_expectancy(self):
        wf = _wf_from_r([-1.0] * 30 + [0.5] * 30)   # net negative, 60 trades
        rep = go_no_go(wf, sandbox_settings(), mc_runs=500)
        self.assertEqual(rep.verdict, "NO-GO")
        self.assertFalse(rep.checks["positive OOS expectancy"])

    def test_go_when_all_checks_pass(self):
        # a strong, realistic edge: ~55% win at +2R / -1R over a large sample
        r = ([2.0] * 55 + [-1.0] * 45) * 3    # 300 trades, +0.65R expectancy
        wf = _wf_from_r(r)
        thresholds = GoNoGoThresholds()  # defaults
        cost_pts = None
        rep = go_no_go(wf, sandbox_settings(), cost_points=cost_pts,
                       thresholds=thresholds, mc_runs=3000)
        # with no cost_points the cost check defaults to pass; all others should pass
        self.assertTrue(rep.checks["sufficient OOS sample"])
        self.assertTrue(rep.checks["positive OOS expectancy"])
        self.assertTrue(rep.checks["P(breach) acceptable"])
        self.assertTrue(rep.checks["P(pass) sufficient"])
        self.assertEqual(rep.verdict, "GO")

    def test_cost_failure_forces_no_go(self):
        from liq_ai_bot.walkforward import CostPoint
        r = ([2.0] * 55 + [-1.0] * 45) * 3
        wf = _wf_from_r(r)
        # realistic-cost expectancy is negative -> must fail the cost check
        cps = [CostPoint(taker_fee=0.0005, slippage_frac=0.0002,
                         oos_expectancy_r=-0.2, oos_trades=300)]
        rep = go_no_go(wf, sandbox_settings(), cost_points=cps,
                       realistic_fee=0.0005, realistic_slippage=0.0002, mc_runs=1000)
        self.assertFalse(rep.checks["survives realistic costs"])
        self.assertEqual(rep.verdict, "NO-GO")

    def test_report_renders(self):
        wf = _wf_from_r([1.0, -1.0] * 30)
        rep = go_no_go(wf, sandbox_settings(), mc_runs=500)
        text = rep.render()
        self.assertIn("VERDICT:", text)
        self.assertIn("Checks:", text)


if __name__ == "__main__":
    unittest.main(verbosity=2)
