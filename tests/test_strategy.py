"""Unit tests for the deterministic strategy core (stdlib unittest, no deps).

Run:  python3 -m unittest discover -s tests -v   (from project root)

Covers the three pieces of strategy.py that the risk-engine tests don't touch:
  * Rolling      — ATR / rvol / confirmed swing pivots
  * compute_dol  — the draw-on-liquidity gravity model (bias + conviction)
  * LiquiditySFPStrategy.on_bar — the sweep+reversal (SFP) entry trigger and its gates

The strategy never sizes or touches money, so these assert *setup shape* and the
gating logic (volume, DOL alignment, confluence), not P&L.
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from liq_ai_bot.levels import LevelParams  # noqa: E402
from liq_ai_bot.strategy import (  # noqa: E402
    DOLParams, Rolling, SignalParams, compute_dol, LiquiditySFPStrategy,
)
from liq_ai_bot.types import (  # noqa: E402
    Bar, Bias, Level, LevelType, PoolSide, Side,
)


def bar(ts, o, h, l, c, v=1000.0):
    return Bar(ts=ts, open=o, high=h, low=l, close=c, volume=v)


class TestRolling(unittest.TestCase):
    def test_atr_and_avg_vol(self):
        r = Rolling(atr_len=3, vol_len=3, swing_len=2)
        # first bar: TR = high-low = 10
        r.update(bar(0, 100, 110, 100, 105, v=500))
        self.assertAlmostEqual(r.atr, 10.0, places=6)
        self.assertAlmostEqual(r.avg_vol, 500.0, places=6)
        # second bar: prev_close=105, TR = max(20, |120-105|, |100-105|) = 20
        r.update(bar(1, 105, 120, 100, 115, v=1500))
        self.assertAlmostEqual(r.atr, (10 + 20) / 2, places=6)
        self.assertAlmostEqual(r.avg_vol, 1000.0, places=6)

    def test_rvol(self):
        r = Rolling(vol_len=2)
        r.update(bar(0, 100, 101, 99, 100, v=1000))
        r.update(bar(1, 100, 101, 99, 100, v=1000))
        # a bar with 2x the average volume -> rvol ~ 2 (avg includes recent bars)
        b = bar(2, 100, 101, 99, 100, v=2000)
        self.assertGreater(r.rvol(b), 1.0)

    def test_confirmed_swing_high(self):
        r = Rolling(swing_len=2)
        # need 2*n+1 = 5 bars; make the middle bar the highest -> confirmed high
        highs = [100, 105, 130, 106, 101]
        for i, h in enumerate(highs):
            r.update(bar(i, h - 5, h, h - 10, h - 2))
        sw = r.confirmed_swing()
        self.assertIsNotNone(sw)
        kind, pivot = sw
        self.assertEqual(kind, "high")
        self.assertEqual(pivot.high, 130)

    def test_confirmed_swing_low(self):
        r = Rolling(swing_len=2)
        lows = [50, 45, 20, 44, 49]
        for i, lo in enumerate(lows):
            r.update(bar(i, lo + 10, lo + 12, lo, lo + 8))
        sw = r.confirmed_swing()
        self.assertIsNotNone(sw)
        kind, pivot = sw
        self.assertEqual(kind, "low")
        self.assertEqual(pivot.low, 20)

    def test_no_swing_when_insufficient_bars(self):
        r = Rolling(swing_len=3)
        for i in range(4):  # fewer than 2*3+1 = 7
            r.update(bar(i, 100, 101, 99, 100))
        self.assertIsNone(r.confirmed_swing())


class TestComputeDOL(unittest.TestCase):
    def test_neutral_when_no_levels(self):
        dol = compute_dol([], price=100.0, atr=1.0)
        self.assertEqual(dol.bias, Bias.NEUTRAL)
        self.assertEqual(dol.conviction, 0.0)

    def test_zero_atr_is_neutral(self):
        lv = Level(110.0, LevelType.SWING_H, PoolSide.BSL, birth_ts=0, confluence=50)
        dol = compute_dol([lv], price=100.0, atr=0.0)
        self.assertEqual(dol.bias, Bias.NEUTRAL)

    def test_bias_up_when_pool_above(self):
        # one un-swept buy-side pool above price -> pull is upward -> bias UP
        lv = Level(105.0, LevelType.SWING_H, PoolSide.BSL, birth_ts=0, confluence=50)
        dol = compute_dol([lv], price=100.0, atr=2.0)
        self.assertEqual(dol.bias, Bias.UP)
        self.assertGreater(dol.conviction, 0.0)
        self.assertEqual(dol.magnet_price, 105.0)

    def test_bias_down_when_pool_below(self):
        lv = Level(95.0, LevelType.SWING_L, PoolSide.SSL, birth_ts=0, confluence=50)
        dol = compute_dol([lv], price=100.0, atr=2.0)
        self.assertEqual(dol.bias, Bias.DOWN)

    def test_swept_levels_exert_no_pull(self):
        lv = Level(105.0, LevelType.SWING_H, PoolSide.BSL, birth_ts=0, confluence=50)
        lv.swept = True
        dol = compute_dol([lv], price=100.0, atr=2.0)
        self.assertEqual(dol.bias, Bias.NEUTRAL)

    def test_equal_highs_are_stronger_magnets(self):
        """An EQ_H above should out-pull an ordinary SWING_L the same distance below,
        because eq_weight (2.4) exceeds the type weight (1.0) of a normal level."""
        eq_high = Level(105.0, LevelType.EQ_H, PoolSide.BSL, birth_ts=0, confluence=50)
        swing_low = Level(95.0, LevelType.SWING_L, PoolSide.SSL, birth_ts=0, confluence=50)
        dol = compute_dol([eq_high, swing_low], price=100.0, atr=2.0)
        self.assertEqual(dol.bias, Bias.UP)

    def test_conviction_floor_yields_neutral(self):
        """Balanced opposing pools -> conviction below floor -> NEUTRAL bias."""
        up = Level(105.0, LevelType.SWING_H, PoolSide.BSL, birth_ts=0, confluence=50)
        down = Level(95.0, LevelType.SWING_L, PoolSide.SSL, birth_ts=0, confluence=50)
        p = DOLParams(conviction_floor=0.9)  # force a high bar so symmetric -> neutral
        dol = compute_dol([up, down], price=100.0, atr=2.0, params=p)
        self.assertEqual(dol.bias, Bias.NEUTRAL)


class TestSFPTrigger(unittest.TestCase):
    """Drive the full LiquiditySFPStrategy.on_bar path with hand-built bars.

    The M2 engine auto-seeds a real level map from the bar stream, so to test the
    SFP *trigger* in isolation we suppress auto-seeding (a LevelEngine with a very
    long swing pivot so no pivots confirm on the short warmup, and sessions that
    won't produce competing pools within the flat window) and drive the trigger off
    a manually-seeded SSL/BSL pair. Level seeding itself is covered by
    test_levels.py.
    """

    def _make_strategy(self, seed_target=True, low_conf=False):
        params = SignalParams(rvol_min=1.2, confluence_min=40.0, conviction_min=0.2,
                              stop_pad_atr=0.25, fallback_r=2.0)
        # auto_seed=False => run purely off manual levels (isolate the SFP trigger)
        strat = LiquiditySFPStrategy("TEST/USDT:USDT", params=params, auto_seed=False)
        conf = 10 if low_conf else 60
        strat.add_level(Level(99.0, LevelType.SWING_L, PoolSide.SSL, birth_ts=0, confluence=conf))
        if seed_target:
            # BSL magnet above -> DOL points up and gives the SFP a target
            strat.add_level(Level(110.0, LevelType.SWING_H, PoolSide.BSL, birth_ts=0, confluence=60))
        return strat

    def _warmup(self, strat, n=20, price=100.0):
        # flat, low-volume bars: ATR well-defined; the 99.0 SSL stays un-swept
        for i in range(n):
            strat.on_bar(bar(i, price, price + 0.5, price - 0.5, price, v=1000))

    def test_long_sfp_fires_on_sweep_and_reclaim(self):
        strat = self._make_strategy()
        self._warmup(strat, n=20, price=100.0)
        # Sweep bar: wick pierces below the 99.0 SSL, closes back above it, high volume.
        setup = strat.on_bar(bar(100, 100.0, 100.5, 98.0, 100.2, v=5000))
        self.assertIsNotNone(setup, "expected a LONG SFP setup on sweep+reclaim")
        self.assertEqual(setup.side, Side.LONG)
        self.assertLess(setup.stop, 98.0)          # stop below the sweep wick
        # target is the opposing BSL pool above (the DOL magnet)
        self.assertEqual(setup.target, 110.0)
        self.assertIn("rvol", setup.features)
        self.assertIn("dol_conviction", setup.features)
        self.assertGreater(setup.entry, setup.stop)

    def test_no_trigger_without_opposing_pool_for_dol(self):
        """With no BSL pool above, DOL cannot point up, so the DOL-agreement gate
        blocks the long (the target's magnet and the bias share the same pool)."""
        strat = self._make_strategy(seed_target=False)
        self._warmup(strat, n=20, price=100.0)
        setup = strat.on_bar(bar(100, 100.0, 100.5, 98.0, 100.2, v=5000))
        self.assertIsNone(setup)

    def test_no_trigger_without_volume(self):
        strat = self._make_strategy()
        self._warmup(strat, n=20, price=100.0)
        setup = strat.on_bar(bar(100, 100.0, 100.5, 98.0, 100.2, v=1000))
        self.assertIsNone(setup)

    def test_no_trigger_when_close_stays_below_swept_level(self):
        strat = self._make_strategy()
        self._warmup(strat, n=20, price=100.0)
        # wick pierces 99 but the bar CLOSES below it -> not a reclaim -> no SFP
        setup = strat.on_bar(bar(100, 99.5, 99.6, 98.0, 98.5, v=5000))
        self.assertIsNone(setup)

    def test_no_trigger_when_dol_opposes(self):
        """A dominant opposing pool keeps DOL pointing DOWN, filtering out a bullish
        sweep+reclaim (DOL-agreement gate)."""
        params = SignalParams(rvol_min=1.2, confluence_min=40.0, conviction_min=0.2)
        strat = LiquiditySFPStrategy("TEST/USDT:USDT", params=params, auto_seed=False)
        strat.add_level(Level(99.0, LevelType.SWING_L, PoolSide.SSL, birth_ts=0, confluence=60))
        # a strong, near EQ_L just below price dominates the pull -> DOL DOWN
        strat.add_level(Level(98.6, LevelType.EQ_L, PoolSide.SSL, birth_ts=0, confluence=95))
        self._warmup(strat, n=20, price=100.0)
        # sweep the 99.0 only (keep the 98.6 EQ_L intact so it still pulls down)
        setup = strat.on_bar(bar(100, 100.0, 100.4, 98.7, 100.2, v=5000))
        self.assertIsNone(setup)

    def test_no_trigger_when_confluence_below_gate(self):
        strat = self._make_strategy(low_conf=True)  # swept level confluence=10 < gate 40
        self._warmup(strat, n=20, price=100.0)
        setup = strat.on_bar(bar(100, 100.0, 100.5, 98.0, 100.2, v=5000))
        self.assertIsNone(setup)

    def test_r_multiple_sign_is_consistent(self):
        """A LONG setup: price above entry => positive R, below entry => negative R."""
        strat = self._make_strategy()
        self._warmup(strat, n=20, price=100.0)
        setup = strat.on_bar(bar(100, 100.0, 100.5, 98.0, 100.2, v=5000))
        self.assertIsNotNone(setup)
        self.assertGreater(setup.r_multiple_to(setup.target), 0.0)
        self.assertLess(setup.r_multiple_to(setup.stop), 0.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
