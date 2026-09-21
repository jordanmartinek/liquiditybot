"""Tests for the M2 LevelEngine — the full port of LiquidityRadar.pine seeding.

Run:  python3 -m unittest discover -s tests -v   (from project root)

Covers previous-period levels (day/week/month), session H/L, swing pivots,
equal highs/lows, order-block tracking, the dealing-range, and the confluence
score. Bars are hand-built with explicit UTC timestamps so period/session
rollovers are deterministic.
"""
import sys
import unittest
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from liq_ai_bot.levels import LevelEngine, LevelParams, OrderBlock  # noqa: E402
from liq_ai_bot.types import Bar, LevelType, PoolSide  # noqa: E402

HOUR = 3600


def ts(y, mo, d, h=0, mi=0):
    return int(datetime(y, mo, d, h, mi, tzinfo=timezone.utc).timestamp())


def bar(t, o, h, l, c, v=1000.0):
    return Bar(ts=t, open=o, high=h, low=l, close=c, volume=v)


class TestPreviousPeriodLevels(unittest.TestCase):
    def test_pdh_pdl_after_day_rollover(self):
        eng = LevelEngine(LevelParams(swing_len=500))  # suppress swing noise
        # Day 1 (2026-01-01): high 105, low 95 across two hourly bars.
        eng.on_bar(bar(ts(2026, 1, 1, 0), 100, 105, 98, 102))
        eng.on_bar(bar(ts(2026, 1, 1, 12), 102, 103, 95, 100))
        # Day 2: a bar in the new day triggers PDH/PDL from day 1.
        eng.on_bar(bar(ts(2026, 1, 2, 0), 100, 101, 99, 100))
        levels = {lv.ltype: lv for lv in eng.levels()}
        self.assertIn(LevelType.PDH, levels)
        self.assertIn(LevelType.PDL, levels)
        self.assertAlmostEqual(levels[LevelType.PDH].price, 105.0)
        self.assertAlmostEqual(levels[LevelType.PDL].price, 95.0)
        self.assertIs(levels[LevelType.PDH].side, PoolSide.BSL)
        self.assertIs(levels[LevelType.PDL].side, PoolSide.SSL)

    def test_pdh_removed_once_swept(self):
        eng = LevelEngine(LevelParams(swing_len=500))
        eng.on_bar(bar(ts(2026, 1, 1, 0), 100, 105, 98, 102))
        eng.on_bar(bar(ts(2026, 1, 2, 0), 100, 101, 99, 100))
        self.assertTrue(any(lv.ltype is LevelType.PDH for lv in eng.levels()))
        # a later bar in day 2 trades above PDH (105) -> swept -> dropped
        eng.on_bar(bar(ts(2026, 1, 2, 1), 100, 106, 99, 104))
        self.assertFalse(any(lv.ltype is LevelType.PDH for lv in eng.levels()))


class TestSessionLevels(unittest.TestCase):
    def test_session_high_low_tracked(self):
        eng = LevelEngine(LevelParams(swing_len=500))
        # Asian session 00:00-08:00 UTC
        eng.on_bar(bar(ts(2026, 3, 2, 1), 100, 110, 95, 105))
        eng.on_bar(bar(ts(2026, 3, 2, 3), 105, 108, 90, 100))
        sess_h = [lv for lv in eng.levels() if lv.ltype is LevelType.SESSION_H]
        sess_l = [lv for lv in eng.levels() if lv.ltype is LevelType.SESSION_L]
        self.assertTrue(sess_h and sess_l)
        self.assertAlmostEqual(max(l.price for l in sess_h), 110.0)
        self.assertAlmostEqual(min(l.price for l in sess_l), 90.0)

    def test_session_reset_on_new_day(self):
        eng = LevelEngine(LevelParams(swing_len=500))
        eng.on_bar(bar(ts(2026, 3, 2, 1), 100, 110, 95, 105))   # day 1 Asia
        # new day, in Asia again: session H/L should reset to this bar's range
        eng.on_bar(bar(ts(2026, 3, 3, 1), 100, 101, 99, 100))
        sess_h = [lv.price for lv in eng.levels() if lv.ltype is LevelType.SESSION_H]
        # yesterday's 110 must not linger as a live session high
        self.assertNotIn(110.0, sess_h)


class TestSwingsAndEqualLevels(unittest.TestCase):
    def _feed_pivot_low(self, eng, base_ts, low_price, n):
        """Feed 2n+1 bars where the middle bar is the lowest -> confirmed pivot low."""
        seq = [102, 101, low_price, 101, 102]  # for n=2
        for i, lo in enumerate(seq):
            eng.on_bar(bar(base_ts + i * HOUR, lo + 5, lo + 6, lo, lo + 4))

    def test_confirmed_swing_low_seeded(self):
        eng = LevelEngine(LevelParams(swing_len=2, htf_factor=9999))
        self._feed_pivot_low(eng, ts(2026, 5, 1, 0), 80, n=2)
        swings = [lv for lv in eng.levels() if lv.ltype is LevelType.SWING_L]
        self.assertTrue(any(abs(lv.price - 80) < 1e-6 for lv in swings))

    def test_equal_lows_detected(self):
        eng = LevelEngine(LevelParams(swing_len=2, htf_factor=9999,
                                      eq_tol_atr=1.0, eq_lookback=6))
        # two pivot lows at nearly the same price within tolerance -> EQ_L
        self._feed_pivot_low(eng, ts(2026, 5, 1, 0), 80.0, n=2)
        self._feed_pivot_low(eng, ts(2026, 5, 1, 10), 80.3, n=2)
        eqs = [lv for lv in eng.levels() if lv.ltype is LevelType.EQ_L]
        self.assertTrue(eqs, "expected an equal-low magnet from two near-equal pivots")


class TestOrderBlocks(unittest.TestCase):
    def test_bullish_order_block_after_displacement(self):
        eng = LevelEngine(LevelParams(swing_len=500))
        # warm up ATR with small bars
        for i in range(20):
            eng.on_bar(bar(ts(2026, 6, 1, 0) + i * HOUR, 100, 100.5, 99.5, 100))
        prev = bar(ts(2026, 6, 1, 21), 100, 100.5, 99.5, 100)
        eng.on_bar(prev)
        # a large bullish displacement bar closing above prev high -> bull OB
        eng.on_bar(bar(ts(2026, 6, 1, 22), 100, 106, 100, 105.5))
        self.assertTrue(any(ob.bull for ob in eng.order_blocks))


class TestConfluenceScore(unittest.TestCase):
    def _warm(self, eng, n=20, price=100.0):
        for i in range(n):
            eng.on_bar(bar(ts(2026, 7, 1, 0) + i * HOUR, price, price + 1, price - 1, price))

    def test_score_in_range_and_stacking_helps(self):
        eng = LevelEngine(LevelParams(swing_len=500))
        self._warm(eng)
        lone = eng.confluence_score(100.0, 99.9, 100.1, members=1)
        stacked = eng.confluence_score(100.0, 99.9, 100.1, members=3)
        self.assertGreaterEqual(lone, 0.0)
        self.assertLessEqual(stacked, 100.0)
        self.assertGreater(stacked, lone)   # more stacked members -> higher score

    def test_round_number_bonus(self):
        eng = LevelEngine(LevelParams(swing_len=500, conf_round_step=100.0))
        self._warm(eng, price=100.0)
        on_round = eng.confluence_score(100.0, 99.99, 100.01, members=1)
        off_round = eng.confluence_score(137.0, 136.99, 137.01, members=1)
        self.assertGreater(on_round, off_round)

    def test_confluence_attached_to_levels(self):
        eng = LevelEngine(LevelParams(swing_len=500))
        eng.on_bar(bar(ts(2026, 8, 1, 0), 100, 105, 98, 102))
        eng.on_bar(bar(ts(2026, 8, 2, 0), 100, 101, 99, 100))
        for lv in eng.levels():
            self.assertGreaterEqual(lv.confluence, 0.0)
            self.assertLessEqual(lv.confluence, 100.0)


class TestOrderBlockBand(unittest.TestCase):
    def test_contains_band(self):
        ob = OrderBlock(top=105.0, bot=100.0, bull=True, birth_ts=0)
        self.assertTrue(ob.contains_band(101.0, 104.0))   # inside
        self.assertTrue(ob.contains_band(99.0, 101.0))    # overlaps lower edge
        self.assertFalse(ob.contains_band(106.0, 110.0))  # entirely above


if __name__ == "__main__":
    unittest.main(verbosity=2)
