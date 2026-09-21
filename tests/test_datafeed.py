"""Unit tests for the M2 data feed (stdlib unittest, no deps).

Run:  python3 -m unittest discover -s tests -v   (from project root)

Covers timeframe parsing, ms/second timestamp coercion, bar normalization
(sort + dedupe), OHLC validation warnings, the CSV save/load round-trip, and the
graceful failure of the ccxt path when the optional dep / network is absent.
"""
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from liq_ai_bot.datafeed import (  # noqa: E402
    fetch_ccxt, load_bars, load_csv, normalize_bars, save_csv,
    synthetic_bars, timeframe_to_seconds, validate_bars,
)
from liq_ai_bot.types import Bar  # noqa: E402


class TestTimeframe(unittest.TestCase):
    def test_known_units(self):
        self.assertEqual(timeframe_to_seconds("15m"), 900)
        self.assertEqual(timeframe_to_seconds("1h"), 3600)
        self.assertEqual(timeframe_to_seconds("4h"), 14_400)
        self.assertEqual(timeframe_to_seconds("1d"), 86_400)
        self.assertEqual(timeframe_to_seconds("1w"), 604_800)

    def test_invalid_raises(self):
        for bad in ("", "m", "15", "0m", "-1h", "15x", "abc"):
            with self.assertRaises(ValueError, msg=f"{bad!r} should be invalid"):
                timeframe_to_seconds(bad)


class TestNormalize(unittest.TestCase):
    def test_ms_and_second_coercion(self):
        # ms timestamp (>= 1e12) coerced to seconds; second timestamp left as-is
        bars = normalize_bars([
            [1_700_000_000_000, 1, 2, 0.5, 1.5, 10],   # ms
            [1_700_000_900, 1, 2, 0.5, 1.5, 10],        # seconds
        ])
        self.assertEqual(bars[0].ts, 1_700_000_000)
        self.assertEqual(bars[1].ts, 1_700_000_900)

    def test_sorted_and_deduped(self):
        bars = normalize_bars([
            [30, 1, 2, 0, 1, 5],
            [10, 1, 2, 0, 1, 5],
            [20, 1, 2, 0, 1, 5],
            [10, 9, 9, 9, 9, 9],   # duplicate ts -> last wins
        ])
        self.assertEqual([b.ts for b in bars], [10, 20, 30])
        self.assertEqual(bars[0].open, 9)  # the later duplicate replaced the earlier

    def test_missing_volume_defaults_zero(self):
        bars = normalize_bars([[10, 1, 2, 0, 1]])  # no volume column
        self.assertEqual(bars[0].volume, 0.0)


class TestValidate(unittest.TestCase):
    def test_clean_series_has_no_warnings(self):
        bars = synthetic_bars(n=100, timeframe="15m")
        self.assertEqual(validate_bars(bars, expected_tf="15m"), [])

    def test_flags_bad_ohlc(self):
        bad = [Bar(0, open=1, high=0.5, low=0.9, close=1, volume=1)]  # high<low, etc.
        warns = validate_bars(bad)
        self.assertTrue(any("high" in w for w in warns))

    def test_flags_time_gap(self):
        bars = [Bar(0, 1, 2, 0, 1, 1), Bar(1800, 1, 2, 0, 1, 1)]  # 30m gap on 15m tf
        warns = validate_bars(bars, expected_tf="15m")
        self.assertTrue(any("missing candle" in w for w in warns))

    def test_flags_non_monotonic_time(self):
        bars = [Bar(100, 1, 2, 0, 1, 1), Bar(100, 1, 2, 0, 1, 1)]
        # normalize would dedupe, but validate should catch raw non-increasing ts
        warns = validate_bars(bars)
        self.assertTrue(any("not strictly after" in w for w in warns))

    def test_empty_series(self):
        self.assertEqual(validate_bars([]), ["empty bar series"])


class TestCsvRoundTrip(unittest.TestCase):
    def test_save_then_load_is_lossless(self):
        bars = synthetic_bars(n=200, timeframe="1h", seed=3)
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "ohlcv.csv"
            save_csv(bars, p)
            loaded = load_csv(p)
        self.assertEqual(len(loaded), len(bars))
        for a, b in zip(bars, loaded):
            self.assertEqual(a.ts, b.ts)
            self.assertAlmostEqual(a.open, b.open, places=6)
            self.assertAlmostEqual(a.high, b.high, places=6)
            self.assertAlmostEqual(a.low, b.low, places=6)
            self.assertAlmostEqual(a.close, b.close, places=6)
            self.assertAlmostEqual(a.volume, b.volume, places=6)

    def test_load_headerless_csv(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "raw.csv"
            p.write_text("10,1,2,0.5,1.5,100\n20,1.5,2.5,1,2,120\n")
            loaded = load_csv(p)
        self.assertEqual([b.ts for b in loaded], [10, 20])

    def test_load_missing_file_raises(self):
        with self.assertRaises(FileNotFoundError):
            load_csv("/nonexistent/does_not_exist.csv")


class TestSynthetic(unittest.TestCase):
    def test_deterministic_with_seed(self):
        a = synthetic_bars(n=50, seed=42)
        b = synthetic_bars(n=50, seed=42)
        self.assertEqual([x.close for x in a], [x.close for x in b])

    def test_timeframe_spacing(self):
        bars = synthetic_bars(n=10, timeframe="4h")
        self.assertEqual(bars[1].ts - bars[0].ts, timeframe_to_seconds("4h"))


class TestCcxtGracefulFailure(unittest.TestCase):
    def test_ccxt_path_raises_actionable_error_when_unavailable(self):
        """In this restricted sandbox ccxt/network is unavailable; the loader must
        raise a clear RuntimeError (not crash with an opaque traceback)."""
        try:
            import ccxt  # noqa: F401
            has_ccxt = True
        except ImportError:
            has_ccxt = False
        if has_ccxt:
            self.skipTest("ccxt is installed; skipping the missing-dep failure test")
        with self.assertRaises(RuntimeError) as ctx:
            fetch_ccxt(symbol="BTC/USDT:USDT", timeframe="15m", limit=10)
        self.assertIn("ccxt", str(ctx.exception).lower())

    def test_load_bars_rejects_unknown_source(self):
        with self.assertRaises(ValueError):
            load_bars("not-a-source")


if __name__ == "__main__":
    unittest.main(verbosity=2)
