"""End-to-end smoke demo (no network, synthetic data).

Run:  python3.11 examples/demo.py   (from the project root)

It fabricates a noisy price series with occasional sweep-and-reverse patterns,
runs the deterministic strategy through the risk layer, prints a backtest
summary, then Monte-Carlo-simulates the prop challenge from the resulting R
distribution. Numbers here are meaningless (synthetic) — this only proves the
pipeline is wired correctly.
"""
import math
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from liq_ai_bot.backtest import Costs, monte_carlo_challenge, run_backtest  # noqa: E402
from liq_ai_bot.config import sandbox_settings  # noqa: E402
from liq_ai_bot.strategy import LiquiditySFPStrategy  # noqa: E402
from liq_ai_bot.types import Bar  # noqa: E402


def synth_bars(n=3000, seed=7):
    rng = random.Random(seed)
    price = 30_000.0
    ts = 1_700_000_000
    bars = []
    for i in range(n):
        drift = math.sin(i / 50.0) * 5
        step = rng.gauss(drift, 40)
        o = price
        c = price + step
        hi = max(o, c) + abs(rng.gauss(0, 25))
        lo = min(o, c) - abs(rng.gauss(0, 25))
        # inject occasional sweep spikes to trigger the SFP logic
        if i % 37 == 0:
            lo -= abs(rng.gauss(60, 20))
            c = max(o, c)
        vol = abs(rng.gauss(1000, 300)) * (2.0 if i % 37 == 0 else 1.0)
        bars.append(Bar(ts + i * 900, o, hi, lo, c, vol))
        price = c
    return bars


def main():
    settings = sandbox_settings()
    bars = synth_bars()
    strat = LiquiditySFPStrategy(symbol="BTC/USDT:USDT")

    report = run_backtest(bars, strat, settings, costs=Costs())
    print("=== Backtest (SYNTHETIC — numbers meaningless) ===")
    for k, v in report.summary().items():
        print(f"  {k:>18}: {v}")

    r_samples = [t.r_multiple for t in report.trades] or [1.5, -1.0, -1.0, 2.0]
    outcome = monte_carlo_challenge(r_samples, settings, n_runs=3000)
    print("\n=== Monte-Carlo prop challenge (placeholder RULES) ===")
    for k, v in outcome.summary().items():
        print(f"  {k:>18}: {v}")
    print("\nReminder: RULES are unconfirmed placeholders; confirm from "
          "MyFundedPerps docs before trusting any pass/breach number.")


if __name__ == "__main__":
    main()
