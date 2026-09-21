"""M2 real-data backtest runner (DESIGN.md §4).

Wires the data feed -> deterministic strategy -> risk engine -> backtest ->
Monte-Carlo challenge sim, on REAL data instead of the synthetic smoke series.

Sources (see --source):
  * csv       : load OHLCV you downloaded once (fully offline, reproducible).
  * ccxt      : fetch live perp history (needs network + `pip install ccxt`).
  * synthetic : deterministic fabricated data (offline smoke test; numbers meaningless).

Examples
--------
Offline smoke (works anywhere, including this restricted sandbox):
    python3 examples/backtest_real.py --source synthetic --bars 4000

Fetch once where PyPI + network are available, save for offline replay:
    python3 examples/backtest_real.py --source ccxt --symbol BTC/USDT:USDT \
        --timeframe 15m --bars 5000 --save data/btc_15m.csv

Replay that real data offline, forever, deterministically:
    python3 examples/backtest_real.py --source csv --path data/btc_15m.csv --timeframe 15m

IMPORTANT: real numbers here are still only step 1 of the validation plan
(in-sample backtest). A positive expectancy on one window is NOT an edge —
see DESIGN.md §4 for the walk-forward / cost / OOS gates that come next.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from liq_ai_bot.backtest import Costs, monte_carlo_challenge, run_backtest  # noqa: E402
from liq_ai_bot.config import sandbox_settings  # noqa: E402
from liq_ai_bot.datafeed import load_bars, save_csv  # noqa: E402
from liq_ai_bot.strategy import LiquiditySFPStrategy  # noqa: E402


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Backtest the liquidity SFP strategy on real data.")
    p.add_argument("--source", choices=["synthetic", "csv", "ccxt"], default="synthetic")
    p.add_argument("--symbol", default="BTC/USDT:USDT", help="ccxt unified perp symbol")
    p.add_argument("--timeframe", default="15m", help="candle size, e.g. 15m/1h/4h/1d")
    p.add_argument("--path", default=None, help="CSV path (source=csv), or omit")
    p.add_argument("--exchange", default="binanceusdm", help="ccxt exchange id (source=ccxt)")
    p.add_argument("--bars", type=int, default=4000, help="candles to fetch/generate")
    p.add_argument("--save", default=None, help="write fetched/generated bars to this CSV")
    # realistic cost knobs (DESIGN.md §4 gate 4 — never report edge without them)
    p.add_argument("--taker-fee", type=float, default=0.0005, help="per-side taker fee frac")
    p.add_argument("--slippage", type=float, default=0.0002, help="per-side slippage frac")
    p.add_argument("--mc-runs", type=int, default=5000, help="Monte-Carlo challenge runs")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    load_kwargs = {}
    if args.source == "ccxt":
        load_kwargs = {"exchange": args.exchange, "limit": min(args.bars, 1500)}
        if args.bars > 1500:
            # paginate: start far enough back to collect ~args.bars candles
            from liq_ai_bot.datafeed import timeframe_to_seconds
            import time
            span_ms = args.bars * timeframe_to_seconds(args.timeframe) * 1000
            load_kwargs["since_ms"] = int(time.time() * 1000) - span_ms
    elif args.source == "synthetic":
        load_kwargs = {"n": args.bars}

    print(f"Loading bars: source={args.source} symbol={args.symbol} tf={args.timeframe} ...")
    try:
        bars = load_bars(args.source, symbol=args.symbol, timeframe=args.timeframe,
                         path=args.path, **load_kwargs)
    except (RuntimeError, FileNotFoundError, ValueError) as exc:
        print(f"\nERROR loading data: {exc}\n")
        if args.source == "ccxt":
            print("Tip: this sandbox has restricted network. Either run in a networked "
                  "environment, or fetch once + --save a CSV and replay with --source csv.")
        return 2

    if not bars:
        print("No bars loaded; aborting.")
        return 2
    print(f"Loaded {len(bars)} bars "
          f"[{bars[0].ts} .. {bars[-1].ts}]  first close={bars[0].close} last={bars[-1].close}")

    if args.save:
        out = save_csv(bars, args.save)
        print(f"Saved bars -> {out}")

    settings = sandbox_settings(execution_timeframe=args.timeframe)
    strat = LiquiditySFPStrategy(symbol=args.symbol)
    costs = Costs(taker_fee=args.taker_fee, slippage_frac=args.slippage)

    report = run_backtest(bars, strat, settings, costs=costs)
    synthetic = args.source == "synthetic"
    tag = " (SYNTHETIC — numbers meaningless)" if synthetic else ""
    print(f"\n=== Backtest{tag} ===")
    for k, v in report.summary().items():
        print(f"  {k:>18}: {v}")

    r_samples = [t.r_multiple for t in report.trades]
    if not r_samples:
        print("\nNo trades were taken — nothing to Monte-Carlo. "
              "Try more bars, a different symbol/timeframe, or loosen SignalParams.")
        return 0

    outcome = monte_carlo_challenge(r_samples, settings, n_runs=args.mc_runs)
    print(f"\n=== Monte-Carlo prop challenge (placeholder RULES) ===")
    for k, v in outcome.summary().items():
        print(f"  {k:>18}: {v}")

    print("\nReminder: (1) RULES are unconfirmed placeholders — confirm from "
          "MyFundedPerps docs.\n(2) A single in-sample window is NOT proof of edge; "
          "see DESIGN.md §4 for walk-forward + OOS + cost gates.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
