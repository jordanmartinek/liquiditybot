"""M3/M4 walk-forward validation runner (DESIGN.md §4).

Ties it all together: load bars -> rolling walk-forward (tune on train, evaluate
out-of-sample) -> cost-sensitivity sweep -> consolidated GO / NO-GO verdict under
the CONFIRMED prop rules.

Sources (mirror examples/backtest_real.py):
  * csv       : real OHLCV you downloaded once (offline, reproducible) -- USE THIS.
  * ccxt      : fetch live perp history (needs network + `pip install ccxt`).
  * synthetic : deterministic fabricated data (smoke test; numbers meaningless).

Examples
--------
Offline smoke (works anywhere):
    python3 examples/walkforward.py --source synthetic --bars 4000 \
        --train 800 --test 300 --embargo 20

Real validation (in a networked env), fetch once then validate:
    python3 examples/backtest_real.py --source ccxt --symbol BTC/USDT:USDT \
        --timeframe 15m --bars 20000 --save data/btc_15m.csv
    python3 examples/walkforward.py --source csv --path data/btc_15m.csv \
        --timeframe 15m --train 3000 --test 1000 --embargo 50

Honesty: a NO-GO is the expected, correct output until the strategy shows a real
edge on real data. The harness is built to say NO plainly.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from liq_ai_bot.backtest import Costs  # noqa: E402
from liq_ai_bot.config import sandbox_settings  # noqa: E402
from liq_ai_bot.datafeed import load_bars  # noqa: E402
from liq_ai_bot.levels import LevelParams  # noqa: E402
from liq_ai_bot.walkforward import (  # noqa: E402
    ParamGrid, cost_sensitivity, go_no_go, run_walk_forward,
)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Walk-forward validate the liquidity strategy.")
    p.add_argument("--source", choices=["synthetic", "csv", "ccxt"], default="synthetic")
    p.add_argument("--symbol", default="BTC/USDT:USDT")
    p.add_argument("--timeframe", default="15m")
    p.add_argument("--path", default=None, help="CSV path (source=csv)")
    p.add_argument("--exchange", default="binanceusdm")
    p.add_argument("--bars", type=int, default=4000)
    p.add_argument("--train", type=int, default=800, help="train window (bars)")
    p.add_argument("--test", type=int, default=300, help="test window (bars)")
    p.add_argument("--embargo", type=int, default=20, help="gap between train and test")
    p.add_argument("--anchored", action="store_true", help="expanding train window")
    p.add_argument("--swing-len", type=int, default=10, help="LevelEngine swing pivot length")
    p.add_argument("--min-trades", type=int, default=15, help="in-sample trade floor for tuning")
    p.add_argument("--taker-fee", type=float, default=0.0005)
    p.add_argument("--slippage", type=float, default=0.0002)
    p.add_argument("--mc-runs", type=int, default=5000)
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    load_kwargs = {}
    if args.source == "ccxt":
        load_kwargs = {"exchange": args.exchange, "limit": min(args.bars, 1500)}
        if args.bars > 1500:
            from liq_ai_bot.datafeed import timeframe_to_seconds
            import time
            span = args.bars * timeframe_to_seconds(args.timeframe) * 1000
            load_kwargs["since_ms"] = int(time.time() * 1000) - span
    elif args.source == "synthetic":
        load_kwargs = {"n": args.bars}

    print(f"Loading bars: source={args.source} tf={args.timeframe} ...")
    try:
        bars = load_bars(args.source, symbol=args.symbol, timeframe=args.timeframe,
                         path=args.path, **load_kwargs)
    except (RuntimeError, FileNotFoundError, ValueError) as exc:
        print(f"\nERROR loading data: {exc}")
        if args.source == "ccxt":
            print("Tip: restricted network here; fetch+save a CSV then use --source csv.")
        return 2

    need = args.train + args.embargo + args.test
    if len(bars) < need:
        print(f"Not enough bars: have {len(bars)}, need >= {need} "
              f"(train+embargo+test). Increase --bars or shrink windows.")
        return 2
    print(f"Loaded {len(bars)} bars.")

    settings = sandbox_settings(execution_timeframe=args.timeframe)
    grid = ParamGrid()
    level_params = LevelParams(swing_len=args.swing_len)
    costs = Costs(taker_fee=args.taker_fee, slippage_frac=args.slippage)

    print(f"\nRunning walk-forward: train={args.train} test={args.test} "
          f"embargo={args.embargo} anchored={args.anchored} ...")
    wf = run_walk_forward(
        bars, train_size=args.train, test_size=args.test, settings=settings,
        symbol=args.symbol, grid=grid, costs=costs, level_params=level_params,
        embargo=args.embargo, anchored=args.anchored, min_trades=args.min_trades,
    )

    print("\n=== Walk-forward folds (out-of-sample) ===")
    for f in wf.folds:
        print(f"  fold {f.index}: IS exp {f.in_sample_expectancy_r:+.3f}R "
              f"({f.in_sample_trades} tr) -> OOS exp {f.oos_expectancy_r:+.3f}R "
              f"({f.oos_trades} tr) | params rvol>={f.params.rvol_min} "
              f"conf>={f.params.confluence_min} conv>={f.params.conviction_min}")
    print("\n=== Pooled OOS summary ===")
    for k, v in wf.summary().items():
        print(f"  {k:>28}: {v}")

    # Cost sensitivity on a CONTINUOUS held-out tail (the last test window), since
    # the LevelEngine needs continuous bars -- concatenated OOS slices would break
    # its warmup. Use the most recent fold's tuned params.
    cost_points = None
    if wf.folds:
        tail = bars[-args.test:]
        best_params = wf.folds[-1].params
        cost_points = cost_sensitivity(tail, best_params, settings,
                                       symbol=args.symbol, level_params=level_params)
        print("\n=== Cost sensitivity (continuous held-out tail) ===")
        print("   fee    slip   OOS_expR  trades")
        for c in cost_points:
            print(f"  {c.taker_fee:.4f} {c.slippage_frac:.4f}  "
                  f"{c.oos_expectancy_r:+.3f}   {c.oos_trades}")

    report = go_no_go(wf, settings, cost_points=cost_points,
                      realistic_fee=args.taker_fee, realistic_slippage=args.slippage,
                      mc_runs=args.mc_runs)
    print("\n" + "=" * 60)
    print(report.render())
    print("=" * 60)
    if args.source == "synthetic":
        print("\nNOTE: synthetic data -> these numbers are meaningless. Run on real "
              "perp history (--source csv/ccxt) for a verdict that means anything.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
