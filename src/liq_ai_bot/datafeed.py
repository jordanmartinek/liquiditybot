"""M2 data feed — turn real historical perp OHLCV into `Bar` objects (DESIGN.md §3, §4).

This is the seam between the deterministic strategy/backtest (which are stdlib-only
and work offline) and the outside world of real market data. It provides three
interchangeable sources behind one shape — every loader returns ``List[Bar]``
sorted by ascending timestamp, deduped, so `run_backtest` can consume them
directly:

  1. ``load_csv``       — read OHLCV from a local CSV (fully offline, deterministic,
                          the recommended path for reproducible backtests).
  2. ``fetch_ccxt``     — pull real perp history from any ccxt exchange, paginating
                          back over as many candles as requested. ``ccxt`` is
                          imported LAZILY so this module still imports with only the
                          standard library (M1 stays dependency-free). Requires
                          network + ``pip install ccxt``.
  3. ``synthetic_bars`` — deterministic fabricated series for smoke tests / CI.

Plus ``save_csv`` so you can fetch once (in a networked env) and replay offline
forever.

Timeframe helpers map strings like ``"15m"`` / ``"1h"`` / ``"4h"`` / ``"1d"`` to
seconds, matching the ``execution_timeframe`` in ``Settings``.

Pure standard library at import time. ``ccxt`` is only touched inside ``fetch_ccxt``.
"""
from __future__ import annotations

import csv
import math
import random
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Union

from .types import Bar

# ---- timeframe parsing ------------------------------------------------------
_TF_UNIT_SECONDS = {"s": 1, "m": 60, "h": 3600, "d": 86_400, "w": 604_800}


def timeframe_to_seconds(tf: str) -> int:
    """Parse a ccxt-style timeframe string (e.g. '15m', '1h', '4h', '1d') to seconds."""
    tf = tf.strip().lower()
    if not tf or not tf[-1].isalpha() or not tf[:-1].isdigit():
        raise ValueError(f"invalid timeframe {tf!r} (expected forms like '15m','1h','1d')")
    qty = int(tf[:-1])
    unit = tf[-1]
    if unit not in _TF_UNIT_SECONDS or qty <= 0:
        raise ValueError(f"invalid timeframe {tf!r}")
    return qty * _TF_UNIT_SECONDS[unit]


# ---- normalization ----------------------------------------------------------
def _coerce_ts_to_epoch_seconds(ts: Union[int, float, str]) -> int:
    """Accept epoch seconds or milliseconds (ccxt uses ms) and normalize to seconds.

    Heuristic: values >= 1e12 are treated as milliseconds. This safely covers all
    real trading dates from ~2001 onward without ambiguity.
    """
    v = float(ts)
    if v >= 1e12:  # milliseconds
        v /= 1000.0
    return int(v)


def normalize_bars(rows: Iterable[Sequence], *, ts_in_ms: Optional[bool] = None) -> List[Bar]:
    """Turn raw ``[ts, open, high, low, close, volume]`` rows into a clean list of
    ``Bar``: coerced types, sorted ascending, de-duplicated by timestamp (last wins).

    ``ts_in_ms`` forces the timestamp unit; when None it is auto-detected per row.
    """
    by_ts: Dict[int, Bar] = {}
    for row in rows:
        if row is None:
            continue
        ts_raw, o, h, l, c, *rest = row
        vol = rest[0] if rest else 0.0
        if ts_in_ms is True:
            ts = int(float(ts_raw) / 1000.0)
        elif ts_in_ms is False:
            ts = int(float(ts_raw))
        else:
            ts = _coerce_ts_to_epoch_seconds(ts_raw)
        bar = Bar(ts=ts, open=float(o), high=float(h), low=float(l),
                  close=float(c), volume=float(vol))
        by_ts[ts] = bar
    return [by_ts[k] for k in sorted(by_ts)]


def validate_bars(bars: Sequence[Bar], *, expected_tf: Optional[str] = None) -> List[str]:
    """Return a list of human-readable data-quality warnings (empty == clean).

    Checks OHLC sanity (high>=low, high>=open/close, etc.), non-negative volume,
    strict time ordering, and — if ``expected_tf`` is given — flags gaps/irregular
    spacing. Bad data silently ruins a backtest, so surface it loudly (DESIGN.md §4
    gate 4: realism).
    """
    warnings: List[str] = []
    if not bars:
        return ["empty bar series"]
    step = timeframe_to_seconds(expected_tf) if expected_tf else None
    for i, b in enumerate(bars):
        if not (b.high >= b.low):
            warnings.append(f"bar[{i}] ts={b.ts}: high {b.high} < low {b.low}")
        if not (b.high >= b.open and b.high >= b.close):
            warnings.append(f"bar[{i}] ts={b.ts}: high below open/close")
        if not (b.low <= b.open and b.low <= b.close):
            warnings.append(f"bar[{i}] ts={b.ts}: low above open/close")
        if b.volume < 0:
            warnings.append(f"bar[{i}] ts={b.ts}: negative volume {b.volume}")
        if i > 0:
            prev = bars[i - 1]
            if b.ts <= prev.ts:
                warnings.append(f"bar[{i}] ts={b.ts}: not strictly after prev {prev.ts}")
            elif step is not None:
                gap = b.ts - prev.ts
                if gap != step:
                    n_missing = gap // step - 1
                    warnings.append(
                        f"bar[{i}] ts={b.ts}: spacing {gap}s != {step}s "
                        f"(~{n_missing} missing candle(s))")
    return warnings


# ---- CSV I/O (offline, deterministic) ---------------------------------------
_CSV_HEADER = ["timestamp", "open", "high", "low", "close", "volume"]


def save_csv(bars: Sequence[Bar], path: Union[str, Path]) -> Path:
    """Persist bars to CSV (epoch seconds) so a one-time fetch can be replayed offline."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(_CSV_HEADER)
        for b in bars:
            w.writerow([b.ts, b.open, b.high, b.low, b.close, b.volume])
    return path


def load_csv(path: Union[str, Path], *, ts_in_ms: Optional[bool] = None) -> List[Bar]:
    """Load OHLCV from CSV. Accepts a header row (any of timestamp/time/ts/date for
    the time column) or headerless ``ts,o,h,l,c,v`` rows. Extra columns are ignored.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"no such CSV: {path}")
    rows: List[Sequence] = []
    with path.open("r", newline="") as fh:
        reader = csv.reader(fh)
        first = True
        for raw in reader:
            if not raw:
                continue
            if first:
                first = False
                # detect a header row (non-numeric first cell)
                try:
                    float(raw[0])
                except ValueError:
                    continue  # skip header
            rows.append(raw[:6])
    return normalize_bars(rows, ts_in_ms=ts_in_ms)


# ---- ccxt fetch (lazy import; needs network + `pip install ccxt`) -----------
def fetch_ccxt(
    symbol: str = "BTC/USDT:USDT",
    timeframe: str = "15m",
    *,
    exchange: str = "binanceusdm",
    limit: int = 1500,
    since_ms: Optional[int] = None,
    max_batches: int = 50,
    params: Optional[dict] = None,
) -> List[Bar]:
    """Fetch real historical perp OHLCV via ccxt, paginating forward from ``since_ms``.

    ``ccxt`` is imported here (not at module load) so the rest of the package stays
    stdlib-only and importable in an offline sandbox. Raises a clear, actionable
    error if ccxt is missing or the network is unavailable.

    Args:
        symbol: ccxt unified symbol; for perps use the ``BASE/QUOTE:SETTLE`` form,
            e.g. ``"BTC/USDT:USDT"``.
        timeframe: candle size, e.g. ``"15m"``, ``"1h"``, ``"4h"``, ``"1d"``.
        exchange: any ccxt exchange id with a USDⓈ-M perp market
            (default ``binanceusdm``).
        limit: candles per request (exchange-capped, commonly ~1000-1500).
        since_ms: start time in epoch **milliseconds**; None => most recent ``limit``.
        max_batches: safety cap on pagination loops.
    """
    try:
        import ccxt  # type: ignore  # lazy: only needed for live fetch
    except ImportError as exc:  # pragma: no cover - exercised only without ccxt
        raise RuntimeError(
            "fetch_ccxt requires the optional 'ccxt' package and network access. "
            "Install it where PyPI is reachable:  pip install ccxt  (or "
            "pip install -e '.[data]').  For offline work use load_csv/synthetic_bars."
        ) from exc

    if not hasattr(ccxt, exchange):
        raise ValueError(f"unknown ccxt exchange id {exchange!r}")
    client = getattr(ccxt, exchange)({"enableRateLimit": True})
    tf_ms = timeframe_to_seconds(timeframe) * 1000
    all_rows: List[Sequence] = []
    cursor = since_ms
    try:
        if cursor is None:
            # most recent `limit` candles in a single call
            all_rows = client.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit,
                                          params=params or {})
        else:
            for _ in range(max_batches):
                batch = client.fetch_ohlcv(symbol, timeframe=timeframe, since=cursor,
                                           limit=limit, params=params or {})
                if not batch:
                    break
                all_rows.extend(batch)
                cursor = batch[-1][0] + tf_ms
                if len(batch) < limit:
                    break
    except Exception as exc:  # ccxt raises network/exchange errors; make them clear
        raise RuntimeError(
            f"ccxt fetch failed on {exchange}:{symbol} ({timeframe}). In this sandbox "
            f"the network is restricted; run where the exchange is reachable. "
            f"Underlying error: {exc}"
        ) from exc
    return normalize_bars(all_rows, ts_in_ms=True)


# ---- synthetic (deterministic smoke data) -----------------------------------
def synthetic_bars(
    n: int = 3000,
    *,
    seed: int = 7,
    start_price: float = 30_000.0,
    start_ts: int = 1_700_000_000,
    timeframe: str = "15m",
    sweep_every: int = 37,
) -> List[Bar]:
    """Deterministic fabricated OHLCV with periodic sweep spikes to exercise the SFP
    logic. Numbers are meaningless — this only proves the pipeline is wired.
    Mirrors the generator in ``examples/demo.py`` but timeframe-aware and reusable.
    """
    rng = random.Random(seed)
    step_s = timeframe_to_seconds(timeframe)
    price = start_price
    rows: List[Sequence] = []
    for i in range(n):
        drift = math.sin(i / 50.0) * 5
        move = rng.gauss(drift, 40)
        o = price
        c = price + move
        hi = max(o, c) + abs(rng.gauss(0, 25))
        lo = min(o, c) - abs(rng.gauss(0, 25))
        if sweep_every and i % sweep_every == 0:
            lo -= abs(rng.gauss(60, 20))  # inject a downside sweep wick
            c = max(o, c)
        vol = abs(rng.gauss(1000, 300)) * (2.0 if (sweep_every and i % sweep_every == 0) else 1.0)
        rows.append([start_ts + i * step_s, o, hi, lo, c, vol])
        price = c
    return normalize_bars(rows, ts_in_ms=False)


# ---- unified entry point ----------------------------------------------------
def load_bars(
    source: str = "synthetic",
    *,
    symbol: str = "BTC/USDT:USDT",
    timeframe: str = "15m",
    path: Optional[Union[str, Path]] = None,
    validate: bool = True,
    **kwargs,
) -> List[Bar]:
    """One call to rule them all. ``source`` in {"synthetic","csv","ccxt"}.

    Returns clean, sorted ``List[Bar]`` ready for ``run_backtest``. When
    ``validate`` is True, data-quality warnings are printed (they do not raise) so a
    bad feed can't silently poison a backtest.
    """
    source = source.lower()
    if source == "csv":
        if path is None:
            raise ValueError("source='csv' requires path=...")
        bars = load_csv(path, ts_in_ms=kwargs.get("ts_in_ms"))
    elif source == "ccxt":
        bars = fetch_ccxt(symbol=symbol, timeframe=timeframe, **kwargs)
    elif source == "synthetic":
        bars = synthetic_bars(timeframe=timeframe,
                              **{k: v for k, v in kwargs.items()
                                 if k in {"n", "seed", "start_price", "start_ts", "sweep_every"}})
    else:
        raise ValueError(f"unknown source {source!r} (expected 'synthetic'|'csv'|'ccxt')")

    if validate:
        for w in validate_bars(bars, expected_tf=timeframe):
            print(f"[datafeed] WARNING: {w}")
    return bars
