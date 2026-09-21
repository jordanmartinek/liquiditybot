"""Deterministic strategy core (DESIGN.md §2).

Direct port of the liquidity indicator's logic to executable Python:
  * level tracking (the map),
  * DOL bias via a per-level "pull" gravity model,
  * SFP (sweep + reversal) entry trigger with volume, DOL, confluence gates.

Rules decide WHETHER to trade. The optional ML filter (M5) decides whether
*this instance* is worth taking; the risk engine decides HOW MUCH. This module
never sizes positions and never touches money — it only emits `Setup`s.

Pure standard library. `atr`/`rvol` are simple rolling computations here; swap
for pandas/numpy vectorized versions in M2 if desired.
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Deque, List, Optional, Protocol

from .types import Bar, Bias, DOLState, Level, LevelType, PoolSide, Setup, Side


# ---- rolling indicators (stdlib) --------------------------------------------
class Rolling:
    """Maintains rolling ATR and volume average over a window of closed bars."""

    def __init__(self, atr_len: int = 14, vol_len: int = 20, swing_len: int = 5):
        self.atr_len = atr_len
        self.vol_len = vol_len
        self.swing_len = swing_len
        self._trs: Deque[float] = deque(maxlen=atr_len)
        self._vols: Deque[float] = deque(maxlen=vol_len)
        self._bars: Deque[Bar] = deque(maxlen=max(swing_len * 2 + 1, 64))
        self._prev_close: Optional[float] = None

    def update(self, bar: Bar) -> None:
        if self._prev_close is None:
            tr = bar.high - bar.low
        else:
            tr = max(
                bar.high - bar.low,
                abs(bar.high - self._prev_close),
                abs(bar.low - self._prev_close),
            )
        self._trs.append(tr)
        self._vols.append(bar.volume)
        self._bars.append(bar)
        self._prev_close = bar.close

    @property
    def atr(self) -> float:
        return sum(self._trs) / len(self._trs) if self._trs else 0.0

    @property
    def avg_vol(self) -> float:
        return sum(self._vols) / len(self._vols) if self._vols else 0.0

    def rvol(self, bar: Bar) -> float:
        av = self.avg_vol
        return bar.volume / av if av > 0 else 0.0

    def confirmed_swing(self) -> Optional[tuple]:
        """Return (kind, Bar) for a swing confirmed `swing_len` bars ago, or None.
        kind is 'high' or 'low'. A pivot needs swing_len bars on each side."""
        n = self.swing_len
        if len(self._bars) < 2 * n + 1:
            return None
        bars = list(self._bars)
        pivot = bars[-(n + 1)]
        left = bars[-(2 * n + 1):-(n + 1)]
        right = bars[-n:]
        if pivot.high > max(b.high for b in left + right):
            return ("high", pivot)
        if pivot.low < min(b.low for b in left + right):
            return ("low", pivot)
        return None


# ---- DOL gravity model (ported from indicator's f_pull) ---------------------
@dataclass
class DOLParams:
    near_bias: float = 1.6        # distance-decay exponent (nearer pools pull harder)
    eq_weight: float = 2.4        # equal highs/lows are the strongest magnets
    freshness_atr: float = 0.15   # a level touched within this ATR fraction is "fresh"
    freshness_factor: float = 0.6
    conviction_floor: float = 0.15


def _level_pull(level: Level, price: float, atr: float, p: DOLParams) -> float:
    """Per-level pull toward `price`. Mirrors the Pine f_pull(): distance-decay ×
    type-weight × freshness × confluence. Age is left to the caller (M2)."""
    if level.swept or atr <= 0:
        return 0.0
    dist = abs(level.price - price) / atr
    dist = max(dist, 1e-6)
    decay = math.pow(1.0 / (1.0 + dist), p.near_bias)
    type_w = p.eq_weight if level.ltype in (LevelType.EQ_H, LevelType.EQ_L) else 1.0
    fresh = p.freshness_factor if dist <= p.freshness_atr else 1.0
    conf = 1.0 + level.confluence / 100.0
    return decay * type_w * fresh * conf


def compute_dol(levels: List[Level], price: float, atr: float,
                params: Optional[DOLParams] = None) -> DOLState:
    """Aggregate per-level pull into a directional bias + conviction."""
    p = params or DOLParams()
    up_pull = 0.0    # pull toward BSL (highs) above -> bias UP
    down_pull = 0.0  # pull toward SSL (lows) below -> bias DOWN
    best_price, best_pull = None, 0.0
    for lv in levels:
        pull = _level_pull(lv, price, atr, p)
        if pull <= 0:
            continue
        if lv.price >= price:
            up_pull += pull
        else:
            down_pull += pull
        if pull > best_pull:
            best_pull, best_price = pull, lv.price
    total = up_pull + down_pull
    if total <= 0:
        return DOLState(Bias.NEUTRAL, 0.0, None)
    conviction = abs(up_pull - down_pull) / total
    if conviction < p.conviction_floor:
        return DOLState(Bias.NEUTRAL, conviction, best_price)
    bias = Bias.UP if up_pull >= down_pull else Bias.DOWN
    return DOLState(bias, conviction, best_price)


# ---- signal engine ----------------------------------------------------------
@dataclass
class SignalParams:
    rvol_min: float = 1.3         # volume-confirmation threshold
    confluence_min: float = 40.0  # confluence gate (0..100)
    conviction_min: float = 0.25  # DOL conviction floor to take a setup
    stop_pad_atr: float = 0.25    # stop padding beyond the sweep wick
    fallback_r: float = 2.0       # target R multiple if no opposing pool


class Strategy(Protocol):
    """Interface every strategy implements. `on_bar` is called once per CLOSED
    bar and returns at most one candidate Setup (the risk layer vets it)."""

    def on_bar(self, bar: Bar) -> Optional[Setup]: ...


class LiquiditySFPStrategy:
    """The deterministic core. Tracks levels, computes DOL, fires on sweep+reversal.

    Level seeding (PDH/PDL, sessions, equal highs/lows, HTF swings) is stubbed:
    M2 wires in real session/day boundaries and the confluence model. For now
    swing highs/lows are auto-tracked so the engine is runnable end-to-end.
    """

    def __init__(self, symbol: str, params: Optional[SignalParams] = None,
                 rolling: Optional[Rolling] = None, dol_params: Optional[DOLParams] = None):
        self.symbol = symbol
        self.p = params or SignalParams()
        self.roll = rolling or Rolling()
        self.dol_params = dol_params or DOLParams()
        self.levels: List[Level] = []
        self._last_bias: Optional[DOLState] = None

    # --- level bookkeeping ---------------------------------------------------
    def add_level(self, level: Level) -> None:
        self.levels.append(level)

    def _track_swings(self, bar: Bar) -> None:
        sw = self.roll.confirmed_swing()
        if sw is None:
            return
        kind, pivot = sw
        if kind == "high":
            self.levels.append(Level(pivot.high, LevelType.SWING_H, PoolSide.BSL,
                                     pivot.ts, confluence=50.0))
        else:
            self.levels.append(Level(pivot.low, LevelType.SWING_L, PoolSide.SSL,
                                     pivot.ts, confluence=50.0))

    def _mark_swept(self, bar: Bar) -> List[Level]:
        """Mark levels the current bar's wick pierced; return those swept THIS bar."""
        swept_now: List[Level] = []
        for lv in self.levels:
            if lv.swept:
                continue
            if lv.side is PoolSide.SSL and bar.low < lv.price:
                lv.swept = True; lv.swept_ts = bar.ts; swept_now.append(lv)
            elif lv.side is PoolSide.BSL and bar.high > lv.price:
                lv.swept = True; lv.swept_ts = bar.ts; swept_now.append(lv)
        return swept_now

    def _opposing_pool(self, side: Side, price: float) -> Optional[Level]:
        """Nearest un-swept opposing pool = the DOL magnet target."""
        if side is Side.LONG:
            cands = [lv for lv in self.levels if not lv.swept and lv.price > price
                     and lv.side is PoolSide.BSL]
            return min(cands, key=lambda lv: lv.price) if cands else None
        cands = [lv for lv in self.levels if not lv.swept and lv.price < price
                 and lv.side is PoolSide.SSL]
        return max(cands, key=lambda lv: lv.price) if cands else None

    # --- main entry point ----------------------------------------------------
    def on_bar(self, bar: Bar) -> Optional[Setup]:
        self.roll.update(bar)
        atr = self.roll.atr
        self._track_swings(bar)
        swept_now = self._mark_swept(bar)
        dol = compute_dol(self.levels, bar.close, atr, self.dol_params)
        self._last_bias = dol

        if atr <= 0 or not swept_now:
            return None

        rvol = self.roll.rvol(bar)
        if rvol < self.p.rvol_min:
            return None
        if dol.conviction < self.p.conviction_min:
            return None

        # LONG: swept an SSL (low) and closed back above it, DOL points up.
        for lv in swept_now:
            if (lv.side is PoolSide.SSL and bar.close > lv.price
                    and dol.bias is Bias.UP and lv.confluence >= self.p.confluence_min):
                stop = bar.low - self.p.stop_pad_atr * atr
                target = self._opposing_pool(Side.LONG, bar.close)
                tgt_price = target.price if target else bar.close + self.p.fallback_r * (
                    bar.close - stop)
                return self._make_setup(bar, Side.LONG, bar.close, stop, tgt_price, lv, rvol, dol)
            # SHORT: swept a BSL (high) and closed back below it, DOL points down.
            if (lv.side is PoolSide.BSL and bar.close < lv.price
                    and dol.bias is Bias.DOWN and lv.confluence >= self.p.confluence_min):
                stop = bar.high + self.p.stop_pad_atr * atr
                target = self._opposing_pool(Side.SHORT, bar.close)
                tgt_price = target.price if target else bar.close - self.p.fallback_r * (
                    stop - bar.close)
                return self._make_setup(bar, Side.SHORT, bar.close, stop, tgt_price, lv, rvol, dol)
        return None

    def _make_setup(self, bar: Bar, side: Side, entry: float, stop: float,
                    target: float, lv: Level, rvol: float, dol: DOLState) -> Setup:
        return Setup(
            ts=bar.ts, symbol=self.symbol, side=side, entry=entry, stop=stop, target=target,
            swept_level=lv,
            features={
                "confluence": lv.confluence,
                "rvol": rvol,
                "dol_bias": dol.bias.value,
                "dol_conviction": dol.conviction,
                "atr": self.roll.atr,
                "level_type": lv.ltype.value,
            },
        )
