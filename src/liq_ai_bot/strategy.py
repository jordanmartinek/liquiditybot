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

from .levels import LevelEngine, LevelParams
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
# Level-type importance weights (Pine f_pull typeW table). EQ (engineered
# liquidity) ranks at the top with monthly/HTF; sessions pull least.
_TYPE_WEIGHT = {
    LevelType.EQ_H: 2.4, LevelType.EQ_L: 2.4,
    LevelType.PMH: 2.2, LevelType.PML: 2.2,
    LevelType.PWH: 1.9, LevelType.PWL: 1.9,
    LevelType.PDH: 1.6, LevelType.PDL: 1.6,
    LevelType.SESSION_H: 1.2, LevelType.SESSION_L: 1.2,
    LevelType.SWING_H: 1.0, LevelType.SWING_L: 1.0,
}


@dataclass
class DOLParams:
    near_bias: float = 1.5        # distance-decay exponent add-on (Pine dolNearBias)
    sens: float = 1.0             # gravity sensitivity (Pine dolSens: Low .5/Med 1/High 2)
    freshness_atr: float = 0.15   # a level touched within this ATR fraction is "fresh"
    freshness_factor: float = 0.6
    conviction_floor: float = 0.15
    use_type: bool = True         # weight by level importance (Pine dolUseType)
    use_age: bool = True          # weight by resting time (Pine dolUseAge)
    use_confluence: bool = True   # amplify pull by the level's own confluence (dolConfW)
    now_ts: Optional[int] = None  # current bar ts, for the age factor (set by caller)


def _level_pull(level: Level, price: float, atr: float, p: DOLParams) -> float:
    """Per-level pull toward `price`. Faithful port of Pine f_pull():
    distance-decay × type-weight × age × freshness × confluence."""
    if level.swept or atr <= 0:
        return 0.0
    dist = abs(level.price - price)
    # sharper decay so the nearest pool dominates (dolNearBias raises the exponent)
    decay = math.pow(1.0 / (1.0 + p.sens * (dist / atr)), 1.0 + p.near_bias)
    # type weight, scaled by sensitivity: typeW = 1 + (base - 1) * sens
    type_w = 1.0
    if p.use_type:
        base = _TYPE_WEIGHT.get(level.ltype, 1.0)
        type_w = 1.0 + (base - 1.0) * p.sens
    # age weight: older resting liquidity pulls harder (log-scaled)
    age_w = 1.0
    if p.use_age and p.now_ts is not None and level.birth_ts:
        age_bars = max(0.0, (p.now_ts - level.birth_ts))
        # normalize age in ATR-independent "resting" terms via log
        age_w = 1.0 + min(1.0, math.log1p(age_bars) / 15.0) * p.sens
    # freshness discount: a level price is sitting right on is being tested
    fresh = p.freshness_factor if (dist / atr) < p.freshness_atr else 1.0
    # confluence amplifier: 1.0 .. 2.0
    conf = (1.0 + level.confluence / 100.0) if p.use_confluence else 1.0
    return decay * type_w * age_w * fresh * conf


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

    M2: level seeding is now the full port of the LiquidityRadar indicator via
    `LevelEngine` — PDH/PDL, PWH/PWL, PMH/PML, sessions, swings (+HTF), and equal
    highs/lows, each carrying a real 0-100 confluence score. The engine is fed one
    closed bar per `on_bar`, and the strategy reads its un-swept level map to detect
    sweeps, compute DOL bias, and fire SFP setups.

    A level is considered swept THIS bar when the engine drops it from its map
    while the current bar's wick pierced it (SSL low undercut / BSL high exceeded).

    Levels can still be seeded manually via `add_level` (used by focused tests);
    those live alongside the engine-produced map.
    """

    def __init__(self, symbol: str, params: Optional[SignalParams] = None,
                 dol_params: Optional[DOLParams] = None,
                 level_params: Optional[LevelParams] = None,
                 engine: Optional[LevelEngine] = None,
                 auto_seed: bool = True):
        self.symbol = symbol
        self.p = params or SignalParams()
        self.dol_params = dol_params or DOLParams()
        self.engine = engine or LevelEngine(level_params)
        # When False the engine still tracks ATR/volume but does NOT contribute its
        # seeded level map — the strategy runs purely off manually-added levels.
        # (Used by focused SFP-trigger unit tests; production leaves it True.)
        self.auto_seed = auto_seed
        # rolling volume (for the rvol confirmation gate; ATR comes from the engine)
        self.roll = Rolling(vol_len=20)
        self._manual_levels: List[Level] = []
        self._last_bias: Optional[DOLState] = None

    @property
    def levels(self) -> List[Level]:
        """Current un-swept level map: engine-seeded levels (if auto_seed) + manual."""
        seeded = self.engine.levels() if self.auto_seed else []
        return seeded + [lv for lv in self._manual_levels if not lv.swept]

    # --- level bookkeeping ---------------------------------------------------
    def add_level(self, level: Level) -> None:
        """Seed a level manually (alongside the engine's map)."""
        self._manual_levels.append(level)

    def _mark_swept(self, bar: Bar, levels: List[Level]) -> List[Level]:
        """Return levels whose wick was pierced by THIS bar (SSL low / BSL high),
        and mark manual levels swept. Engine levels are dropped by the engine
        itself once swept, so we detect the piercing on the pre-update snapshot."""
        swept_now: List[Level] = []
        for lv in levels:
            if lv.swept:
                continue
            if lv.side is PoolSide.SSL and bar.low < lv.price:
                swept_now.append(lv)
                if lv in self._manual_levels:
                    lv.swept = True; lv.swept_ts = bar.ts
            elif lv.side is PoolSide.BSL and bar.high > lv.price:
                swept_now.append(lv)
                if lv in self._manual_levels:
                    lv.swept = True; lv.swept_ts = bar.ts
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
        # snapshot the level map BEFORE this bar updates the engine, so we can
        # detect which levels THIS bar's wick pierced (the engine drops swept
        # levels during its own update).
        pre_levels = self.levels
        swept_now = self._mark_swept(bar, pre_levels)

        # advance the engine (seeds/refreshes levels, ATR, order blocks) and volume
        self.engine.on_bar(bar)
        self.roll.update(bar)
        atr = self.engine.atr

        # DOL over the fresh, un-swept map (age factor needs the current ts)
        self.dol_params.now_ts = bar.ts
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
