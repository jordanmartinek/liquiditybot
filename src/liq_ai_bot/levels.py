"""Full level-seeding + confluence engine (M2) — faithful port of LiquidityRadar.pine.

The M1 strategy stubbed the level map (only auto-tracked swing highs/lows with a
hardcoded confluence=50). This module ports the indicator's real logic so the
Python backtest matches the chart:

  * Previous-period levels: PDH/PDL (day), PWH/PWL (week), PMH/PML (month), D.Open.
  * Session H/L: Asia 0000-0800, London 0800-1600, NY 1300-2100 (UTC), reset daily,
    removed once swept.
  * Swing highs/lows via confirmed pivots; kept until swept (capped per side).
  * Equal highs/lows (EQH/EQL): a new pivot within `eq_tol * ATR` of a recent
    same-type pivot -> the strongest sweep magnet.
  * Order blocks after displacement (used by the confluence score).
  * Live dealing range -> OTE-fib / premium-discount confluence factor.
  * `confluence_score(...)` 0..100 mirroring the Pine `f_confScore`.

The engine is fed one CLOSED `Bar` at a time (event-driven, matching live), and
exposes the current set of un-swept `Level`s with real confluence scores. It is
timeframe-agnostic: day/week/month/session boundaries are derived from each bar's
UTC timestamp, so it works on any intraday feed.

Pure standard library.
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Deque, List, Optional, Tuple

from .types import Bar, Level, LevelType, PoolSide


# ---- tunables (ported from the Pine inputs) ---------------------------------
@dataclass(frozen=True)
class LevelParams:
    swing_len: int = 10           # pivot length (bars each side); Pine swingLen
    swing_hist: int = 40          # max un-swept swings kept per side; Pine swingHist
    htf_factor: int = 16          # HTF swing timeframe as a multiple of base bars (~4h on 15m)
    htf_len: int = 5              # HTF pivot length; Pine htfLen
    eq_tol_atr: float = 0.35      # EQ tolerance in ATR; Pine eqTol
    eq_lookback: int = 6          # recent same-type pivots to compare; Pine eqLookback
    ob_disp_atr: float = 1.2      # displacement body size in ATR; Pine obDisp
    ob_keep: int = 8              # max order blocks kept; Pine obKeep
    conf_round_step: float = 0.0  # round-number confluence step; Pine confRound (0=off)
    session_tz_offset_h: int = 0  # session timezone offset from UTC (Pine default UTC)
    # OTE retracement ratios used by the confluence score (Pine pdFib1..3)
    ote_fibs: Tuple[float, float, float] = (0.705, 0.788, 0.886)


# session windows in minutes-since-midnight [start, end) (Pine defaults, UTC)
_SESSIONS = {
    "Asia":   (0 * 60, 8 * 60),
    "London": (8 * 60, 16 * 60),
    "NY":     (13 * 60, 21 * 60),
}


def _mins_of_day(ts: int, tz_offset_h: int) -> int:
    dt = datetime.fromtimestamp(ts + tz_offset_h * 3600, tz=timezone.utc)
    return dt.hour * 60 + dt.minute


def _period_keys(ts: int, tz_offset_h: int) -> Tuple[str, str, str]:
    """(day_key, week_key, month_key) buckets for previous-period rollover."""
    dt = datetime.fromtimestamp(ts + tz_offset_h * 3600, tz=timezone.utc)
    iso = dt.isocalendar()
    return (dt.strftime("%Y-%m-%d"),
            f"{iso[0]}-W{iso[1]:02d}",
            dt.strftime("%Y-%m"))


@dataclass
class _PeriodAgg:
    """Accumulates the high/low of the *current* period; on rollover the finished
    period's extremes become the 'previous-period' levels."""
    key: Optional[str] = None
    hi: float = -math.inf
    lo: float = math.inf
    prev_hi: Optional[float] = None
    prev_lo: Optional[float] = None
    open_price: Optional[float] = None      # open of the current period (for D.Open)
    prev_open: Optional[float] = None

    def update(self, bar: Bar, key: str) -> bool:
        """Feed a bar; return True if this bar started a NEW period (rollover)."""
        rolled = False
        if key != self.key:
            if self.key is not None:
                self.prev_hi, self.prev_lo, self.prev_open = self.hi, self.lo, self.open_price
                rolled = True
            self.key = key
            self.hi, self.lo, self.open_price = bar.high, bar.low, bar.open
        else:
            self.hi = max(self.hi, bar.high)
            self.lo = min(self.lo, bar.low)
        return rolled


@dataclass
class _SessionAgg:
    """Tracks a single session's H/L for the current day; reset each new day."""
    name: str
    start_min: int
    end_min: int
    hi: Optional[float] = None
    lo: Optional[float] = None
    was_in: bool = False

    def in_window(self, mins: int) -> bool:
        return self.start_min <= mins < self.end_min


@dataclass
class OrderBlock:
    top: float
    bot: float
    bull: bool
    birth_ts: int

    def contains_band(self, lo_b: float, hi_b: float) -> bool:
        return hi_b >= self.bot and lo_b <= self.top


class LevelEngine:
    """Stateful level map. Feed closed bars via `on_bar`; read `levels`."""

    def __init__(self, params: Optional[LevelParams] = None, atr_len: int = 14):
        self.p = params or LevelParams()
        self.atr_len = atr_len
        # rolling ATR
        self._trs: Deque[float] = deque(maxlen=atr_len)
        self._prev_close: Optional[float] = None
        self.atr: float = 0.0
        # bar history for pivots (base + HTF)
        self._bars: Deque[Bar] = deque(maxlen=max(self.p.swing_len * 2 + 1, 64))
        self._htf_bars: Deque[Bar] = deque(maxlen=max(self.p.htf_len * 2 + 1, 64))
        self._htf_accum: Optional[Bar] = None
        self._htf_count = 0
        # previous-period aggregators
        self._day = _PeriodAgg()
        self._week = _PeriodAgg()
        self._month = _PeriodAgg()
        self._prev_day_key: Optional[str] = None
        # sessions
        self._sessions = [_SessionAgg(n, s, e) for n, (s, e) in _SESSIONS.items()]
        # structural stores (price, birth_ts) kept until swept
        self._sw_hi: List[Tuple[float, int]] = []
        self._sw_lo: List[Tuple[float, int]] = []
        self._eq_hi: List[Tuple[float, int]] = []
        self._eq_lo: List[Tuple[float, int]] = []
        self._pv_hi: Deque[float] = deque(maxlen=40)   # recent high pivots for EQ hunt
        self._pv_lo: Deque[float] = deque(maxlen=40)
        # order blocks
        self.order_blocks: List[OrderBlock] = []
        # dealing range (for OTE / premium-discount confluence)
        self.rng_hi: Optional[float] = None
        self.rng_lo: Optional[float] = None
        self.range_dir: int = 0        # +1 bull leg (after low pivot), -1 bear leg
        self._last_ph: Optional[float] = None
        self._last_pl: Optional[float] = None
        # previous-period sweep masks
        self._sw_mask = {k: False for k in
                         ("pdh", "pdl", "pwh", "pwl", "pmh", "pml")}
        self._prev_period_prices = {k: None for k in self._sw_mask}
        self._last_bar: Optional[Bar] = None

    # ---- rolling ATR --------------------------------------------------------
    def _update_atr(self, bar: Bar) -> None:
        if self._prev_close is None:
            tr = bar.high - bar.low
        else:
            tr = max(bar.high - bar.low, abs(bar.high - self._prev_close),
                     abs(bar.low - self._prev_close))
        self._trs.append(tr)
        self._prev_close = bar.close
        self.atr = sum(self._trs) / len(self._trs) if self._trs else 0.0

    # ---- pivots -------------------------------------------------------------
    @staticmethod
    def _confirmed_pivot(bars, n: int) -> Optional[Tuple[str, Bar]]:
        if len(bars) < 2 * n + 1:
            return None
        pivot = bars[-(n + 1)]
        left = bars[-(2 * n + 1):-(n + 1)]
        right = bars[-n:]
        if pivot.high > max(b.high for b in left + right):
            return ("high", pivot)
        if pivot.low < min(b.low for b in left + right):
            return ("low", pivot)
        return None

    # ---- main entry ---------------------------------------------------------
    def on_bar(self, bar: Bar) -> None:
        self._update_atr(bar)
        prev_bar = self._last_bar

        # 1) previous-period levels (day/week/month) + daily open
        dk, wk, mk = _period_keys(bar.ts, self.p.session_tz_offset_h)
        self._day.update(bar, dk)
        self._week.update(bar, wk)
        self._month.update(bar, mk)
        new_day = dk != self._prev_day_key

        # 2) sessions
        self._update_sessions(bar, new_day)
        self._prev_day_key = dk

        # 3) previous-period sweep masking (reset mask when the level refreshes)
        self._update_prev_period_masks(bar)

        # 4) swings (base + HTF) and equal highs/lows
        self._bars.append(bar)
        self._track_swings(bar)
        self._track_htf(bar)

        # 5) order blocks (needs prior bar)
        if prev_bar is not None:
            self._track_order_blocks(bar, prev_bar)

        # 6) remove swept structural levels
        self._remove_swept(bar)

        self._last_bar = bar

    # ---- sessions -----------------------------------------------------------
    def _update_sessions(self, bar: Bar, new_day: bool) -> None:
        mins = _mins_of_day(bar.ts, self.p.session_tz_offset_h)
        for s in self._sessions:
            if new_day:
                # reset daily: an unswept level from yesterday is dropped here
                s.hi = s.lo = None
                s.was_in = False
            in_s = s.in_window(mins)
            if in_s and not s.was_in:
                s.hi, s.lo = bar.high, bar.low
            elif in_s:
                s.hi = max(s.hi, bar.high) if s.hi is not None else bar.high
                s.lo = min(s.lo, bar.low) if s.lo is not None else bar.low
            # remove once swept
            if s.hi is not None and bar.high > s.hi:
                s.hi = None
            if s.lo is not None and bar.low < s.lo:
                s.lo = None
            s.was_in = in_s

    # ---- previous-period sweep masks ---------------------------------------
    def _update_prev_period_masks(self, bar: Bar) -> None:
        cur = {
            "pdh": self._day.prev_hi, "pdl": self._day.prev_lo,
            "pwh": self._week.prev_hi, "pwl": self._week.prev_lo,
            "pmh": self._month.prev_hi, "pml": self._month.prev_lo,
        }
        for k, price in cur.items():
            if price is None:
                continue
            if price != self._prev_period_prices[k]:
                self._sw_mask[k] = False        # refreshed level -> un-sweep
                self._prev_period_prices[k] = price
            is_high = k.endswith("h")
            if is_high and bar.high > price:
                self._sw_mask[k] = True
            elif not is_high and bar.low < price:
                self._sw_mask[k] = True

    # ---- swings + EQ --------------------------------------------------------
    def _track_swings(self, bar: Bar) -> None:
        piv = self._confirmed_pivot(list(self._bars), self.p.swing_len)
        if piv is None:
            return
        kind, pv = piv
        if kind == "high":
            self._sw_hi.append((pv.high, pv.ts))
            self._last_ph = pv.high
            self.range_dir = -1
            self._hunt_equal(pv.high, is_high=True)
            self._pv_hi.append(pv.high)
        else:
            self._sw_lo.append((pv.low, pv.ts))
            self._last_pl = pv.low
            self.range_dir = 1
            self._hunt_equal(pv.low, is_high=False)
            self._pv_lo.append(pv.low)
        self._update_dealing_range(bar)
        # cap history per side
        while len(self._sw_hi) > self.p.swing_hist:
            self._sw_hi.pop(0)
        while len(self._sw_lo) > self.p.swing_hist:
            self._sw_lo.pop(0)

    def _hunt_equal(self, price: float, is_high: bool) -> None:
        tol = self.atr * self.p.eq_tol_atr
        if tol <= 0:
            return
        recent = list(self._pv_hi if is_high else self._pv_lo)[-self.p.eq_lookback:]
        for prev in recent:
            if abs(price - prev) <= tol:
                matched = max(price, prev) if is_high else min(price, prev)
                ts = self._bars[-1].ts if self._bars else 0
                (self._eq_hi if is_high else self._eq_lo).append((matched, ts))
                break
        while len(self._eq_hi) > self.p.swing_hist:
            self._eq_hi.pop(0)
        while len(self._eq_lo) > self.p.swing_hist:
            self._eq_lo.pop(0)

    def _track_htf(self, bar: Bar) -> None:
        # aggregate `htf_factor` base bars into one synthetic HTF bar
        if self._htf_accum is None:
            self._htf_accum = bar
            self._htf_count = 1
        else:
            a = self._htf_accum
            self._htf_accum = Bar(a.ts, a.open, max(a.high, bar.high),
                                  min(a.low, bar.low), bar.close,
                                  a.volume + bar.volume)
            self._htf_count += 1
        if self._htf_count >= self.p.htf_factor:
            self._htf_bars.append(self._htf_accum)
            self._htf_accum = None
            self._htf_count = 0
            piv = self._confirmed_pivot(list(self._htf_bars), self.p.htf_len)
            if piv is not None:
                kind, pv = piv
                if kind == "high":
                    self._sw_hi.append((pv.high, pv.ts))
                else:
                    self._sw_lo.append((pv.low, pv.ts))

    # ---- dealing range ------------------------------------------------------
    def _update_dealing_range(self, bar: Bar) -> None:
        if self._last_pl is not None:
            self.rng_lo = self._last_pl
        if self._last_ph is not None:
            self.rng_hi = self._last_ph
        if self.rng_hi is None:
            self.rng_hi = bar.high
        if self.rng_lo is None:
            self.rng_lo = bar.low
        if self.range_dir >= 0:
            self.rng_hi = max(self.rng_hi, bar.high)
        else:
            self.rng_lo = min(self.rng_lo, bar.low)

    # ---- order blocks -------------------------------------------------------
    def _track_order_blocks(self, bar: Bar, prev: Bar) -> None:
        body = abs(bar.close - bar.open)
        disp = self.atr > 0 and body >= self.p.ob_disp_atr * self.atr
        if disp and bar.close > bar.open and bar.close > prev.high:
            self.order_blocks.append(OrderBlock(prev.high, prev.low, True, prev.ts))
        elif disp and bar.close < bar.open and bar.close < prev.low:
            self.order_blocks.append(OrderBlock(prev.high, prev.low, False, prev.ts))
        # mitigate
        alive: List[OrderBlock] = []
        for ob in self.order_blocks:
            dead = (bar.close < ob.bot) if ob.bull else (bar.close > ob.top)
            if not dead:
                alive.append(ob)
        self.order_blocks = alive[-self.p.ob_keep:]

    # ---- sweep removal for structural stores --------------------------------
    def _remove_swept(self, bar: Bar) -> None:
        self._sw_hi = [(p, b) for (p, b) in self._sw_hi if not bar.high > p]
        self._sw_lo = [(p, b) for (p, b) in self._sw_lo if not bar.low < p]
        self._eq_hi = [(p, b) for (p, b) in self._eq_hi if not bar.high > p]
        self._eq_lo = [(p, b) for (p, b) in self._eq_lo if not bar.low < p]

    # ---- confluence score (port of f_confScore) -----------------------------
    def confluence_score(self, mid: float, lo_b: float, hi_b: float,
                         members: int = 1) -> float:
        score = 0.0
        tol = max(self.atr * 0.25, 1e-9)
        score += min(members, 5) * 12.0
        # order-block overlap
        if any(ob.contains_band(lo_b, hi_b) for ob in self.order_blocks):
            score += 22.0
        # OTE fib + premium/discount from the live dealing range
        if (self.rng_hi is not None and self.rng_lo is not None
                and self.rng_hi > self.rng_lo):
            rng = self.rng_hi - self.rng_lo
            bull_leg = self.range_dir >= 0
            fibs = [
                (self.rng_hi - f * rng) if bull_leg else (self.rng_lo + f * rng)
                for f in self.p.ote_fibs
            ]
            if any(abs(mid - f) <= tol for f in fibs):
                score += 18.0
            pos = (mid - self.rng_lo) / rng
            if pos >= 0.75 or pos <= 0.25:
                score += 8.0
        # round-number proximity
        if self.p.conf_round_step > 0:
            nearest = round(mid / self.p.conf_round_step) * self.p.conf_round_step
            if abs(mid - nearest) <= tol:
                score += 10.0
        return float(min(round(score), 100))

    # ---- assemble the current level map -------------------------------------
    def levels(self) -> List[Level]:
        """Return every currently-tracked, un-swept level with a real confluence
        score attached. Prices/types mirror the indicator's f_addLevel set."""
        out: List[Level] = []

        def add(price: Optional[float], ltype: LevelType, side: PoolSide, birth: int):
            if price is None:
                return
            conf = self.confluence_score(price, price - self.atr * 0.05,
                                         price + self.atr * 0.05, members=1)
            out.append(Level(price, ltype, side, birth, confluence=conf))

        # previous-period (respect sweep mask)
        d, w, m = self._day, self._week, self._month
        if not self._sw_mask["pdh"]:
            add(d.prev_hi, LevelType.PDH, PoolSide.BSL, 0)
        if not self._sw_mask["pdl"]:
            add(d.prev_lo, LevelType.PDL, PoolSide.SSL, 0)
        if not self._sw_mask["pwh"]:
            add(w.prev_hi, LevelType.PWH, PoolSide.BSL, 0)
        if not self._sw_mask["pwl"]:
            add(w.prev_lo, LevelType.PWL, PoolSide.SSL, 0)
        if not self._sw_mask["pmh"]:
            add(m.prev_hi, LevelType.PMH, PoolSide.BSL, 0)
        if not self._sw_mask["pml"]:
            add(m.prev_lo, LevelType.PML, PoolSide.SSL, 0)

        # sessions
        session_types = {
            "Asia": (LevelType.SESSION_H, LevelType.SESSION_L),
            "London": (LevelType.SESSION_H, LevelType.SESSION_L),
            "NY": (LevelType.SESSION_H, LevelType.SESSION_L),
        }
        for s in self._sessions:
            th, tl = session_types[s.name]
            add(s.hi, th, PoolSide.BSL, 0)
            add(s.lo, tl, PoolSide.SSL, 0)

        # swings
        for p, b in self._sw_hi:
            add(p, LevelType.SWING_H, PoolSide.BSL, b)
        for p, b in self._sw_lo:
            add(p, LevelType.SWING_L, PoolSide.SSL, b)

        # equal highs/lows (strongest magnets)
        for p, b in self._eq_hi:
            add(p, LevelType.EQ_H, PoolSide.BSL, b)
        for p, b in self._eq_lo:
            add(p, LevelType.EQ_L, PoolSide.SSL, b)

        return out
