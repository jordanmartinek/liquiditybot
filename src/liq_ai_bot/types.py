"""Core data types shared across the strategy, risk, and backtest layers.

Pure standard library. These mirror the concepts in the LiquidityRadar Pine
indicator so backtest behavior matches live behavior.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class Side(str, Enum):
    LONG = "long"
    SHORT = "short"


class PoolSide(str, Enum):
    BSL = "bsl"   # buy-side liquidity: resting stops ABOVE price (highs)
    SSL = "ssl"   # sell-side liquidity: resting stops BELOW price (lows)


class LevelType(str, Enum):
    PDH = "PDH"; PDL = "PDL"
    PWH = "PWH"; PWL = "PWL"
    PMH = "PMH"; PML = "PML"
    SESSION_H = "SESSION_H"; SESSION_L = "SESSION_L"
    SWING_H = "SWING_H"; SWING_L = "SWING_L"
    EQ_H = "EQ_H"; EQ_L = "EQ_L"      # equal highs/lows (strongest sweep magnets)


@dataclass(frozen=True)
class Bar:
    """One closed OHLCV candle. Times are epoch seconds (UTC)."""
    ts: int
    open: float
    high: float
    low: float
    close: float
    volume: float


@dataclass
class Level:
    """A tracked liquidity level (the 'map')."""
    price: float
    ltype: LevelType
    side: PoolSide
    birth_ts: int
    confluence: float = 0.0     # 0..100 (from the indicator's confluence model)
    swept: bool = False
    swept_ts: Optional[int] = None

    def is_high(self) -> bool:
        return self.side is PoolSide.BSL


@dataclass
class Setup:
    """A candidate trade produced by the signal engine, with features attached
    for the (optional) ML filter and for journaling."""
    ts: int
    symbol: str
    side: Side
    entry: float
    stop: float
    target: Optional[float]
    swept_level: Level
    features: dict = field(default_factory=dict)  # confluence, rvol, dol_align, etc.

    def r_multiple_to(self, price: float) -> float:
        risk = abs(self.entry - self.stop)
        if risk <= 0:
            return 0.0
        return (price - self.entry) / risk if self.side is Side.LONG else (self.entry - price) / risk


class Bias(str, Enum):
    UP = "up"
    DOWN = "down"
    NEUTRAL = "neutral"


@dataclass
class DOLState:
    """Draw-on-liquidity bias + conviction (mirrors indicator's gravity model)."""
    bias: Bias
    conviction: float           # 0..1
    magnet_price: Optional[float] = None   # dominant draw (highest per-level pull)
