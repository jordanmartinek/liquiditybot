"""liq_ai_bot — rules-first liquidity-strategy trading system (MyFundedPerps R&D).

Layering (see DESIGN.md §3):

    market data -> SIGNAL ENGINE (deterministic) -> ML FILTER (optional)
                -> RISK / COMPLIANCE LAYER (hard guards) -> EXECUTION -> JOURNAL

Design invariant: the risk/compliance layer can only ever REDUCE or BLOCK
risk, never add it. Hard limits are deterministic, never ML.

The M1 scaffold is pure standard library so it runs offline. Third-party
deps (pandas/numpy/ccxt/pydantic/sklearn) belong to M2+ and are not imported
at module load.
"""

__version__ = "0.1.0"
