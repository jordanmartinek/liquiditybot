# liq-ai-bot

R&D scaffold for turning the **liquidity strategy** (from the `LiquidityRadar`
TradingView indicator) into a **rules-first automated trading system**, engineered
around a **MyFundedPerps** crypto-perps prop challenge.

> ⚠️ This is a research/engineering framework, **not** a money-making promise and
> **not** financial advice. No strategy is guaranteed to be profitable or to pass a
> challenge. Deploy real money only if honest, out-of-sample testing supports it —
> and only after you personally verify MyFundedPerps' current rules and their
> policy on automation/API access. See [`DESIGN.md`](./DESIGN.md).

This project is intentionally **separate** from the indicator repository.

## Philosophy (why it's not a black box)

The core execution is a **deterministic, auditable rule engine** — not a neural net.
"AI"/ML is used only as *filters and optimizers around the rules* (setup-quality
classifier, regime detection, adaptive params, sizing), never as the trigger, and
**never** for the hard risk limits. Rules decide *whether* to trade; ML decides
*whether this instance is worth taking* and *how much to risk*; the risk layer can
only ever **reduce or block** risk, never add it. (Full rationale in `DESIGN.md`.)

## Layout

```
DESIGN.md                     architecture, validation plan, prop tactics, RULES table
src/liq_ai_bot/
  config.py                   Settings + PropRules (confirmed rules) + AutomationPolicy + RiskConfig
  types.py                    Bar, Level, Setup, DOLState, enums
  levels.py                   M2 level engine: PDH/PDL·PWH/PWL·PMH/PML·sessions·swings·EQ·OB·confluence
  strategy.py                 deterministic core: DOL gravity model + SFP trigger over the level map
  risk_engine.py              THE hard-constraint layer (two DD trackers, daily stop, kill-switch)
  datafeed.py                 real OHLCV -> Bar (csv/ccxt/synthetic), validation, CSV round-trip
  backtest.py                 event-driven harness + Monte-Carlo pass/breach simulator
examples/demo.py              end-to-end smoke test on synthetic data (no network)
examples/backtest_real.py     backtest runner over csv/ccxt/synthetic sources
tests/                        risk engine · strategy · levels · datafeed · config/policy
```

## Status: M2 (full strategy + confirmed rules)

Runnable, tested, **offline, no live connectivity, no real money**. The level
engine (`levels.py`) is a faithful port of the LiquidityRadar Pine indicator:
previous-period levels, sessions, swings (+HTF), equal highs/lows, order blocks,
the live dealing range, and a real 0-100 confluence score — wired into the SFP
strategy. The prop `RULES` are **confirmed** (2% DLL / 4% static MLL / 6% target,
no consistency rule, automation allowed via documented API only), so an `EVAL`
profile now arms; the fail-closed gate still refuses configs that drift into a
prohibited automation class.

> Honesty check: the framework is complete, but a demonstrated **edge** is not.
> Backtest on real data (M2 feed) then run the walk-forward / OOS / cost gates
> (DESIGN.md §4) before trusting any pass/breach number.

## Requirements & running

The M1 scaffold is **pure Python standard library** (works offline; no pip
install needed). Python 3.9+ (developed/tested on 3.11).

```bash
# from the project root
python3 -m unittest discover -s tests -v     # run all tests (risk engine + strategy + datafeed)
python3 examples/demo.py                      # end-to-end pipeline on synthetic data
python3 examples/backtest_real.py --help      # backtest runner: csv | ccxt | synthetic sources
```

### Backtesting on real data (M2 data feed)

`src/liq_ai_bot/datafeed.py` turns real historical perp OHLCV into `Bar`s behind
one interface, with three interchangeable sources:

```bash
# offline smoke (works anywhere, incl. no-network sandboxes)
python3 examples/backtest_real.py --source synthetic --bars 4000

# fetch real perp history once where PyPI + network are available, save for replay
#   (needs: pip install ccxt)
python3 examples/backtest_real.py --source ccxt --symbol BTC/USDT:USDT \
    --timeframe 15m --bars 5000 --save data/btc_15m.csv

# replay real data offline, deterministically, forever
python3 examples/backtest_real.py --source csv --path data/btc_15m.csv --timeframe 15m
```

`ccxt` is imported lazily, so the package still runs on the standard library
alone when you stick to the `csv`/`synthetic` sources.

### Walk-forward validation (M3/M4) — the go/no-go gate

Proves (or retires) the edge: tunes params in-sample, evaluates out-of-sample,
sweeps costs, and prints a GO / NO-GO verdict under the confirmed prop rules.

```bash
# offline smoke (numbers meaningless, but exercises the whole pipeline)
python3 examples/walkforward.py --source synthetic --bars 4000 \
    --train 800 --test 300 --embargo 20

# real validation: fetch once (networked env), then validate offline
python3 examples/backtest_real.py --source ccxt --symbol BTC/USDT:USDT \
    --timeframe 15m --bars 20000 --save data/btc_15m.csv
python3 examples/walkforward.py --source csv --path data/btc_15m.csv \
    --timeframe 15m --train 3000 --test 1000 --embargo 50
```

A **NO-GO is the expected, correct output** until the strategy demonstrates a real
edge on real out-of-sample data. The harness is built to say NO plainly.

`requirements.txt` / the `pyproject.toml` optional-deps list the **target** stack
for M2+ (pandas, numpy, ccxt, pydantic; scikit-learn/lightgbm for the ML filter).
Install those only in an environment with PyPI access — they are **not** imported
by the M1 code.

## Roadmap

- **M1 (done):** scaffold — config/RULES, risk engine, strategy interfaces, backtest + Monte-Carlo, tests.
- **M2 (done):** real historical perp data feed (`datafeed.py`: csv/ccxt/synthetic) + real-data backtest runner; **full level/confluence port** from the Pine indicator (`levels.py`: PDH/PDL, PWH/PWL, PMH/PML, sessions, swings +HTF, equal highs/lows, order blocks, dealing-range OTE, 0-100 confluence) wired into the strategy; tests throughout.
- **M3/M4 (done):** walk-forward validation harness (`walkforward.py`) — train/test splitter with embargo, in-sample param tuning, pooled out-of-sample evaluation, cost-sensitivity sweep, and a GO/NO-GO verdict combining OOS edge + overfit decay + cost survival + Monte-Carlo P(pass)/P(breach). Runner: `examples/walkforward.py`. **Needs real perp data for a meaningful verdict.**
- **M3:** Monte-Carlo challenge sim through the risk layer → P(pass)/P(breach).
- **M4:** walk-forward + cost modeling → go/no-go on the base edge.
- **M5 (only if M4 passes):** optional ML setup-filter trained on the journaled dataset.
- **M6:** paper-trade on the MyFundedPerps sandbox; compare to backtest.
- **M7:** live evaluation, conservative size.

## Open questions (blocking full automation) — see `DESIGN.md §7`

1. Does MyFundedPerps expose an **API**, or manual UI only?
2. Do their rules **permit bots/automation**? (If not → pivot to a compliant signal/alert assistant.)
3. Target **account size** and **instruments**?
4. Preferred **execution timeframe** (default `15m`)?
5. Comfortable with the honest possibility that testing shows **no reliable edge**?
