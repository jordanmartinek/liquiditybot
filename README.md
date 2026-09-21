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
  config.py                   Settings + PropRules (placeholder rules + confirm flags), RiskConfig
  types.py                    Bar, Level, Setup, DOLState, enums
  strategy.py                 deterministic core: level tracking, DOL gravity model, SFP trigger
  risk_engine.py              THE hard-constraint layer (two DD trackers, daily stop, kill-switch)
  backtest.py                 event-driven harness + Monte-Carlo pass/breach simulator
examples/demo.py              end-to-end smoke test on synthetic data (no network)
tests/test_risk_engine.py     unit tests for the risk guards + the "never adds risk" invariant
```

## Status: M1 (scaffold)

Runnable, tested, **offline, no live connectivity, no real money**. The strategy's
level-seeding (PDH/PDL, sessions, equal highs/lows, HTF swings, full confluence
model) and the real data feed are **stubbed for M2**. The `RULES` are conservative
**placeholders** — every rule is flagged `confirmed=False` and the `EVAL`/`FUNDED`
profiles refuse to arm until you fill them from the firm's live docs.

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

`requirements.txt` / the `pyproject.toml` optional-deps list the **target** stack
for M2+ (pandas, numpy, ccxt, pydantic; scikit-learn/lightgbm for the ML filter).
Install those only in an environment with PyPI access — they are **not** imported
by the M1 code.

## Roadmap

- **M1 (done):** scaffold — config/RULES, risk engine, strategy interfaces, backtest + Monte-Carlo, tests.
- **M2 (in progress):** real historical perp data feed (`datafeed.py`: csv/ccxt/synthetic) + real-data backtest runner + strategy/datafeed tests **[done]**; port full level/confluence logic from the Pine indicator **[next]**.
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
