# Liquidity Strategy → Automated Trading System (MyFundedPerps R&D)

**Status:** design / R&D. This is a *separate* project from the `LiquidityRadar.pine` indicator.
**Goal:** translate the liquidity-trading logic (levels, sweeps/SFP, DOL bias, confluence) into a rules-first automated system engineered to pass and hold a **MyFundedPerps** crypto-perps evaluation.

> ⚠️ **Read this first.** No strategy is guaranteed to be profitable or to pass a challenge. This document is an engineering + validation framework, not a promise. Nothing here is financial advice. Deploy real money only if honest, out-of-sample testing supports it — and only after you have personally verified the firm's current rules and their policy on automation/API access.

---

## 0. Non-negotiable prerequisites (do these before writing trading logic)

The risk/compliance layer matters more than the entry logic. Two things must be confirmed **from MyFundedPerps' own live docs/support**, because the whole system is built around them and I will not guess them:

1. **Is automated / API / bot trading permitted?** MyFundedPerps is a **paper-trading simulation** for crypto perpetuals (trades run against real market data and order books, but fills/balances/payouts are simulated on their platform — per their [documentation](https://docs.myfundedperpetuals.com/)). Confirm:
   - Do they expose an **API** (or only a manual web/trading UI)? If no API, "full automation" may require UI automation (fragile, often against ToS) or a semi-automated "signal + 1-click" workflow.
   - Do their rules **allow EAs/bots**, and what specific behaviors get an account closed (copy-trading, latency arbitrage, exploiting sim fills, HFT, "gambling"/all-in, etc.)?
2. **The exact rule numbers**, per plan/phase. Fill the `RULES` table below from their docs. Related perpetuals programs commonly use: **daily drawdown that resets at 00:00 UTC**, a **static or trailing max drawdown floor**, a **max single-trade loss** on funded accounts, and a **"Best Day" / consistency cap** on payout (e.g. your best day can't exceed ~20% of total profit). Treat these as *likely shapes to verify*, not confirmed values.

Until #1 is answered, build against the **paper/sandbox** only. Until #2 is filled, the risk engine uses conservative placeholders.

### RULES (fill from official docs — placeholders shown)
| Parameter | Placeholder | Confirmed value |
|---|---|---|
| Account sizes | e.g. 25k / 50k / 100k … up to 300k allocation | ? |
| Phase 1 profit target | ~8–10% | ? |
| Phase 2 profit target (if 2-step) | ~5–6% | ? |
| Max overall drawdown | ~6–12% (static or trailing?) | ? |
| Daily loss limit | ~4–5% (reset time? 00:00 UTC?) | ? |
| Max single-trade loss (funded) | ? | ? |
| Min trading days | ? | ? |
| Consistency / "Best Day" rule | best day ≤ ~20% of total profit? | ? |
| Time limit per phase | often none | ? |
| Payout split / cadence | up to 90%? | ? |
| Automation / API policy | **must confirm** | ? |
| Instruments allowed | which perps (BTC/ETH majors?) | ? |

---

## 1. Where "AI" genuinely helps — and where it's hype

Being honest about this up front, because "AI trading bot" is an overloaded phrase.

**The core execution should NOT be a black-box neural net.** A deterministic, rule-based engine that mirrors the liquidity strategy is:
- auditable (you can see exactly why every trade fired),
- testable (backtest = live behavior),
- compliant (predictable, no rogue behavior that trips prop rules).

**Where machine learning / "AI" adds real value (as filters/optimizers around the rules, not as the trigger):**
1. **Setup-quality classifier.** Given a candidate SFP/level setup and its features (confluence score, DOL alignment, wick strength, session, rvol, distance-to-target), predict P(reach target before stop). Trade only setups above a learned probability threshold. This is exactly what the indicator's backtest harness was built to feed.
2. **Regime detection.** Cluster market state (trending / ranging / high-vol) so the bot sizes down or stands aside in conditions where the edge historically decays.
3. **Adaptive parameter selection.** Walk-forward-tuned parameters (swing length, cluster distance, R-multiple) chosen per regime instead of fixed.
4. **Position-sizing / risk optimization.** RL or convex optimization to allocate the daily risk budget across the day to maximize P(pass) subject to the hard drawdown constraint — a genuinely good fit for the prop-challenge objective.

**Rule of thumb:** rules decide *whether* to trade; ML decides *whether this instance of the rule is worth taking* and *how much to risk*. The hard risk limits are never ML — they're deterministic guards.

---

## 2. Strategy, formalized (the deterministic core)

Direct port of the indicator's proven logic into executable rules.

### Levels (the map)
- Track: PDH/PDL, PWH/PWL, PMH/PML, session H/L (Asian/London/NY), swing highs/lows (+HTF swings), equal highs/lows. Remove once swept.
- Each level carries: price, type, birth bar, side (BSL above / SSL below), and a **confluence score (0–100)**.

### Entry: sweep + reversal (SFP), the validity trigger
A long setup fires when, on a closed bar:
1. price **swept** a tracked SSL level (wick pierced below it) and **closed back above** it;
2. **volume confirmation**: rvol ≥ threshold;
3. **DOL agreement**: the DOL bias points up (buy-side liquidity is the dominant draw) — this is the higher-timeframe context filter;
4. **confluence gate**: the swept level's confluence score ≥ threshold;
5. optional ML gate (§1.1): P(win) ≥ threshold.

Short setup is the mirror.

### Exit
- **Stop:** beyond the sweep wick (± padding × ATR). This defines 1R.
- **Target:** primary = next opposing liquidity pool (the DOL magnet / highest per-level confidence on the other side); fallback = fixed R multiple. Optionally scale out (TP1 at 1R, runner to pool).
- **Invalidations:** close beyond stop, or setup context breaks (structure shift).

### Bias / filters (context)
- DOL gravity model → directional bias + conviction. Only take setups aligned with bias above a conviction floor.
- Session filter: prefer London/NY killzones (where sweeps are statistically cleaner); optionally avoid dead hours.

Everything above already exists as logic in the indicator — the bot re-implements it in Python so backtest = live.

---

## 3. System architecture

```
┌─────────────┐   market data (REST/WS, real order book)
│  DATA FEED  │───────────────────────────────┐
└─────────────┘                                ▼
                                     ┌──────────────────────┐
                                     │   SIGNAL ENGINE       │  deterministic strategy (§2)
                                     │  levels · SFP · DOL   │  → candidate setups + features
                                     │  · confluence         │
                                     └──────────┬───────────┘
                                                ▼
                                     ┌──────────────────────┐
                                     │  ML FILTER (optional) │  P(win) / regime / sizing (§1)
                                     └──────────┬───────────┘
                                                ▼
                            ┌───────────────────────────────────────┐
                            │  RISK / COMPLIANCE LAYER (hard guards) │  ← THE MOST IMPORTANT PART
                            │  • per-trade risk ≤ X% (0.25–0.5%)     │
                            │  • personal daily stop = ½ firm limit  │
                            │  • block trade if it could breach      │
                            │    daily loss or max drawdown          │
                            │  • max concurrent positions / exposure │
                            │  • consistency guard (cap best day)    │
                            │  • min-trading-day pacing              │
                            │  • kill-switch on rule proximity       │
                            └──────────────────┬────────────────────┘
                                                ▼
                                     ┌──────────────────────┐
                                     │   EXECUTION ADAPTER   │  paper (MyFundedPerps) │ exchange sandbox
                                     └──────────┬───────────┘
                                                ▼
                            ┌───────────────────────────────────────┐
                            │  STATE / MONITORING / JOURNAL          │  equity, drawdown, open risk,
                            │  logs every decision + outcome (→ ML)  │  alerts, daily reset at 00:00 UTC
                            └───────────────────────────────────────┘
```

### Design principles
- **The risk layer can only ever REDUCE or BLOCK, never add risk.** It wraps the signal engine; a valid signal that would violate any prop constraint is silently skipped and logged.
- **Two independent drawdown trackers:** the firm's official calc *and* a stricter internal one (personal daily stop = half the firm's daily limit; internal max-DD buffer well inside the firm's). The bot stops trading for the day at the *internal* limit, giving margin against slippage/latency.
- **Fail-safe defaults:** on any data gap, disconnect, or ambiguous state → flatten/така no new entries.
- **Every decision is journaled** (setup, features, size, outcome) — this dataset trains the ML filter and validates the edge.

### Tech stack (proposed)
- **Python 3.11+**, `pandas`/`numpy`, `ccxt` (exchange-agnostic market data & — if permitted — order routing), `pydantic` for config, `pytest` for tests. ML later: `scikit-learn` / `lightgbm` (keep it simple and explainable before anything deep).
- Config-driven `RULES` object so the same code targets sandbox → challenge → funded by swapping a profile.
- Event-driven loop (on closed bar) rather than tick HFT — matches the strategy and avoids the "toxic flow / HFT" behaviors prop firms ban.

---

## 4. Validation plan (this is where an edge is proven or killed)

No live money until each gate passes. Order matters.

1. **Backtest (in-sample)** on historical perp data for the target instruments. Metrics: win rate, avg R, expectancy, max drawdown, worst day, distribution of daily P&L. **Gate:** positive expectancy after costs.
2. **Prop-challenge simulation.** Replay the backtest *through the risk/compliance layer* with the firm's exact rules. Measure the metric that actually matters: **P(pass phase 1) and P(pass without breaching)** across many randomized start dates (Monte Carlo over the equity path). **Gate:** acceptable pass probability *and* low breach probability.
3. **Walk-forward / out-of-sample.** Re-tune only on past data, test on unseen future windows. Guards against curve-fitting. **Gate:** OOS expectancy ≈ in-sample (no collapse).
4. **Costs & realism.** Model perp funding, taker/maker fees, slippage, and — since this is a *simulated fill* platform — how their sim fills differ from a real book. **Gate:** edge survives realistic costs.
5. **Paper-trade live** on the actual MyFundedPerps sandbox for weeks. **Gate:** live results track the backtest (no regime break, no execution surprises).
6. **Only then**, a real evaluation, sized conservatively.

**Honesty checkpoint:** if steps 1–2 show the base strategy doesn't clear costs or has an unacceptable breach rate, the answer is to fix/retire the strategy — not to add ML lipstick or over-optimize. The backtest harness we built in the indicator is the seed of this; expect to *prune* factors that don't earn their place.

---

## 5. Prop-challenge–specific tactics (engineering the pass, not just the edge)

- **Risk 0.25–0.5% per trade**, personal daily stop at **half** the firm's daily loss limit, and spread the profit target across **many sessions** (avoids consistency-rule failure and reduces single-day blowup risk). (This is standard, widely-recommended prop discipline; see e.g. [fundedtrading.com](https://fundedtrading.com/how-to-pass-a-prop-firm-challenge/) — *content rephrased for compliance*.)
- **Consistency guard:** if a would-be win pushes the day's profit past the firm's best-day cap, scale the target down / bank partial so no single day dominates.
- **Trailing vs static drawdown** changes everything — if trailing intraday, the bot must treat unrealized peak equity as the drawdown anchor and never let an open position's drawup create a fragile floor.
- **Pace to min trading days**: don't hit the target in 2 days if the firm requires more; the bot schedules participation.

---

## 6. Roadmap

- **M0 (this doc):** design + prerequisites.
- **M1:** Python scaffold — config/RULES, risk-engine skeleton, strategy interfaces, backtest harness stub (see `/src`). No live connectivity.
- **M2 (done):** ported the deterministic strategy from the Pine logic — full level engine (`levels.py`: PDH/PDL, PWH/PWL, PMH/PML, sessions, swings +HTF, equal highs/lows, order blocks, dealing-range OTE, 0-100 confluence) wired into the SFP strategy; real data feed (`datafeed.py`) + backtest runner for historical perp data.
- **M3:** prop-challenge Monte-Carlo simulation through the risk layer → P(pass)/P(breach).
- **M4:** walk-forward + cost modeling; decide go/no-go on the base edge.
- **M5 (only if M4 passes):** optional ML setup-filter trained on the journaled dataset.
- **M6:** paper-trade on MyFundedPerps sandbox; compare to backtest.
- **M7:** live evaluation, conservative size.

---

## 7. Open questions for you
1. Does MyFundedPerps expose an **API** for order placement, or is it a manual UI only? (Determines full-auto vs semi-auto.)
2. Do their rules **permit bots/automation**? (If not, we pivot to a *signal/alert assistant* you execute manually — still hugely useful, fully compliant.)
3. Which **account size** and **instruments** are you targeting?
4. Preferred **timeframe** for the strategy (the indicator is TF-agnostic; the bot needs a chosen execution TF, e.g. 5m/15m)?
5. Are you comfortable with the honest possibility that **testing shows no reliable edge**, in which case we don't deploy?
