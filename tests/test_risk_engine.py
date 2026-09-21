"""Unit tests for the risk/compliance layer (stdlib unittest, no deps).

Run:  python3.11 -m unittest discover -s tests -v   (from project root)
The conftest-free sys.path shim lets it run without installation.
"""
import sys
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from liq_ai_bot.config import DrawdownType, PropRules, RiskConfig, Settings  # noqa: E402
from liq_ai_bot.risk_engine import (  # noqa: E402
    AccountState, Decision, ProposedTrade, RiskEngine,
)


def make_engine(**rule_overrides):
    rules = PropRules(
        account_size=50_000.0,
        max_drawdown=0.10,
        daily_loss_limit=0.05,
        drawdown_type=DrawdownType.STATIC,
        **rule_overrides,
    )
    settings = Settings(rules=rules, risk=RiskConfig())
    state = AccountState.new(rules.account_size)
    return RiskEngine(settings, state)


class TestSizing(unittest.TestCase):
    def test_size_matches_per_trade_risk(self):
        eng = make_engine()
        # 0.35% of 50k = $175 risk; stop 100 away => 1.75 units
        t = ProposedTrade("BTC", "long", entry=30_000, stop=29_900)
        v = eng.vet(t)
        self.assertTrue(v.approved)
        self.assertAlmostEqual(v.risk_abs, 175.0, places=2)
        self.assertAlmostEqual(v.approved_size, 1.75, places=4)

    def test_never_exceeds_configured_risk(self):
        """Even if the trade asks for more, the engine caps at configured max."""
        eng = make_engine()
        t = ProposedTrade("BTC", "long", entry=30_000, stop=29_900,
                          intended_risk_frac=0.05)  # asks for 5%!
        v = eng.vet(t)
        # capped to 0.35% -> $175, not $2500
        self.assertLessEqual(v.risk_abs, 175.0 + 1e-6)


class TestGuardsBlock(unittest.TestCase):
    def test_zero_distance_stop_raises(self):
        eng = make_engine()
        t = ProposedTrade("BTC", "long", entry=30_000, stop=30_000)
        with self.assertRaises(ValueError):
            eng.vet(t)

    def test_automation_forbidden_blocks(self):
        eng = make_engine(automation_allowed=False)
        t = ProposedTrade("BTC", "long", entry=30_000, stop=29_900)
        v = eng.vet(t)
        self.assertEqual(v.decision, Decision.BLOCK)
        self.assertEqual(v.approved_size, 0.0)

    def test_max_concurrent_blocks(self):
        eng = make_engine()
        t = ProposedTrade("BTC", "long", entry=30_000, stop=29_900)
        v = eng.vet(t)
        eng.register_open(t, v.approved_size)
        # second concurrent trade blocked (max_concurrent_positions=1)
        v2 = eng.vet(ProposedTrade("ETH", "long", entry=2000, stop=1990))
        self.assertEqual(v2.decision, Decision.BLOCK)


class TestDailyAndDrawdown(unittest.TestCase):
    def test_daily_loss_reduces_then_halts(self):
        eng = make_engine()
        # Internal daily budget = 50% of firm 5% of 50k = $1250.
        self.assertAlmostEqual(eng.internal_daily_budget(), 1250.0, places=2)
        # Book a loss that eats most of the daily budget.
        eng.on_fill_closed(-1100.0)   # $150 left of internal budget
        t = ProposedTrade("BTC", "long", entry=30_000, stop=29_900)
        v = eng.vet(t)
        # remaining budget (150) < kill-switch fraction (15% of 1250 = 187.5) -> halt
        self.assertEqual(v.decision, Decision.BLOCK)
        self.assertTrue(eng.state.halted_today)

    def test_reduce_when_budget_tight(self):
        eng = make_engine()
        # Leave ~ $300 of daily budget: desired risk 175 fits, but push DD budget low.
        eng.on_fill_closed(-950.0)   # 300 left of internal daily budget (1250-950)
        t = ProposedTrade("BTC", "long", entry=30_000, stop=29_900)
        v = eng.vet(t)
        # 300 remaining > kill fraction 187.5, desired risk < 300 -> APPROVE full.
        # NOTE: risk sizes off CURRENT balance ($49,050 after the loss), so the
        # desired risk is 0.35% * 49_050 = $171.675, NOT $175. This is correct:
        # per-trade risk always tracks live balance.
        self.assertTrue(v.approved)
        self.assertAlmostEqual(v.risk_abs, 49_050.0 * 0.0035, places=2)

    def test_day_roll_resets_halt(self):
        eng = make_engine()
        eng.on_fill_closed(-1200.0)
        base = datetime.now(timezone.utc).replace(hour=12)
        eng.vet(ProposedTrade("BTC", "long", 30_000, 29_900), now=base)
        self.assertTrue(eng.state.halted_today)
        # next day
        eng.on_time(base + timedelta(days=1))
        self.assertFalse(eng.state.halted_today)


class TestInvariantNeverAddsRisk(unittest.TestCase):
    def test_approved_risk_never_exceeds_desired(self):
        """Fuzz: across many stop distances the approved risk is always <= the
        configured per-trade risk (the engine can only reduce/block)."""
        eng = make_engine()
        desired = eng.state.balance * eng.risk.per_trade_risk
        for dist in range(10, 2000, 37):
            fresh = make_engine()
            v = fresh.vet(ProposedTrade("BTC", "long", 30_000, 30_000 - dist))
            self.assertLessEqual(v.risk_abs, desired + 1e-6,
                                 f"risk exceeded desired at dist={dist}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
