"""Tests for confirmed prop RULES + the automation-policy fail-closed gate.

Run:  python3 -m unittest discover -s tests -v   (from project root)

These lock in the confirmed MyFundedPerps numbers (so a regression in the
defaults is caught) and verify that `Settings.require_ready_for_live()`:
  * arms an EVAL profile once every rule is confirmed and the config is ordinary,
  * still refuses configs that drift into a prohibited automation class (HFT).
"""
import sys
import unittest
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from liq_ai_bot.config import (  # noqa: E402
    AutomationPolicy, ConfigNotConfirmedError, Profile, PropRules, RiskConfig, Settings,
)


class TestConfirmedRules(unittest.TestCase):
    def test_dollar_values_on_50k(self):
        r = PropRules()  # defaults are now the confirmed values
        self.assertAlmostEqual(r.daily_loss_abs(), 1_000.0, places=2)
        self.assertAlmostEqual(r.max_drawdown_abs(), 2_000.0, places=2)
        self.assertAlmostEqual(r.phase1_target_abs(), 3_000.0, places=2)

    def test_rules_scale_with_account_size(self):
        for size, dll, mll, tgt in [
            (25_000, 500, 1_000, 1_500),
            (100_000, 2_000, 4_000, 6_000),
            (300_000, 6_000, 12_000, 18_000),
        ]:
            r = replace(PropRules(), account_size=float(size))
            self.assertAlmostEqual(r.daily_loss_abs(), dll, places=2)
            self.assertAlmostEqual(r.max_drawdown_abs(), mll, places=2)
            self.assertAlmostEqual(r.phase1_target_abs(), tgt, places=2)

    def test_static_drawdown_and_no_consistency_rule(self):
        r = PropRules()
        self.assertEqual(r.drawdown_type.value, "static")
        self.assertIsNone(r.best_day_cap)          # confirmed: no consistency rule
        self.assertIsNone(r.phase2_target)         # single-phase

    def test_all_rules_confirmed(self):
        r = PropRules()
        self.assertTrue(r.all_confirmed())
        self.assertEqual(r.unconfirmed_fields(), [])
        self.assertTrue(r.automation_allowed)


class TestAutomationPolicyGate(unittest.TestCase):
    def test_sandbox_never_blocks(self):
        # sandbox profile short-circuits — always safe to run offline
        Settings(profile=Profile.SANDBOX).require_ready_for_live()  # must not raise

    def test_eval_arms_when_confirmed_and_ordinary(self):
        s = Settings(profile=Profile.EVAL, execution_timeframe="15m")
        s.require_ready_for_live()  # must not raise now that all rules are confirmed

    def test_eval_refuses_sub_minute_timeframe(self):
        s = Settings(profile=Profile.EVAL, execution_timeframe="30s")
        with self.assertRaises(ConfigNotConfirmedError) as ctx:
            s.require_ready_for_live()
        self.assertIn("HFT", str(ctx.exception))

    def test_eval_refuses_when_a_rule_unconfirmed(self):
        rules = replace(PropRules(), confirmed_automation=False)
        s = Settings(profile=Profile.EVAL, rules=rules)
        with self.assertRaises(ConfigNotConfirmedError):
            s.require_ready_for_live()

    def test_policy_flags_many_concurrent_positions(self):
        s = Settings(profile=Profile.EVAL, risk=RiskConfig(max_concurrent_positions=10))
        problems = s.automation.violations(s)
        self.assertTrue(any("HFT-like" in p for p in problems))

    def test_ordinary_config_has_no_violations(self):
        s = Settings(profile=Profile.EVAL, execution_timeframe="1h")
        self.assertEqual(s.automation.violations(s), [])

    def test_policy_bans_are_all_set(self):
        p = AutomationPolicy()
        self.assertTrue(p.documented_api_only)
        self.assertTrue(p.forbid_internal_browser_endpoints)
        self.assertTrue(p.forbid_hft)
        self.assertTrue(p.forbid_latency_arbitrage)
        self.assertTrue(p.forbid_platform_exploitation)
        self.assertTrue(p.forbid_cross_account_coordination)
        self.assertTrue(p.forbid_risk_limit_circumvention)


if __name__ == "__main__":
    unittest.main(verbosity=2)
