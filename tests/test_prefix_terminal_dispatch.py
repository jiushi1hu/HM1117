"""Hand-calculated prefix budgets and command-preserving forecast rejection."""
from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from harl.envs.cascade_reservoir.dispatch_kernel import dispatch_release


class PrefixTerminalDispatchTest(unittest.TestCase):
    def dispatch(self, storage, external, actions, terminal=None, lower=None, upper=None,
                 cap=None, dt=1.):
        storage = np.asarray(storage, dtype=float)
        result = dispatch_release(storage[None], np.asarray(external, dtype=float)[None],
            np.asarray(actions, dtype=float)[None],
            (np.zeros(5) if lower is None else np.asarray(lower, dtype=float))[None],
            (np.full(5, 20.) if upper is None else np.asarray(upper, dtype=float))[None],
            np.full(5, 10.) if cap is None else np.asarray(cap, dtype=float), dt, terminal)
        self.assertTrue(np.all(result['release'] >= -1e-8))
        return result

    def test_upstream_future_water_reaches_every_outlet_budget(self):
        terminal = dict(target=np.full(5, 10.), remaining_days=2,
            forecast=np.array([2., 0., 0., 0., 0.]),
            future_external_volume_m3=np.array([2., 0., 0., 0., 0.]))
        result = self.dispatch(np.full(5, 10.), [2., 0., 0., 0., 0.], np.full(5, .5), terminal)
        np.testing.assert_array_equal(result['terminal_window_high_m3s'][0], np.full(5, 4.))
        np.testing.assert_array_equal(result['release'][0], np.ones(5))
        np.testing.assert_array_equal(result['controls'][0], np.ones(5))
        self.assertFalse(result['terminal_plan_rejected'][0])

    def test_upstream_target_restoration_is_included_in_prefix_budget(self):
        # One cubic metre must stay at reservoir zero; its future water cannot
        # all be counted as available to every downstream reservoir.
        result = self.dispatch([9., 10., 10., 10., 10.], [1., 0., 0., 0., 0.],
            np.full(5, .5), dict(target=np.full(5, 10.), remaining_days=2,
                forecast=np.zeros(5), future_external_volume_m3=[2., 0., 0., 0., 0.]))
        np.testing.assert_array_equal(result['terminal_window_high_m3s'][0], np.full(5, 2.))

    def test_downstream_terminal_window_restricts_upstream_before_action_execution(self):
        # Today S1 must remain five, so q1=q0. The prefix budget at outlet1 is
        # zero, forcing q0=0 even though outlet0's own terminal range allows two.
        result = self.dispatch(np.full(5, 5.), np.zeros(5), np.ones(5),
            dict(target=[3., 7., 5., 5., 5.], remaining_days=2,
                 forecast=np.zeros(5), future_external_volume_m3=np.zeros(5)),
            lower=[0., 5., 0., 0., 0.], upper=[10., 5., 10., 10., 10.])
        self.assertFalse(result['terminal_plan_rejected'][0])
        self.assertEqual(result['terminal_window_high_m3s'][0, 0], 2.)
        self.assertEqual(result['high'][0, 0], 0.)
        np.testing.assert_array_equal(result['release'][0], np.zeros(5))

    def test_rejected_nonfinal_joint_plan_preserves_physical_actor_choices(self):
        # A hundred cubic metres must leave outlet0 over only two decisions,
        # whose capacity totals twenty. Reject this forecast plan for all five
        # outlets; do not turn an arbitrary command into forced zero release.
        terminal = dict(target=np.full(5, 10.), remaining_days=2,
            forecast=np.zeros(5), future_external_volume_m3=[100., 0., 0., 0., 0.])
        for action in (0., .5, 1.):
            with self.subTest(action=action):
                result = self.dispatch(np.full(5, 10.), np.zeros(5), np.full(5, action), terminal)
                baseline = self.dispatch(np.full(5, 10.), np.zeros(5), np.full(5, action))
                self.assertTrue(result['terminal_plan_rejected'][0])
                self.assertFalse(result['terminal_fallback'][0])
                for name in ('release', 'inflow', 'low', 'high', 'controls', 'actions'):
                    np.testing.assert_array_equal(result[name], baseline[name])
                self.assertEqual(result['release'][0, 0], 10.*action**2)

    def test_feasible_final_day_collapses_to_exact_target(self):
        storage = np.array([9., 10., 10., 10., 10.])
        target = np.array([10., 9., 10., 10., 10.])
        terminal = dict(target=target, remaining_days=1, forecast=np.ones(5),
                        future_external_volume_m3=np.zeros(5))
        for action in (0., 1.):
            result = self.dispatch(storage, [2., 1., 0., 0., 0.], np.full(5, action), terminal)
            np.testing.assert_array_equal(result['release'][0], [1., 3., 3., 3., 3.])
            np.testing.assert_array_equal(storage+result['inflow'][0]-result['release'][0], target)
            np.testing.assert_array_equal(result['controls'][0], np.zeros(5))
            self.assertFalse(result['terminal_plan_rejected'][0])
            self.assertFalse(result['terminal_fallback'][0])

    def test_unreachable_final_target_is_safe_and_command_independent(self):
        storage = np.array([0., 10., 10., 10., 10.])
        target = np.array([1., 10., 10., 10., 10.])
        results = [self.dispatch(storage, np.zeros(5), np.full(5, action),
            dict(target=target, remaining_days=1, forecast=np.zeros(5))) for action in (0., 1.)]
        for result in results:
            self.assertTrue(result['terminal_plan_rejected'][0])
            self.assertTrue(result['terminal_fallback'][0])
            np.testing.assert_array_equal(result['controls'][0], np.zeros(5))
            np.testing.assert_array_equal(result['release'][0], np.zeros(5))
            np.testing.assert_array_equal(storage+result['inflow'][0]-result['release'][0], storage)
            self.assertNotEqual(storage[0], target[0])
        np.testing.assert_array_equal(results[0]['release'], results[1]['release'])

    def test_missing_upstream_forecast_does_not_artificially_force_bht_zero(self):
        # Old local BHT arithmetic gave 20 + today_inflow + 9*1 -100 <0,
        # despite a continuing ten-unit upstream water supply. Prefix accounting
        # gives outlet1 a thirty-unit budget and preserves its five-unit release.
        result = self.dispatch([100., 20., 100., 100., 100.], [10., 1., 0., 0., 0.],
            np.full(5, .5), dict(target=np.full(5, 100.), remaining_days=10,
                forecast=[10., 1., 0., 0., 0.]), upper=np.full(5, 200.), cap=np.full(5, 20.))
        self.assertEqual(result['terminal_window_high_m3s'][0, 1], 30.)
        self.assertEqual(result['release'][0, 1], 5.)
        self.assertEqual(result['controls'][0, 1], 1.)
        self.assertFalse(result['terminal_plan_rejected'][0])
        self.assertEqual(result['terminal_forecast_model'][0], 'constant_next7d_legacy_fallback')

    def test_final_fallback_keeps_tolerance_sized_physical_inversion_nonnegative(self):
        # A tolerated 5e-9 m3/s interval inversion at station zero
        # must still execute zero, even when another station rejects the plan.
        result = self.dispatch([-5e-9, 10., 10., 10., 10.], np.zeros(5), np.ones(5),
            dict(target=[0., 10., 10., 10., 30.], remaining_days=1,
                 forecast=np.zeros(5)))
        self.assertTrue(result['terminal_fallback'][0])
        self.assertEqual(result['release'][0, 0], 0.)
        np.testing.assert_array_equal(result['controls'][0], np.zeros(5))

    def test_no_terminal_mapping_retains_hand_calculated_quadratic_commands(self):
        result = self.dispatch(np.full(5, 10.), [2., 1., 1., 1., 1.], [0., .25, .5, .75, 1.])
        np.testing.assert_array_equal(result['release'][0], [0., .625, 2.5, 5.625, 10.])
        np.testing.assert_array_equal(result['low'], result['physical_low_m3s'])
        np.testing.assert_array_equal(result['high'], result['physical_high_m3s'])

    def test_provided_volume_overrides_legacy_constant_forecast_and_rejects_invalid_context(self):
        terminal = dict(target=np.full(5, 10.), remaining_days=2, forecast=np.full(5, 999.),
                        future_external_volume_m3=np.zeros(5))
        result = self.dispatch(np.full(5, 10.), np.zeros(5), np.ones(5), terminal)
        np.testing.assert_array_equal(result['terminal_window_high_m3s'], np.zeros((1, 5)))
        self.assertEqual(result['terminal_forecast_model'][0], 'provided_future_external_volume')
        for changes in ({'future_external_volume_m3': [-1., 0., 0., 0., 0.]},
                        {'remaining_days': 1, 'future_external_volume_m3': np.ones(5)},
                        {'remaining_days': 0}, {'forecast': np.full(5, np.nan)}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                self.dispatch(np.full(5, 10.), np.zeros(5), np.ones(5), dict(terminal, **changes))


if __name__ == '__main__':
    unittest.main(verbosity=2)
