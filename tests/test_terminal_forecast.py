"""Independent forecast-budget checks and split/issue-date isolation."""
from datetime import date, timedelta
from types import SimpleNamespace
import unittest

import numpy as np

from harl.envs.cascade_reservoir.terminal_forecast import (
    calendar_slot, fit_seasonal_external_flow, TerminalForecastBudget)


class TerminalForecastTest(unittest.TestCase):
    def test_fit_excludes_held_out_flows_and_freezes_profile(self):
        dates = tuple(date(2000, 1, 1)+timedelta(days=i) for i in range(731))
        flows = np.full((731, 5), 10.)
        flows[366:] = 1e8
        profile = fit_seasonal_external_flow(dates, flows, [2000])
        np.testing.assert_array_equal(profile, np.full((366, 5), 10.))
        self.assertFalse(profile.flags.writeable)
        flows[366:] = 0.
        np.testing.assert_array_equal(profile, fit_seasonal_external_flow(dates, flows, [2000]))

    def test_month_day_slots_do_not_shift_after_leap_day(self):
        self.assertEqual(calendar_slot(date(1999, 3, 1)), calendar_slot(date(2000, 3, 1)))
        self.assertEqual(calendar_slot(date(2000, 2, 29)), 59)

    def test_budget_matches_separate_daily_sum_across_source_year_join(self):
        dates = tuple([date(1999, 12, 27)+timedelta(days=i) for i in range(5)]+
                      [date(1970, 1, 1)+timedelta(days=i) for i in range(15)])
        profile = np.arange(366*5, dtype=float).reshape(366, 5)+1.
        data = SimpleNamespace(dates=dates, seasonal_external_flow_m3s=profile)
        budget = TerminalForecastBudget(data, 0, len(dates), 86400.)
        issued = np.arange(5, dtype=float)+100.
        for index in (0, 4, 11, 18, 19):
            expected = np.zeros(5)
            for ahead, day in enumerate(dates[index+1:], start=1):
                expected += issued if ahead <= 7 else profile[calendar_slot(day)]
            np.testing.assert_allclose(budget.future_volume(index, issued), expected*86400., rtol=0., atol=0.)

    def test_later_actuals_and_later_forecast_issues_are_not_read(self):
        dates = tuple(date(2000, 1, 1)+timedelta(days=i) for i in range(20))
        data = SimpleNamespace(dates=dates, seasonal_external_flow_m3s=np.full((366, 5), 20.),
                               external=np.zeros((20, 5)), inflow_forecast=np.ones((20, 5)))
        budget = TerminalForecastBudget(data, 0, 20, 86400.)
        expected = budget.future_volume(0, data.inflow_forecast[0])
        data.external[1:] = 1e8
        data.inflow_forecast[1:] = 1e8
        np.testing.assert_array_equal(budget.future_volume(0, data.inflow_forecast[0]), expected)
        self.assertTrue(np.all(budget.future_volume(0, data.inflow_forecast[0]+1.) > expected))

    def test_invalid_forecast_is_rejected(self):
        data = SimpleNamespace(dates=(date(2000, 1, 1),),
                               seasonal_external_flow_m3s=np.zeros((366, 5)))
        budget = TerminalForecastBudget(data, 0, 1, 86400.)
        np.testing.assert_array_equal(budget.future_volume(0, np.ones(5)), np.zeros(5))
        for value in (np.full(5, -1.), np.full(5, np.nan), np.ones(4)):
            with self.assertRaises(ValueError):
                budget.future_volume(0, value)


if __name__ == '__main__':
    unittest.main()
