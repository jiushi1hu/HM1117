"""Issued near-term forecasts plus a frozen training-only seasonal tail.

Only the calendar of later decisions is inspected at runtime. Later observed
flows and later-issued forecast rows never enter a current decision's budget.
This is an expected external-water budget, not a robust reachability guarantee.
"""
from datetime import date

import numpy as np


TERMINAL_FORECAST_MODEL = 'issued_next7d_mean_then_training_seasonal_mean_v1'
TERMINAL_MECHANISM = 'cascade_prefix_seasonal_reserve_v2'


def calendar_slot(day):
    """Use a leap-year month/day axis so March never shifts between years."""
    return (date(2000, day.month, day.day)-date(2000, 1, 1)).days


def fit_seasonal_external_flow(dates, external, training_years):
    """Fit a circular 31-day-smoothed mean using training years exclusively."""
    values = np.asarray(external, dtype=np.float64)
    if (values.shape != (len(dates), 5) or not np.isfinite(values).all()
            or (values < 0.).any()):
        raise ValueError('Seasonal fitting requires finite nonnegative daily five-vectors')
    allowed = set(training_years)
    total, count = np.zeros((366, 5)), np.zeros(366, dtype=np.int64)
    for day, row in zip(dates, values):
        if day.year in allowed:
            slot = calendar_slot(day)
            total[slot] += row
            count[slot] += 1
    if not count.all():
        raise ValueError('Training years must cover every month/day, including February 29')
    means = total/count[:, None]
    seasonal = sum(np.roll(means, shift, axis=0) for shift in range(-15, 16))/31.
    seasonal.setflags(write=False)
    return seasonal


class TerminalForecastBudget:
    """O(1) remaining-volume queries on an episode's known source calendar."""

    def __init__(self, data, start, stop, timestep_seconds):
        self.start, self.stop = int(start), int(stop)
        self.dt = float(timestep_seconds)
        profile = np.asarray(data.seasonal_external_flow_m3s, dtype=np.float64)
        if (profile.shape != (366, 5) or not np.isfinite(profile).all()
                or (profile < 0.).any()):
            raise ValueError('Terminal forecasting requires a valid training seasonal profile')
        seasonal = profile[[calendar_slot(day) for day in data.dates[start:stop]]]
        self.tail = np.vstack((np.cumsum(seasonal[::-1], axis=0)[::-1], np.zeros((1, 5))))
        self.tail.setflags(write=False)

    def future_volume(self, index, issued_next7d_mean):
        """Expected external volume AFTER today's decision, in cubic metres."""
        if not self.start <= index < self.stop:
            raise ValueError('Forecast query must be inside the episode')
        issued = np.asarray(issued_next7d_mean, dtype=np.float64)
        if issued.shape != (5,) or not np.isfinite(issued).all() or (issued < 0.).any():
            raise ValueError('Issued forecast must be a finite nonnegative five-vector')
        offset = index-self.start
        near_days = min(7, self.stop-index-1)
        return self.dt*(near_days*issued+self.tail[offset+1+near_days])
