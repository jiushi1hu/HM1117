"""Decision-time forecasts and terminal targets are visible to value estimation."""
import copy
from datetime import date
from pathlib import Path
import sys
import unittest

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from harl.envs.cascade_reservoir.cascade_reservoir_env import (
    ACTOR_OBSERVATION_LAYOUT, CRITIC_STATE_LAYOUT, critic_state_size)
from harl.envs.cascade_reservoir.data_loader import CascadeData
from harl.envs.cascade_reservoir.episodes import episode_environment


class CascadeObservationContextTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.physical = yaml.safe_load(
            (ROOT/'harl/configs/envs_cfgs/cascade_reservoir.yaml').read_text())
        cls.data = CascadeData(ROOT/'data/cascade_reservoir_new', cls.physical)

    def environment(self, limits, source=None, terminal=True, years=None, days=4):
        physical = dict(self.physical, use_daily_level_limits=limits,
                        terminal_storage_equal_initial=terminal)
        return episode_environment(self.data if source is None else source, physical,
                                   [1965] if years is None else years,
                                   days=days, action_bins=11)

    def assert_context(self, env):
        state, obs = env.state(), env.actor_observations()
        forecast = ((env.data.inflow_forecast[env.index]-self.data.flow_mean)/
                    self.data.flow_std).astype(np.float32)
        target = (np.zeros(5, dtype=np.float32) if env.terminal_target is None else
                  ((env.terminal_target-self.data.vmin)/
                   (self.data.vmax-self.data.vmin)).astype(np.float32))
        volume = (env._future_external_volume()/env._future_volume_scale).astype(np.float32)
        np.testing.assert_array_equal(state[-15:-10], forecast)
        np.testing.assert_array_equal(state[-10:-5], target)
        np.testing.assert_array_equal(state[-5:], volume)
        np.testing.assert_array_equal(obs[:, -2], forecast)
        np.testing.assert_array_equal(obs[:, -3], target)
        np.testing.assert_array_equal(obs[:, -1], volume)

    def test_spaces_metadata_and_preserved_physical_prefix(self):
        for limits, actor_size, state_size in ((True, 18, 53), (False, 14, 33)):
            with self.subTest(daily_limits=limits):
                env = self.environment(limits)
                obs, shared, available = env.reset()
                self.assertEqual(critic_state_size(limits), state_size)
                self.assertEqual(obs.shape, (5, actor_size))
                self.assertEqual(shared.shape, (5, state_size))
                self.assertIsNone(available)
                self.assertTrue(all(space.shape == (actor_size,) for space in env.observation_space))
                self.assertTrue(all(space.shape == (state_size,) for space in env.share_observation_space))
                metadata = env.observation_metadata()
                self.assertEqual(metadata['actor_observation_layout'], ACTOR_OBSERVATION_LAYOUT)
                self.assertEqual(metadata['critic_state_layout'], CRITIC_STATE_LAYOUT)
                self.assertEqual(metadata['critic_state_size'], state_size)
                self.assertEqual(metadata['actor_observation_size'], actor_size)
                self.assertEqual(env.observation_features[-3:],
                                 ('terminal_target_storage_fraction',
                                  'next7d_mean_inflow_forecast_standardized',
                                  'remaining_external_volume_normalized'))
                state = env.state()
                storage = ((env.storage-self.data.vmin)/
                           (self.data.vmax-self.data.vmin)).astype(np.float32)
                actual = ((env.data.external[0]-self.data.flow_mean)/
                          self.data.flow_std).astype(np.float32)
                np.testing.assert_array_equal(state[:5], storage)
                np.testing.assert_array_equal(state[5:10], actual)
                np.testing.assert_array_equal(state[10:15], np.zeros(5))
                np.testing.assert_array_equal(obs[:, 0], state[:5])
                np.testing.assert_array_equal(obs[:, 1], state[5:10])
                if limits:
                    for actor_index, state_index in enumerate((18, 23, 28, 33), start=8):
                        np.testing.assert_array_equal(obs[:, actor_index], state[state_index:state_index+5])
                np.testing.assert_array_equal(shared, np.repeat(state[None], 5, axis=0))
                self.assert_context(env)

    def test_changed_forecast_distinguishes_critic_context_and_only_own_actor_feature(self):
        changed = copy.copy(self.data)
        changed.inflow_forecast = self.data.inflow_forecast.copy()
        changed.inflow_forecast[0, 2] += 3.*self.data.flow_std[2]
        for limits in (False, True):
            with self.subTest(daily_limits=limits):
                left, right = self.environment(limits), self.environment(limits, source=changed)
                old_state, new_state = left.state(), right.state()
                old_actor, new_actor = left.actor_observations(), right.actor_observations()
                np.testing.assert_array_equal(old_state[:-15], new_state[:-15])
                np.testing.assert_array_equal(old_state[-10:-5], new_state[-10:-5])
                np.testing.assert_array_equal(old_actor[:, :-2], new_actor[:, :-2])
                np.testing.assert_array_equal(np.flatnonzero(old_state != new_state),
                                              [len(old_state)-15+2, len(old_state)-5+2])
                np.testing.assert_array_equal(np.flatnonzero(old_actor[:, -2] != new_actor[:, -2]), [2])
                self.assertAlmostEqual(float(new_state[-13]-old_state[-13]), 3., places=5)
                self.assert_context(right)

    def test_changed_target_distinguishes_otherwise_identical_context(self):
        for limits in (False, True):
            with self.subTest(daily_limits=limits):
                left, right = self.environment(limits), self.environment(limits)
                right.terminal_target = (self.data.vmin+
                    np.array([.1, .2, .3, .4, .5])*(self.data.vmax-self.data.vmin))
                np.testing.assert_array_equal(left.storage, right.storage)
                np.testing.assert_array_equal(left.initial_storage, right.initial_storage)
                np.testing.assert_array_equal(left.state()[:-10], right.state()[:-10])
                np.testing.assert_array_equal(left.state()[-5:], right.state()[-5:])
                np.testing.assert_array_equal(left.actor_observations()[:, :-3],
                                              right.actor_observations()[:, :-3])
                np.testing.assert_array_equal(left.actor_observations()[:, -2:],
                                              right.actor_observations()[:, -2:])
                self.assertFalse(np.array_equal(left.state()[-10:-5], right.state()[-10:-5]))
                self.assert_context(right)

    def test_disabled_terminal_targets_are_zero_while_forecasts_remain_visible(self):
        for limits in (False, True):
            with self.subTest(daily_limits=limits):
                env = self.environment(limits, terminal=False)
                for _ in range(2):
                    self.assertIsNone(env.terminal_target)
                    np.testing.assert_array_equal(env.state()[-10:], np.zeros(10))
                    np.testing.assert_array_equal(env.actor_observations()[:, -3], np.zeros(5))
                    self.assert_context(env)
                    env.step(np.arange(5))

    def test_context_advances_to_next_source_date_and_terminal_observations_are_zero(self):
        for limits in (False, True):
            with self.subTest(daily_limits=limits):
                env = self.environment(limits)
                target = env.state()[-10:-5].copy()
                for step in range(env.episode_days):
                    self.assert_context(env)
                    np.testing.assert_array_equal(env.state()[-10:-5], target)
                    obs, shared, _, done, _, _ = env.step((np.arange(5)+step) % 11)
                self.assertTrue(done.all())
                np.testing.assert_array_equal(obs, np.zeros((5, env.observation_size)))
                np.testing.assert_array_equal(shared, np.zeros((5, env.state_size)))
                np.testing.assert_array_equal(env.state(), np.zeros(env.state_size))
                env.reset()
                self.assert_context(env)

    def test_forecasts_keep_original_dates_across_leaps_and_nonconsecutive_year_joins(self):
        source_dates = {day: index for index, day in enumerate(self.data.dates)}
        for limits in (False, True):
            with self.subTest(daily_limits=limits):
                env = self.environment(limits, years=[1968, 1970, 2000], days=1097)
                target = env.state()[-10:-5].copy()
                for day in (date(1968, 2, 29), date(1968, 12, 31), date(1970, 1, 1),
                            date(1970, 12, 31), date(2000, 1, 1), date(2000, 2, 29)):
                    env.index = env.data.dates.index(day)
                    np.testing.assert_array_equal(env.data.inflow_forecast[env.index],
                                                  self.data.inflow_forecast[source_dates[day]])
                    np.testing.assert_array_equal(env.state()[-10:-5], target)
                    self.assert_context(env)

    def test_future_observed_inflow_does_not_change_earlier_context_or_dispatch(self):
        changed = copy.copy(self.data)
        changed.external = self.data.external.copy()
        changed.external[2:] += 10000.
        for limits in (False, True):
            with self.subTest(daily_limits=limits):
                left, right = self.environment(limits), self.environment(limits, source=changed)
                for step in range(2):
                    np.testing.assert_array_equal(left.state(), right.state())
                    np.testing.assert_array_equal(left.actor_observations(), right.actor_observations())
                    commands = (np.arange(5)+step) % 11
                    left.step(commands)
                    right.step(commands)
                    for name in ('release', 'storage_end', 'power', 'terminal_forecast'):
                        np.testing.assert_array_equal(left.last_transition[name], right.last_transition[name])
                # The changed forcing becomes visible only once its own date is current.
                self.assertFalse(np.array_equal(left.state()[5:10], right.state()[5:10]))
                np.testing.assert_array_equal(left.state()[-10:], right.state()[-10:])


if __name__ == '__main__':
    unittest.main(verbosity=2)
