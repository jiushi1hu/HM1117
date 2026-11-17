"""Compact console output preserves complete logs and actual training updates."""
import copy
from contextlib import redirect_stdout
import io
import json
import os
from pathlib import Path
import random
import re
import sys
import unittest
from unittest import mock

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from examples.train_cascade import load_configuration
from harl.envs.cascade_reservoir.data_loader import CascadeData
from harl.envs.cascade_reservoir.episodes import episode_environment
from harl.envs.cascade_reservoir.periodic_evaluation import select_periodic_windows
from harl.envs.cascade_reservoir.reward import EpisodeRewardMetrics
from harl.envs.cascade_reservoir.training_metrics import episode_record
from harl.runners import cascade_reservoir_runner as runner_module


class InProcessCascadeEnvs:
    """Use the worker's real environment/metrics contract without subprocesses."""
    def __init__(self, data, physical, bins, days, count, gamma, trace, seed):
        self.data, self.physical, self.bins, self.days = data, physical, bins, days
        self.count, self.gamma, self.trace, self.seed = count, gamma, trace, seed
        self.closed = False

    def reset(self, windows, episode_ids):
        self.envs = [episode_environment(self.data, self.physical, years, self.days, self.bins)
                     for years in windows]
        self.episode_ids = episode_ids
        self.metrics = [EpisodeRewardMetrics(self.gamma) for _ in windows]
        self.energy = [np.zeros(5) for _ in windows]
        return [dict(obs=env.actor_observations(), state=env.state(), pid=os.getpid())
                for env in self.envs]

    def step(self, commands):
        results = []
        for lane, (env, actions) in enumerate(zip(self.envs, commands)):
            env.step(actions)
            row = env.last_transition
            self.metrics[lane].update(row)
            self.energy[lane] += row['energy']
            done = env.index == env.stop
            result = dict(obs=env.actor_observations(), state=env.state(), transition=row, done=done)
            if done:
                record = episode_record(env, self.metrics[lane], self.energy[lane], self.seed,
                    self.episode_ids[lane], self.gamma, self.trace, 0)
                record['worker_pid'] = os.getpid()
                result.update(episode_record=record, reward_metrics=self.metrics[lane].summary())
            results.append(result)
        return results

    def close(self):
        self.closed = True


class CascadeConsoleProgressTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config, cls.physical = load_configuration()
        cls.config['train'].update(episode_length=3, n_rollout_threads=1)
        cls.config['model'].update(hidden_sizes=[8, 8], action_bins=5)
        cls.config['algo'].update(ppo_epoch=1, critic_epoch=1)
        cls.config['experiment'].update(episodes=1, seeds=[1])
        cls.config['experiment'].pop('verbose_console', None)
        cls.data = CascadeData(ROOT/'data/cascade_reservoir_new', cls.physical,
                               cls.config['experiment'])
        cls.windows = select_periodic_windows(cls.data, episode_days=3)

    def execute(self, verbose=None, episodes=1, log_interval=1):
        config = copy.deepcopy(self.config)
        config['experiment']['episodes'] = episodes
        if verbose is not None:
            config['experiment']['verbose_console'] = verbose
        console, log = io.StringIO(), io.StringIO()
        with mock.patch.object(runner_module, 'ParallelCascadeEnvs', InProcessCascadeEnvs), \
                redirect_stdout(console):
            runner = runner_module.CascadeReservoirRunner(self.data, self.physical, config, 1,
                episode_log=log, periodic_windows=self.windows, correctness_enabled=True)
            self.addCleanup(runner.close)
            runner.run(episodes, log_interval=log_interval)
            runner.final_evaluation()
        records = [json.loads(line) for line in log.getvalue().splitlines()]
        return runner, console.getvalue(), records

    def assert_state_equal(self, left, right):
        if isinstance(left, torch.Tensor):
            self.assertTrue(torch.equal(left, right))
        elif isinstance(left, np.ndarray):
            np.testing.assert_array_equal(left, right)
        elif isinstance(left, dict):
            self.assertEqual(set(left), set(right))
            for key in left:
                self.assert_state_equal(left[key], right[key])
        elif isinstance(left, (list, tuple)):
            self.assertEqual(len(left), len(right))
            for first, second in zip(left, right):
                self.assert_state_equal(first, second)
        else:
            self.assertEqual(left, right)

    def test_default_console_is_compact_seed_progress_with_complete_jsonl(self):
        runner, console, records = self.execute()
        for marker in ('EPISODE_METRICS', 'HAPPO_DIAGNOSTICS', 'DIAG_UPDATE', 'SAMPLER', '"record_type"'):
            self.assertNotIn(marker, console)
        self.assertIn('seed=1', console)
        self.assertIn('update=1/1', console)
        self.assertIn('episodes=1', console)
        self.assertIn('return=', console)
        self.assertLessEqual(len(console.splitlines()), 10)
        types = [row['record_type'] for row in records]
        for kind in ('training_sampling_plan', 'training_episode', 'training_update',
                     'training_diagnostics', 'correctness_happo_update',
                     'periodic_evaluation', 'final_evaluation_sequence'):
            self.assertIn(kind, types)
        self.assertEqual(types.count('periodic_evaluation'), 2)
        self.assertEqual(len(runner.correctness_update_history), 1)
        self.assertEqual(runner.correctness_trajectory['steps'], 3)
        self.assertEqual(len(runner.diagnostics_history), 1)

    def test_final_update_is_printed_when_interval_exceeds_budget(self):
        runner, console, records = self.execute(episodes=2, log_interval=3)
        training = [line for line in console.splitlines() if 'phase=training ' in line]
        self.assertEqual(len(training), 1)
        self.assertIn('update=2/2', training[0])
        self.assertIn('episodes=2/2', training[0])
        measured_return = float(re.search(r'\breturn=([^ ]+)', training[0]).group(1))
        self.assertAlmostEqual(measured_return, runner.last_reward_metrics['episode_system_reward'], places=6)
        self.assertIn('phase=training-complete', console)
        self.assertEqual(sum(row['record_type'] == 'training_update' for row in records), 2)
        self.assertEqual(sum(row['record_type'] == 'correctness_happo_update' for row in records), 2)

    def test_verbose_console_restores_json_without_dropping_persisted_records(self):
        _, console, records = self.execute(verbose=True)
        self.assertIn('EPISODE_METRICS', console)
        self.assertIn('DIAG_UPDATE', console)
        self.assertIn('"record_type": "training_sampling_plan"', console)
        self.assertIn('"record_type": "training_diagnostics"', console)
        self.assertIn('"record_type": "correctness_happo_update"', console)
        self.assertIn('"record_type": "final_evaluation_sequence"', console)
        self.assertTrue(records)

    def test_console_verbosity_does_not_change_training_state_or_random_streams(self):
        quiet, _, quiet_records = self.execute(verbose=False)
        rng = (random.getstate(), np.random.get_state(), torch.get_rng_state().clone())
        verbose, _, verbose_records = self.execute(verbose=True)
        self.assert_state_equal(rng, (random.getstate(), np.random.get_state(), torch.get_rng_state()))
        for left, right in zip(quiet.actor, verbose.actor):
            self.assert_state_equal(left.actor.state_dict(), right.actor.state_dict())
            self.assert_state_equal(left.actor_optimizer.state_dict(), right.actor_optimizer.state_dict())
        self.assert_state_equal(quiet.critic.critic.state_dict(), verbose.critic.critic.state_dict())
        self.assert_state_equal(quiet.critic.critic_optimizer.state_dict(), verbose.critic.critic_optimizer.state_dict())
        self.assert_state_equal(quiet.value_normalizer.state_dict(), verbose.value_normalizer.state_dict())
        self.assertEqual([row['record_type'] for row in quiet_records],
                         [row['record_type'] for row in verbose_records])
        self.assertEqual(quiet.periodic_evaluation_history, verbose.periodic_evaluation_history)
        self.assertEqual(quiet.correctness_update_history, verbose.correctness_update_history)
        self.assertEqual(quiet.correctness_trajectory, verbose.correctness_trajectory)


if __name__ == '__main__':
    unittest.main()
