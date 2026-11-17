"""Execution completion must remain distinct from required terminal equality."""
import copy
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from examples.train_cascade_parallel import build_correctness_payload
from harl.envs.cascade_reservoir.evaluation import episode_constraint_status
from harl.envs.cascade_reservoir.periodic_evaluation import evaluate_periodic_policy
from harl.envs.cascade_reservoir.reservoir_physics import InfeasibleCascade


def terminal(met):
    target = [1e9]*5
    final = target.copy()
    final[0] -= 0. if met else 100.
    return dict(target_m3=target, final_storage_m3=final,
        deviation_m3=[end-start for start,end in zip(target, final)],
        tolerance_m3=[.1]*5, met=met)


def shortfalls(count=0, active=5):
    return {'ecological_flow': dict(reservoir_day_count=count, active_reservoir_days=active)}


class ConstraintStatusTest(unittest.TestCase):
    def test_completed_terminal_miss_is_not_required_constraint_success(self):
        row = episode_constraint_status(completed=True, physical_feasible=True,
            terminal_enabled=True, terminal_objective=terminal(False), service_shortfalls=shortfalls())
        self.assertTrue(row['completed'])
        self.assertTrue(row['physical_feasible'])
        self.assertTrue(row['feasible'])
        self.assertIn('physical-only', row['feasible_definition'])
        self.assertFalse(row['terminal_met'])
        self.assertFalse(row['required_constraints_met'])
        self.assertTrue(row['ecological_compliant'])

    def test_ecology_is_soft_and_disabled_or_incomplete_measurements_remain_null(self):
        passed = episode_constraint_status(completed=True, physical_feasible=True,
            terminal_enabled=True, terminal_objective=terminal(True), service_shortfalls=shortfalls(2))
        self.assertTrue(passed['required_constraints_met'])
        self.assertFalse(passed['ecological_compliant'])
        disabled = episode_constraint_status(completed=True, physical_feasible=True,
            terminal_enabled=False, service_shortfalls=shortfalls(active=0))
        self.assertIsNone(disabled['terminal_met'])
        self.assertIsNone(disabled['ecological_compliant'])
        self.assertTrue(disabled['required_constraints_met'])
        partial = episode_constraint_status(completed=False, physical_feasible=False,
            terminal_enabled=True, terminal_objective=terminal(True), service_shortfalls=shortfalls())
        self.assertFalse(partial['required_constraints_met'])
        self.assertIsNone(partial['terminal_met'])
        self.assertIsNone(partial['ecological_compliant'])
        missing = episode_constraint_status(completed=True, physical_feasible=True, terminal_enabled=True)
        self.assertIsNone(missing['required_constraints_met'])


class PeriodicConstraintStatusTest(unittest.TestCase):
    def measured(self, value, met):
        row = dict(episode_reward=value, episode_system_reward=value,
            episode_global_reward=value, episode_mixed_rewards=[value]*5,
            episode_local_rewards=[value]*5, energy_MWh=value, energy_TWh=value/1e6,
            episode_penalty_ecological_flow=0., violation_count=0, max_daily_bound_excess_m3=0.,
            terminal_objective=terminal(met), service_shortfalls=shortfalls())
        row.update({name+'_end_m3':1e9 for name in ('WDD','BHT','XLD','XJB','THR')})
        return row

    def evaluate(self, measurements, enabled=True):
        windows = [dict(window_id=str(i), source_years=[2001+i], episode_horizon_days=1,
                        start_date=str(2001+i)+'-01-01', initial_storage_m3=[1e9]*5) for i in range(3)]
        with mock.patch('harl.envs.cascade_reservoir.periodic_evaluation._validated_windows', return_value=windows), \
                mock.patch('harl.envs.cascade_reservoir.periodic_evaluation.evaluate_policy', side_effect=measurements):
            return evaluate_periodic_policy([], SimpleNamespace(levels=lambda values:np.asarray(values)),
                {'terminal_storage_equal_initial':enabled}, seed=1, evaluation_round=1,
                training_steps=3, windows=windows, gamma=.995)

    def test_terminal_miss_preserves_all_scenario_returns_and_counts_required_passes(self):
        row = self.evaluate([self.measured(10., True), self.measured(20., False), self.measured(30., True)])
        self.assertEqual(row['status'], 'ok')
        self.assertTrue(row['completed'])
        self.assertEqual(row['completed_window_count'], 3)
        self.assertEqual(row['terminal_pass_window_count'], 2)
        self.assertEqual(row['required_pass_window_count'], 2)
        self.assertFalse(row['required_constraints_met'])
        self.assertEqual(row['evaluator_return'], 20.)
        self.assertIsNone(row['constraint_qualified_evaluator_return'])
        self.assertEqual([window['evaluator_return'] for window in row['windows']], [10.,20.,30.])

    def test_partial_failure_never_averages_surviving_windows(self):
        row = self.evaluate([self.measured(10., True), InfeasibleCascade('failed day'), self.measured(30., True)])
        self.assertEqual(row['completed_window_count'], 2)
        self.assertFalse(row['completed'])
        self.assertFalse(row['physical_feasible'])
        self.assertFalse(row['required_constraints_met'])
        self.assertIsNone(row['evaluator_return'])
        self.assertIsNone(row['windows'][1]['terminal_met'])

    def test_disabled_terminal_requirement_has_no_terminal_pass_claim(self):
        row = self.evaluate([self.measured(10., False)]*3, enabled=False)
        self.assertEqual(row['terminal_required_window_count'], 0)
        self.assertEqual(row['terminal_pass_window_count'], 0)
        self.assertEqual(row['required_pass_window_count'], 3)
        self.assertIsNone(row['terminal_met'])
        self.assertTrue(row['required_constraints_met'])
        self.assertEqual(row['constraint_qualified_evaluator_return'], 10.)


class CompleteTerminalHistoryTest(unittest.TestCase):
    def test_payload_includes_all_collection_and_periodic_endpoints(self):
        episodes = [dict(episode=i+1, policy_updates_completed=i//2, completed=True,
            source_years=[2001], terminal_objective=terminal(i%2 == 0)) for i in range(4)]
        periodic = [dict(evaluation_round=number, windows=[dict(window_id=label, completed=True,
            status='ok', source_years=[2001], terminal_objective=terminal(True))
            for label in ('dry','median','wet')]) for number in (0,1)]
        runner = SimpleNamespace(seed=1, update_count=1, episode_history=episodes,
            periodic_evaluation_history=periodic, correctness_trajectory={}, correctness_update_history=[])
        heldout = dict(seed=1, completed=True, source_years=[1998], terminal_objective=terminal(True))
        original = copy.deepcopy(episodes)
        result = build_correctness_payload([runner], {'physical_config':{'terminal_storage_equal_initial':True}}, [heldout])
        run = result['runs'][0]
        self.assertEqual(len(run['terminal_records']), 11)
        self.assertEqual(run['terminal_summary']['training']['record_count'], 4)
        self.assertEqual(run['terminal_summary']['training']['terminal_pass_count'], 2)
        self.assertEqual(run['terminal_summary']['training_evaluation']['record_count'], 6)
        self.assertEqual([row['policy_update'] for row in run['terminal_records'][:4]], [0,0,1,1])
        self.assertTrue(all(row['reported_status_consistent'] for row in run['terminal_records']))
        self.assertEqual(episodes, original)


if __name__ == '__main__':
    unittest.main()
