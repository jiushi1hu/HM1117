"""Render serialized correctness evidence without importing training dependencies."""
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np

try:
    import matplotlib
    matplotlib.use('Agg', force=True)
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from examples import plot_cascade_correctness as plots

FIGURES = ('mass_balance', 'water_levels', 'release_feasibility',
           'terminal_storage', 'reward_reconciliation', 'happo_updates')
IDS = ['WDD', 'BHT', 'XLD', 'XJB', 'THR']


def stats(value, count=4):
    return dict(count=count, min=value, median=value, p95=value, max=value)


def fixture(steps=2, terminal=True, failed=False):
    """Hand-calculated constant-storage daily chain, with two actor updates."""
    release = np.array([5., 10., 20., 30., 40.])
    power = .8829*release
    local_generation = power/1000.
    local_penalty = np.array([.5, .5, 0., .25, .2])
    local = local_generation-local_penalty
    generation = float(power.sum()/5000.)
    system = float(.5*local.sum()+.5*generation)
    base = dict(storage_start=[43200000.]*5, storage_end=[43200000.]*5,
        levels_start=[5.]*5, levels_end=[5.]*5, external=[5., 5., 10., 10., 10.],
        inflow=release.tolist(), release=release.tolist(), low=[0.]*5, high=[10000.]*5,
        controls=[1.]*5, power=power.tolist(), energy=(24.*power).tolist(),
        turbine=release.tolist(), spill=[0.]*5, net_head_m=[100.]*5,
        local_generation_rewards=local_generation.tolist(), local_reward_penalties=local_penalty.tolist(),
        local_rewards=local.tolist(), mixed_rewards=(.5*local+.5*generation).tolist(),
        generation_reward=generation, global_reward=generation, reward_penalty=.3625,
        reward=generation-.3625, system_reward=system,
        storage_lower_m3=[0.]*5, storage_upper_m3=[864000000.]*5,
        next_state_level_lower_m=[0.]*5, next_state_level_upper_m=[100.]*5,
        sampled_actions=[0.]*5, actions=[0.]*5, normalized_actions=[0.]*5,
        terminal_target=[43200000.]*5 if terminal else None,
        terminal_forecast=release.tolist() if terminal else None,
        terminal_window_low_m3s=[0.]*5 if terminal else None,
        terminal_window_high_m3s=[10000.]*5 if terminal else None,
        terminal_window_binding=[0.]*5 if terminal else None,
        terminal_window_infeasible=[0.]*5 if terminal else None,
        terminal_window_reason=[0.]*5 if terminal else None)
    rows = [dict(copy.deepcopy(base), date='2001-01-0'+str(day+1),
                 terminal_remaining_days=2-day if terminal else None) for day in range(steps)]
    series = {key:[row[key] for row in rows] for key in rows[0]} if rows else {}
    independent_base = dict(mass_balance_residual_m3=[0.]*5,
        mass_balance_tolerance_m3=[1e-5]*5, cascade_mass_balance_residual_m3=0.,
        cascade_mass_balance_tolerance_m3=5e-5, inflow_routing_residual_m3s=[0.]*5,
        storage_continuity_residual_m3=[0.]*5,
        levels_start=[5.]*5, levels_end=[5.]*5, net_head_m=[100.]*5,
        turbine=release.tolist(), power=power.tolist(), energy=(24.*power).tolist(),
        ecological_shortfall_m3s=[5., 10., 0., 10., 10.],
        normalized_ecological_shortfall=local_penalty.tolist())
    scalar_rewards = ('reward', 'generation_reward', 'reward_penalty', 'global_reward', 'system_reward')
    vector_rewards = ('local_generation_rewards', 'local_reward_penalties', 'local_rewards', 'mixed_rewards')
    independent_base.update({key:copy.deepcopy(base[key]) for key in scalar_rewards+vector_rewards})
    independent = {key:[copy.deepcopy(value) for _ in rows]
                   for key, value in independent_base.items()} if rows else {}
    differences = {key:[0. if key in scalar_rewards else [0.]*5 for _ in rows]
                   for key in scalar_rewards+vector_rewards} if rows else {}
    vector_flags = ('mass_balance', 'inflow_routing', 'storage_continuity', 'storage_bounds',
        'release_bounds', 'power_capacity', 'turbine_capacity', 'spill_partition',
        'generation_calculation', 'level_lookup', 'forced', 'terminal_window_infeasible', 'hard_physical')
    flags = {key:[[False]*5 for _ in rows] for key in vector_flags} if rows else {}
    if rows:
        flags.update(cascade_mass_balance=[False]*steps, reward_mismatch=[False]*steps,
                     ecological=[[True, True, False, True, True] for _ in rows])
    completed = steps == 2 and not failed
    physical = dict(timestep_seconds=86400., head_model='fixed', efficiency=.9,
        power_cap_mw=[1000.]*5, release_cap_m3s=[10000.]*5, turbine_cap_m3s=[1000.]*5,
        storage_min_m3=[0.]*5, storage_max_m3=[864000000.]*5,
        ecological_flow_m3s=[10., 20., 0., 40., 50.], service_requirements_available=True,
        reward_penalties_enabled=True, ecological_flow_weight=1., local_global_alpha=.5,
        terminal_storage_equal_initial=terminal, initial_storage_m3=[43200000.]*5,
        initial_level_m=[5.]*5, target_storage_m3=[43200000.]*5 if terminal else None,
        target_level_m=[5.]*5 if terminal else None, balance_accumulator_dtype='float128',
        balance_relative_tolerance=64.*np.finfo(np.float64).eps,
        reward_absolute_tolerance=1e-10, reward_relative_tolerance=1e-10)
    terminal_report = dict(enabled=terminal, reported=None,
                           at_episode_end=completed and terminal, met=None)
    if terminal and rows:
        terminal_report.update(target_storage_m3=[43200000.]*5, final_storage_m3=[43200000.]*5,
            target_level_m=[5.]*5, final_level_m=[5.]*5, deviation_m3=[0.]*5,
            tolerance_m3=[.00432]*5, met=True if completed else None)
    trajectory = dict(schema_version=1, record_type='selected_episode_correctness',
        seed=1, episode=4, source_years=[2001], reservoir_ids=IDS.copy(),
        steps=steps, max_steps=2, expected_steps=2, physical=physical,
        series=series, independent=independent, reward_differences=differences,
        violations=flags, terminal=terminal_report,
        phase='training_evaluation', scenario_id='median', policy_update=2,
        status='failed' if failed else 'ok', summary=dict(
            violation_counts={key:int(np.count_nonzero(value)) for key, value in flags.items()},
            max_abs_mass_balance_residual_m3=0. if steps else None,
            max_abs_cascade_mass_balance_residual_m3=0. if steps else None,
            max_abs_reward_difference=0. if steps else None,
            active_ecological_reservoir_days=4*steps,
            ecological_violation_rate=1. if steps else None,
            hard_physical_reservoir_days=0, reward_mismatch_days=0))
    terminal_record = dict(phase='training_evaluation', scenario_id='median',
        policy_update=2, source_years=[2001], enabled=terminal,
        available=terminal and completed,
        status='disabled' if not terminal else 'failed' if failed else 'ok')
    if terminal_record['available']:
        terminal_record.update(target_m3=[43200000.]*5, final_storage_m3=[43200000.]*5,
            deviation_m3=[0.]*5, tolerance_m3=[.00432]*5, met=True)
    updates = []
    for number in (1, 2):
        order = [3, 1, 4, 0, 2] if number == 1 else [0, 1, 2, 3, 4]
        actors = [dict(agent_id=agent, update_position=position, rows=4,
            controllable_rows=3, forced_rows=1, used_factor=stats(1.+position/10.),
            expected_factor=stats(1.+position/10.), factor_max_abs_error=1e-7,
            factor_max_relative_error=1e-7, final_sampled_ratio=stats(1., count=3),
            final_masked_ratio=stats(1.), forced_factor_ratio_max_abs_error=0.,
            final_policy_kl=.002*number, final_clip_fraction=.01*number,
            final_entropy=1.-.1*number) for position, agent in enumerate(order)]
        surrogate = {name+'_'+suffix:value for name in
            ('clipped_surrogate', 'unclipped_surrogate', 'weighted_entropy', 'regularized_objective')
            for suffix, value in (('before', .1), ('after', .12), ('delta', .02))}
        learning = [dict(agent=agent, active_samples=3, policy_kl=.002*number,
                         clip_fraction=.01*number, entropy=1.-.1*number,
                         surrogate=copy.deepcopy(surrogate)) for agent in range(5)]
        updates.append(dict(update=number, agent_order=order, actors=actors,
                            learning_metrics=learning, factor_oracle='independent float64 log ratio'))
    return dict(schema='cascade_correctness_v1',
        configuration=dict(episode_horizon_days=2, parallel_environments=2,
            episodes_per_seed=4, updates_per_seed=2,
            physical_config=dict(terminal_storage_equal_initial=terminal,
                head_model='fixed', efficiency=.9, water_density_kg_m3=1000., gravity_m_s2=9.81,
                timestep_seconds=86400., level_bounds_m={name:[0., 100.] for name in IDS},
                reward_penalties=dict(enabled=True, ecological_flow_weight=1.)),
            config=dict(algo=dict(gamma=.995, clip_param=.2),
                experiment=dict(seeds=[1]), train=dict(episode_length=2, n_rollout_threads=2),
                model=dict(action_bins=3)),
            data_hashes={'inflow.csv':'a'*64}, source_code_hashes={'harl/example.py':'b'*64}),
        interpretation='Independent consistency checks; not proof of hydraulic calibration or optimality.',
        trajectory_scope='deterministic final-policy median training-only evaluation scenario per seed',
        update_scope='all training updates; bounded summaries',
        runs=[dict(seed=1, trajectory=trajectory, terminal_records=[terminal_record], update_records=updates)])


def visible_text(figure):
    values = []
    if figure._suptitle is not None:
        values.append(figure._suptitle.get_text())
    for ax in figure.axes:
        values.extend((ax.get_title(), ax.get_xlabel(), ax.get_ylabel()))
        values.extend(text.get_text() for text in ax.texts)
        values.extend(ax.get_legend_handles_labels()[1])
    return ' '.join(values).lower()


@unittest.skipIf(plt is None, 'Matplotlib is available in the plotting interpreter')
class CorrectnessFigureTest(unittest.TestCase):
    def tearDown(self):
        plt.close('all')

    def test_terminal_history_is_retained_but_only_final_cases_get_tick_labels(self):
        payload = fixture()
        run = payload['runs'][0]
        base = run['terminal_records'][0]
        run['terminal_records'] = [dict(copy.deepcopy(base), phase='training',
            scenario_id='collection_episode_'+str(i+1), policy_update=i//6) for i in range(300)]
        run['terminal_records'] += [dict(copy.deepcopy(base), phase='training_evaluation',
            scenario_id=label, policy_update=number) for number in range(51)
            for label in ('dry', 'median', 'wet')]
        run['terminal_records'].append(dict(copy.deepcopy(base), phase='evaluation',
                                           scenario_id='held_out_sequence', policy_update=50))
        figures = plots.make_figures(payload)
        self.assertEqual(len(figures['terminal_storage'].axes[0].get_xticklabels()), 10)
        text = visible_text(figures['terminal_storage'])
        self.assertIn('300/300', text)
        self.assertIn('153/153', text)
        self.assertEqual(len(run['terminal_records']), 454)

    def test_plan_rejection_and_fallback_are_distinct_from_forced_controls(self):
        payload = fixture()
        series = payload['runs'][0]['trajectory']['series']
        series.update(physical_low_m3s=[[0.]*5]*2, physical_high_m3s=[[20000.]*5]*2,
                      terminal_plan_rejected=[True, False], terminal_fallback=[False, True])
        figures = plots.make_figures(payload)
        labels = figures['release_feasibility'].axes[0].get_legend_handles_labels()[1]
        self.assertIn('Physical release bounds', labels)
        self.assertIn('Plan rejection (1 days)', labels)
        self.assertIn('Final fallback (1 days)', labels)
        self.assertIn('Forced action (0 days)', labels)

    def test_six_figures_use_actual_physical_and_update_evidence_without_mutation(self):
        payload = fixture()
        original = copy.deepcopy(payload)
        figures = plots.make_figures(payload)
        self.assertEqual(set(figures), set(FIGURES))
        for name, figure in figures.items():
            self.assertTrue(figure.axes, name)
            figure.canvas.draw()
        self.assertIn('balance', visible_text(figures['mass_balance']))
        self.assertIn('level', visible_text(figures['water_levels']))
        self.assertIn('release', visible_text(figures['release_feasibility']))
        self.assertIn('terminal', visible_text(figures['terminal_storage']))
        self.assertIn('reward', visible_text(figures['reward_reconciliation']))
        self.assertIn('factor', visible_text(figures['happo_updates']))
        levels = [np.asarray(line.get_ydata(), dtype=float) for ax in figures['water_levels'].axes
                  for line in ax.lines if np.ndim(line.get_ydata()) == 1]
        self.assertTrue(any(np.allclose(values, [5., 5.]) for values in levels if values.size == 2))
        release_values = [np.asarray(line.get_ydata(), dtype=float)
                          for ax in figures['release_feasibility'].axes for line in ax.lines]
        for flow in (5., 10., 20., 30., 40.):
            self.assertTrue(any(values.size == 2 and np.allclose(values, flow)
                                for values in release_values), flow)
        self.assertEqual(payload, original)

    def test_empty_partial_and_disabled_terminal_evidence_remains_renderable(self):
        for steps, terminal in ((0, True), (1, True), (2, False)):
            with self.subTest(steps=steps, terminal=terminal):
                payload = fixture(steps=steps, terminal=terminal, failed=steps < 2)
                if steps < 2:
                    payload['runs'][0]['update_records'] = []
                figures = plots.make_figures(payload)
                self.assertEqual(set(figures), set(FIGURES))
                for figure in figures.values():
                    figure.canvas.draw()
                terminal_text = visible_text(figures['terminal_storage'])
                self.assertTrue(any(word in terminal_text for word in
                                    (('disabled',) if not terminal else ('unavailable', 'failed', 'partial', 'incomplete'))))
                plt.close('all')

    def test_malformed_shape_nonfinite_and_unknown_schema_fail_before_drawing(self):
        cases = []
        payload = fixture()
        payload['schema'] = 'unknown_schema'
        cases.append(payload)
        payload = fixture()
        payload['runs'][0]['trajectory']['series']['release'][0] = [1., 2., 3., 4.]
        cases.append(payload)
        payload = fixture()
        payload['runs'][0]['trajectory']['series']['power'][0][0] = float('nan')
        cases.append(payload)
        for index, payload in enumerate(cases):
            with self.subTest(case=index), mock.patch.object(
                    plt, 'subplots', side_effect=AssertionError('Unexpected drawing')):
                with self.assertRaises(ValueError):
                    plots.make_figures(payload)


@unittest.skipIf(plt is None, 'Matplotlib is available in the plotting interpreter')
class CorrectnessPlotCLITest(unittest.TestCase):
    def command(self, *arguments, directory):
        environment = dict(os.environ, MPLCONFIGDIR=str(Path(directory)/'mpl'))
        return subprocess.run([sys.executable, '-B', str(ROOT/'examples/plot_cascade_correctness.py'),
                               *map(str, arguments)], capture_output=True, text=True, env=environment)

    def test_cli_saves_six_exclusive_pngs_and_preserves_source_evidence_and_code(self):
        sources = [ROOT/'harl/runners/on_policy_ha_runner.py',
                   ROOT/'harl/envs/cascade_reservoir/reservoir_physics.py',
                   ROOT/'examples/plot_cascade_correctness.py']
        hashes = {path:hashlib.sha256(path.read_bytes()).hexdigest() for path in sources}
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory)/'figures'
            target.mkdir()
            evidence = target/'evidence.json'
            evidence.write_text(json.dumps(fixture(), allow_nan=False), encoding='utf-8')
            original = evidence.read_bytes()
            result = self.command('--check-backend', directory=directory)
            self.assertEqual(result.returncode, 0, result.stderr)
            result = self.command('--input', evidence, '--output-dir', target, '--no-show', directory=directory)
            self.assertEqual(result.returncode, 0, result.stderr)
            images = {name:target/(name+'.png') for name in FIGURES}
            for path in images.values():
                self.assertTrue(path.is_file(), str(path))
                self.assertEqual(path.read_bytes()[:8], b'\x89PNG\r\n\x1a\n')
            previous = {name:path.read_bytes() for name, path in images.items()}
            result = self.command('--input', evidence, '--output-dir', target, '--no-show', directory=directory)
            self.assertNotEqual(result.returncode, 0)
            self.assertEqual(evidence.read_bytes(), original)
            self.assertEqual({name:path.read_bytes() for name, path in images.items()}, previous)
        self.assertEqual({path:hashlib.sha256(path.read_bytes()).hexdigest() for path in sources}, hashes)

    def test_png_collision_is_rejected_before_partial_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory)/'figures'
            target.mkdir()
            evidence = target/'evidence.json'
            evidence.write_text(json.dumps(fixture(), allow_nan=False), encoding='utf-8')
            collision = target/'water_levels.png'
            collision.write_bytes(b'original file')
            result = self.command('--input', evidence, '--output-dir', target, '--no-show', directory=directory)
            self.assertNotEqual(result.returncode, 0)
            self.assertEqual(collision.read_bytes(), b'original file')
            self.assertEqual({path.name for path in target.glob('*.png')}, {'water_levels.png'})


if __name__ == '__main__':
    unittest.main()
