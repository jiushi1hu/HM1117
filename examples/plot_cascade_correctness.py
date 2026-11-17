"""Render six observational correctness figures using NumPy and Matplotlib only."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

FIGURE_KEYS = ('mass_balance', 'water_levels', 'release_feasibility',
               'terminal_storage', 'reward_reconciliation', 'happo_updates')
IDS = ('WDD', 'BHT', 'XLD', 'XJB', 'THR')
VECTOR_FIELDS = ('storage_start', 'storage_end', 'levels_start', 'levels_end',
    'external', 'inflow', 'release', 'low', 'high', 'controls', 'power', 'energy',
    'turbine', 'spill', 'net_head_m', 'local_generation_rewards',
    'local_reward_penalties', 'local_rewards', 'mixed_rewards')
SCALAR_REWARDS = ('reward', 'generation_reward', 'reward_penalty', 'global_reward', 'system_reward')
QUANTILES = ('min', 'median', 'p95', 'max')


def _array(value, shape, name):
    try:
        result = np.asarray(value, dtype=float)
    except (ValueError, TypeError) as exc:
        raise ValueError(name+' must be a finite array with shape '+str(shape)) from exc
    if result.shape != shape or not np.isfinite(result).all():
        raise ValueError(name+' must be a finite array with shape '+str(shape))
    return result


def _finite_tree(value, name='evidence'):
    if isinstance(value, dict):
        for key, item in value.items():
            _finite_tree(item, name+'.'+str(key))
    elif isinstance(value, (list, tuple)):
        for item in value:
            _finite_tree(item, name)
    elif isinstance(value, (float, np.floating)) and not np.isfinite(value):
        raise ValueError(name+' contains a non-finite value')


def validate_payload(payload):
    """Reject malformed evidence before allocating figures; preserve source values."""
    if not isinstance(payload, dict) or payload.get('schema') != 'cascade_correctness_v1':
        raise ValueError('Expected evidence schema cascade_correctness_v1')
    _finite_tree(payload)
    runs = payload.get('runs')
    if not isinstance(runs, list) or not runs:
        raise ValueError('Evidence must contain a nonempty runs list')
    seeds = set()
    for run in runs:
        if not isinstance(run, dict):
            raise ValueError('Every run must be an evidence object')
        seed = run.get('seed')
        if isinstance(seed, bool) or not isinstance(seed, int) or seed in seeds:
            raise ValueError('Every run requires a distinct integer seed')
        seeds.add(seed)
        trajectory = run.get('trajectory')
        if not isinstance(trajectory, dict) or trajectory.get('seed') != seed:
            raise ValueError('Each run requires trajectory evidence with the same seed')
        steps = trajectory.get('steps')
        if isinstance(steps, bool) or not isinstance(steps, int) or steps < 0:
            raise ValueError('trajectory.steps must be a nonnegative integer')
        if trajectory.get('reservoir_ids') != list(IDS):
            raise ValueError('Trajectory reservoir_ids must be WDD/BHT/XLD/XJB/THR')
        physical = trajectory['physical']
        if not isinstance(physical, dict):
            raise ValueError('trajectory.physical must be an evidence object')
        for key in ('power_cap_mw', 'release_cap_m3s', 'turbine_cap_m3s',
                    'storage_min_m3', 'storage_max_m3', 'ecological_flow_m3s'):
            _array(physical[key], (5,), 'physical.'+key)
        if steps:
            series, independent = trajectory['series'], trajectory['independent']
            if len(series['date']) != steps:
                raise ValueError('series.date length must equal trajectory.steps')
            for key in VECTOR_FIELDS:
                _array(series[key], (steps, 5), 'series.'+key)
            for key in SCALAR_REWARDS:
                _array(series[key], (steps,), 'series.'+key)
            for key in ('mass_balance_residual_m3', 'mass_balance_tolerance_m3'):
                _array(independent[key], (steps, 5), 'independent.'+key)
            for key in ('cascade_mass_balance_residual_m3', 'cascade_mass_balance_tolerance_m3'):
                _array(independent[key], (steps,), 'independent.'+key)
            if (np.asarray(independent['mass_balance_tolerance_m3']) <= 0).any():
                raise ValueError('Independent mass-balance tolerances must be positive')
            if (np.asarray(independent['cascade_mass_balance_tolerance_m3']) <= 0).any():
                raise ValueError('Independent whole-cascade mass-balance tolerances must be positive')
            for key in SCALAR_REWARDS:
                _array(independent[key], (steps,), 'independent.'+key)
                _array(trajectory['reward_differences'][key], (steps,), 'reward_differences.'+key)
            for key in ('local_generation_rewards', 'local_reward_penalties', 'local_rewards', 'mixed_rewards'):
                _array(independent[key], (steps, 5), 'independent.'+key)
                _array(trajectory['reward_differences'][key], (steps, 5), 'reward_differences.'+key)
            for key in ('next_state_level_lower_m', 'next_state_level_upper_m',
                        'physical_low_m3s', 'physical_high_m3s'):
                rows = series.get(key)
                if rows is not None and any(row is not None for row in rows):
                    _array(rows, (steps, 5), 'series.'+key)
            for key in ('terminal_plan_rejected', 'terminal_fallback'):
                rows = series.get(key)
                if rows is not None and any(row is not None for row in rows):
                    _array(rows, (steps,), 'series.'+key)
        for record in run.get('terminal_records', []):
            if record.get('enabled') and record.get('available', False):
                for key in ('target_m3', 'final_storage_m3', 'deviation_m3', 'tolerance_m3'):
                    _array(record[key], (5,), 'terminal_records.'+key)
                if (np.asarray(record['tolerance_m3']) <= 0).any():
                    raise ValueError('Terminal storage tolerances must be positive')
        previous = 0
        for record in run.get('update_records', []):
            number = record['update']
            if isinstance(number, bool) or not isinstance(number, int) or number <= previous:
                raise ValueError('HAPPO update numbers must be positive and strictly increasing')
            previous = number
            actors = record['actors']
            if sorted(actor['agent_id'] for actor in actors) != list(range(5)):
                raise ValueError('Each HAPPO update requires five distinct actor summaries')
            for actor in actors:
                for key in ('used_factor', 'expected_factor', 'final_sampled_ratio'):
                    stats = actor[key]
                    if stats.get('count', 0):
                        _array([stats[q] for q in QUANTILES], (4,), key)
                for key in ('factor_max_abs_error', 'factor_max_relative_error'):
                    _array(actor[key], (), key)
    return runs


def _panels(plt, runs, title, rows=5):
    figure, axes = plt.subplots(rows, len(runs), figsize=(5.2*len(runs), 2.8*rows),
                               squeeze=False, constrained_layout=True)
    figure.suptitle(title, fontsize=14)
    for column, run in enumerate(runs):
        trajectory = run['trajectory']
        status = trajectory.get('status', 'unknown')
        count = trajectory['steps']
        expected = trajectory.get('expected_steps', count)
        context = 'Seed {} | {}\n{}/{} executed days ({})'.format(
            run['seed'], trajectory.get('scenario_id', 'selected episode'), count, expected, status)
        for row in range(rows):
            ax = axes[row, column]
            ax.grid(alpha=.2)
            if row == 0:
                ax.set_title(context, fontsize=9)
            if row == rows-1:
                ax.set_xlabel('Executed day (source dates retained)')
            if not count:
                ax.text(.5, .5, 'Unavailable: no executed steps\n'+status,
                        ha='center', va='center', transform=ax.transAxes)
    return figure, axes


def _legend(ax):
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles, labels, fontsize=7, loc='best')


def _likelihood_axis(ax, values):
    """Show small changes around one in ordinary decimal likelihood units."""
    values = np.asarray([value for value in values if value is not None], dtype=float)
    values = values[np.isfinite(values)]
    wide = False
    if values.size:
        low, high = float(values.min()), float(values.max())
        wide = high/low > 100. if low > 0. else high > 100.
    if wide:
        ax.set_yscale('symlog', linthresh=.1)
    else:
        from matplotlib.ticker import MaxNLocator
        ax.set_yscale('linear')
        ax.ticklabel_format(axis='y', style='plain', useOffset=False)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.axhline(1., color='black', ls='--', lw=.6, label='_nolegend_')


def make_figures(payload):
    """Return six figures from saved evidence without model imports or mutation."""
    try:
        runs = validate_payload(payload)
    except (KeyError, TypeError) as exc:
        raise ValueError('Missing or malformed correctness evidence: '+str(exc)) from exc
    import matplotlib.pyplot as plt
    figures = {}
    titles = dict(mass_balance='Independent daily mass balance: residual and numerical tolerance',
        water_levels='Executed water levels, permitted next-state limits and terminal target',
        release_feasibility='Executed release, physical / mapping bounds and ecological requirement',
        reward_reconciliation='Reported reward components and independent reconciliation')
    for key, title in titles.items():
        figures[key], axes = _panels(plt, runs, title,
                                    rows=6 if key in ('mass_balance', 'reward_reconciliation') else 5)
        if key == 'release_feasibility':
            figures[key].supxlabel(
                'Nonfinal plan rejection retains actor control; forced markers denote controls = 0.', fontsize=8)
        for column, run in enumerate(runs):
            trajectory = run['trajectory']
            count = trajectory['steps']
            if not count:
                continue
            x = np.arange(1, count+1)
            series, check, physical = trajectory['series'], trajectory['independent'], trajectory['physical']
            for i, reservoir in enumerate(IDS):
                ax = axes[i, column]
                if key == 'mass_balance':
                    residual = np.asarray(check['mass_balance_residual_m3'])[:, i]
                    tolerance = np.asarray(check['mass_balance_tolerance_m3'])[:, i]
                    ax.plot(x, residual, lw=.8, label='Independent residual')
                    ax.plot(x, tolerance, color='darkorange', lw=.7, label='+ tolerance')
                    ax.plot(x, -tolerance, color='darkorange', lw=.7, ls='--', label='− tolerance')
                    ax.set_yscale('symlog', linthresh=max(float(tolerance.max())*.01, 1e-12))
                    ax.set_ylabel(reservoir+' balance (m³)')
                    if i == 0:
                        maximum = trajectory.get('summary', {}).get('max_abs_cascade_mass_balance_residual_m3')
                        if maximum is not None:
                            ax.text(.02, .96, 'Max whole-cascade residual: {:.3g} m³'.format(maximum),
                                    va='top', transform=ax.transAxes, fontsize=7)
                elif key == 'water_levels':
                    ax.plot(x, np.asarray(series['levels_end'])[:, i], lw=1, label='Executed next-state level')
                    lower, upper = series.get('next_state_level_lower_m'), series.get('next_state_level_upper_m')
                    if lower is not None and all(row is not None for row in lower):
                        ax.plot(x, np.asarray(lower)[:, i], color='gray', lw=.7, label='Lower bound')
                        ax.plot(x, np.asarray(upper)[:, i], color='gray', lw=.7, ls='--', label='Upper bound')
                    else:
                        bounds = payload.get('configuration', {}).get('physical_config', {}).get('level_bounds_m', {}).get(reservoir)
                        if bounds is not None:
                            ax.axhline(bounds[0], color='gray', label='Global lower bound')
                            ax.axhline(bounds[1], color='gray', ls='--', label='Global upper bound')
                    target = physical.get('target_level_m')
                    if target is not None:
                        ax.axhline(target[i], color='purple', ls=':', label='Terminal target (final day only)')
                    elif i == 0:
                        status = 'unavailable' if physical.get('terminal_storage_equal_initial') else 'disabled'
                        ax.text(.02, .03, 'Terminal target '+status, transform=ax.transAxes, fontsize=7)
                    ax.set_ylabel(reservoir+' level (m)')
                elif key == 'release_feasibility':
                    low, high, release = (np.asarray(series[name])[:, i] for name in ('low', 'high', 'release'))
                    physical_low, physical_high = series.get('physical_low_m3s'), series.get('physical_high_m3s')
                    if physical_low is not None and physical_high is not None and all(
                            row is not None for row in physical_low+physical_high):
                        ax.fill_between(x, np.asarray(physical_low)[:, i], np.asarray(physical_high)[:, i],
                                        alpha=.12, label='Physical release bounds')
                    else:
                        ax.fill_between(x, low, high, alpha=.12,
                                        label='Dispatcher bounds (before forced override)')
                    ax.plot(x, release, lw=.9, label='Executed release')
                    ax.plot(x, low, color='gray', lw=.5, label='Mapping lower bound')
                    ax.plot(x, high, color='gray', lw=.5, ls='--', label='Mapping upper bound')
                    threshold = physical['ecological_flow_m3s'][i]
                    if threshold > 0:
                        ax.axhline(threshold, color='green', ls=':', label='Ecological requirement')
                    forced = np.asarray(series['controls'])[:, i] == 0
                    ax.scatter(x[forced], release[forced], marker='x', s=12, color='red',
                               label='Forced action ({} days)'.format(int(forced.sum())))
                    for name, marker, color, label in (
                        ('terminal_plan_rejected', '^', 'darkorange', 'Plan rejection'),
                        ('terminal_fallback', 's', 'black', 'Final fallback')):
                        flags = series.get(name)
                        if flags is not None and all(flag is not None for flag in flags):
                            marked = np.asarray(flags, dtype=bool)
                            ax.scatter(x[marked], release[marked], marker=marker, s=13, color=color,
                                       facecolors='none' if marker == 's' else color,
                                       label=label+' ({} days)'.format(int(marked.sum())))
                    ax.set_ylabel(reservoir+' release (m³/s)')
                else:
                    for name, sign, label in (('local_generation_rewards', 1, 'Local generation'),
                        ('local_reward_penalties', -1, 'Ecological cost (negative)'), ('mixed_rewards', 1, 'Mixed reward')):
                        ax.plot(x, sign*np.asarray(series[name])[:, i], lw=.8, label=label)
                    difference = np.asarray(trajectory['reward_differences']['mixed_rewards'])[:, i]
                    error = ax.twinx()
                    error.plot(x, difference, color='red', ls=':', lw=.7, label='Reported − recomputed mixed reward')
                    error.set_ylabel('Reconciliation error', fontsize=7)
                    error.tick_params(labelsize=7)
                    error.legend(fontsize=6, loc='upper right')
                    ax.set_ylabel(reservoir+' reward')
                if key == 'reward_reconciliation':
                    ax.legend(fontsize=6, loc='lower left')
                else:
                    _legend(ax)
            if key == 'mass_balance':
                ax = axes[5, column]
                residual = np.asarray(check['cascade_mass_balance_residual_m3'])
                tolerance = np.asarray(check['cascade_mass_balance_tolerance_m3'])
                ax.plot(x, residual, lw=.8, label='Independent whole-cascade residual')
                ax.plot(x, tolerance, color='darkorange', lw=.7, label='+ tolerance')
                ax.plot(x, -tolerance, color='darkorange', lw=.7, ls='--', label='− tolerance')
                ax.set_yscale('symlog', linthresh=max(float(tolerance.max())*.01, 1e-12))
                ax.set_ylabel('Whole-cascade balance (m³)')
                _legend(ax)
            elif key == 'reward_reconciliation':
                ax = axes[5, column]
                for name, label in (('global_reward', 'Global generation'), ('system_reward', 'System reward'), ('reward', 'Legacy-team reward')):
                    ax.plot(x, series[name], lw=.9, label=label)
                    ax.plot(x, check[name], lw=.6, ls='--', label='Independent '+label.lower())
                ax.set_ylabel('System reward accounting')
                _legend(ax)
                maximum = trajectory.get('summary', {}).get('max_abs_reward_difference')
                if maximum is not None:
                    ax.text(.02, .03, 'Max reconciliation error (all rewards): {:.3g}'.format(maximum),
                            transform=ax.transAxes, fontsize=7)
    figures['terminal_storage'], axes = _panels(plt, runs,
        'Terminal storage: final batch / final evaluation points; all-history pass counts')
    from matplotlib.ticker import SymmetricalLogLocator
    for column, run in enumerate(runs):
        all_records = run.get('terminal_records', [])
        records = []
        history_labels = []
        for phase, label in (('training', 'Train'), ('training_evaluation', 'Eval'), ('evaluation', 'Held-out')):
            history = [record for record in all_records if record.get('phase') == phase]
            measured = [record for record in history if record.get('enabled') and record.get('available')]
            passed = sum(record.get('met') is True for record in measured)
            disabled = sum(not record.get('enabled', False) for record in history)
            history_labels.append('{}: {}/{} met{}'.format(label, passed, len(measured),
                '; {} disabled'.format(disabled) if disabled else ''))
            if history:
                latest = max(record.get('policy_update', 0) for record in history)
                records.extend(record for record in history if phase == 'evaluation' or
                               record.get('policy_update', 0) == latest)
        labels = []
        for record in records:
            phase = {'training': 'Train', 'training_evaluation': 'Eval',
                     'evaluation': 'Held-out'}.get(record.get('phase'), record.get('phase', 'Case'))
            scenario = record.get('scenario_id', 'case').replace('final_collection_', '').replace('collection_', '').replace('episode_', 'ep')
            if scenario == 'held_out_sequence':
                scenario = 'sequence'
            labels.append('{} {}\nu{}'.format(phase, scenario, record.get('policy_update', '?')))
        for i, reservoir in enumerate(IDS):
            ax = axes[i, column]
            ax.clear()
            ax.grid(alpha=.2)
            if i == 0:
                ax.set_title('Seed {} | all-history terminal counts\n{}'.format(
                    run['seed'], ' | '.join(history_labels)), fontsize=8)
            if i == 4:
                ax.set_xlabel('Recorded final episode / scenario')
            ax.set_ylabel(reservoir+' deviation / tolerance')
            ax.axhline(1., color='darkorange', ls='--', lw=.7, label='± tolerance')
            ax.axhline(-1., color='darkorange', ls='--', lw=.7)
            for position, record in enumerate(records):
                if not record.get('enabled', False):
                    ax.text(position, 0, 'Disabled', rotation=90, fontsize=7, ha='center')
                elif not record.get('available', False):
                    ax.text(position, 0, 'Unavailable / '+record.get('status', 'partial'), rotation=90, fontsize=7, ha='center')
                else:
                    ratio = record['deviation_m3'][i]/record['tolerance_m3'][i]
                    ax.scatter(position, ratio, s=24, color='crimson' if abs(ratio) > 1 else 'tab:blue')
            if not records:
                ax.text(.5, .5, 'Unavailable: no terminal records', ha='center', transform=ax.transAxes)
            ax.set_xticks(range(len(records)), labels, rotation=35, ha='right', fontsize=7)
            ax.set_yscale('symlog', linthresh=1.)
            locator = SymmetricalLogLocator(base=10, linthresh=1.)
            locator.set_params(numticks=7)
            ax.yaxis.set_major_locator(locator)
            _legend(ax)
    figures['happo_updates'], axes = _panels(plt, runs,
        'HAPPO independent likelihood-factor checks and policy-update diagnostics', rows=6)
    colors = plt.get_cmap('tab10')
    for column, run in enumerate(runs):
        records = run.get('update_records', [])
        likelihood_values = {0: [], 2: []}
        for row, ylabel in enumerate(('Likelihood factor quantiles', 'Maximum factor error',
            'Final sampled likelihood ratio', 'Policy KL', 'Clip fraction', 'Policy entropy')):
            ax = axes[row, column]
            ax.clear()
            ax.grid(alpha=.2)
            ax.set_ylabel(ylabel)
            ax.set_xlabel('Completed policy update')
            if row == 0:
                ax.set_title('Seed '+str(run['seed']))
            if not records:
                ax.text(.5, .5, 'Unavailable: no HAPPO update evidence', ha='center', transform=ax.transAxes)
        for agent in range(5):
            actors = [next(actor for actor in record['actors'] if actor['agent_id'] == agent) for record in records]
            updates = np.asarray([record['update'] for record in records])
            if not actors:
                continue
            color = colors(agent)
            for quantile, marker in zip(QUANTILES, ('v', 'o', '^', 's')):
                offset = (agent-2)*.035
                used = [actor['used_factor'][quantile] for actor in actors]
                expected = [actor['expected_factor'][quantile] for actor in actors]
                likelihood_values[0].extend(used+expected)
                axes[0, column].scatter(updates+offset, used, s=15, marker=marker, color=color,
                    label=IDS[agent] if quantile == 'median' else '_nolegend_')
                axes[0, column].scatter(updates+offset, expected, s=30, marker=marker,
                    facecolors='none', edgecolors=color, label='_nolegend_')
            axes[1, column].plot(updates, [actor['factor_max_abs_error'] for actor in actors], color=color, lw=.8, label=IDS[agent]+' absolute')
            axes[1, column].plot(updates, [actor['factor_max_relative_error'] for actor in actors], color=color, lw=.7, ls=':', label=IDS[agent]+' relative')
            for quantile in ('min', 'median', 'p95', 'max'):
                values = [actor['final_sampled_ratio'].get(quantile) for actor in actors]
                likelihood_values[2].extend(values)
                axes[2, column].plot(updates, [np.nan if value is None else value for value in values],
                    color=color, lw=.7, ls='-' if quantile == 'median' else ':',
                    label=IDS[agent] if quantile == 'median' else '_nolegend_')
            for row, field in ((3, 'final_policy_kl'), (4, 'final_clip_fraction'), (5, 'final_entropy')):
                axes[row, column].plot(updates, [np.nan if actor.get(field) is None else actor[field] for actor in actors],
                    color=color, lw=.9, label=IDS[agent])
        axes[1, column].set_yscale('symlog', linthresh=1e-8)
        for row in (0, 2):
            _likelihood_axis(axes[row, column], likelihood_values[row])
        for row in range(6):
            handles, labels = axes[row, column].get_legend_handles_labels()
            if handles:
                axes[row, column].legend(handles, labels, fontsize=6, ncol=2 if row == 1 else 1,
                                         loc='best')
        axes[5, column].ticklabel_format(axis='y', style='plain', useOffset=False)
        axes[0, column].text(.02, .98, 'Filled: used; open: independent\nv=min, o=median, ^=p95, square=max\nColors: '+', '.join(IDS),
            transform=axes[0, column].transAxes, va='top', fontsize=6)
    return {key: figures[key] for key in FIGURE_KEYS}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check-backend', action='store_true', help='Check headless Matplotlib, then exit')
    parser.add_argument('--input', type=Path, help='Evidence JSON; omit to read JSON from standard input')
    parser.add_argument('--output-dir', type=Path, help='Output directory; may contain input evidence, but six PNG targets must be new')
    parser.add_argument('--no-show', action='store_true', help='Use Agg and save figures without opening windows')
    args = parser.parse_args(argv)
    import matplotlib
    if args.no_show or args.check_backend:
        matplotlib.use('Agg', force=True)
    if args.check_backend:
        import matplotlib.pyplot as plt
        print('Headless Matplotlib ready: '+matplotlib.get_backend())
        return 0
    if args.output_dir is None:
        parser.error('--output-dir is required when rendering evidence')
    if args.output_dir.exists() and not args.output_dir.is_dir():
        parser.error('--output-dir must name a directory, not an existing file: '+str(args.output_dir))
    targets = {key: args.output_dir/(key+'.png') for key in FIGURE_KEYS}
    for path in targets.values():
        if path.exists():
            parser.error('Output already exists: '+str(path))
        if args.input is not None and path.resolve() == args.input.resolve():
            parser.error('A figure cannot overwrite its evidence input')
    try:
        payload = json.loads(args.input.read_text(encoding='utf-8') if args.input is not None else sys.stdin.read())
        figures = make_figures(payload)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    import matplotlib.pyplot as plt
    try:
        for key, path in targets.items():
            with path.open('xb') as stream:
                figures[key].savefig(stream, format='png', dpi=125)
            print(str(path.resolve()))
        if not args.no_show:
            plt.show()
    finally:
        for figure in figures.values():
            plt.close(figure)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
