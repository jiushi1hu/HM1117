"""Shared scalar/batch release dispatch with joint prefix terminal budgets.

Only current external inflow and explicitly supplied future external volumes
enter the calculation. Prefix release budgets are necessary forecast-based
conditions, not a certificate of feasibility through future operating rules.
An inconsistent nonfinal forecast plan is rejected without removing physically
available actor control. An unreachable final target uses a deterministic,
upstream-priority projection of the desired prefix releases onto physical ranges.
"""
import numpy as np


TERMINAL_TOLERANCE = 1e-8
TERMINAL_MECHANISM = 'prefix_budget_joint_feasibility_v2'


class InfeasibleCascade(RuntimeError):
    pass


def _matrix(value, name, shape=None):
    result = np.asarray(value, dtype=np.float64)
    if result.ndim != 2 or result.shape[1] != 5 or not np.isfinite(result).all():
        raise ValueError(name+' must be a finite batch of five-vectors')
    if shape is not None and result.shape != shape:
        raise ValueError('All dispatch matrices must have the same shape')
    return result


def _vectors(value, name, shape):
    result = np.asarray(value, dtype=np.float64)
    if result.shape not in ((5,), shape) or not np.isfinite(result).all():
        raise ValueError(name+' must be a finite five-vector or matching batch')
    return np.broadcast_to(result, shape)


def terminal_arrays(terminal, shape, dt):
    """Validate/broadcast terminal context and identify the volume source.

    Future volume excludes today's forcing and contains external/incremental
    water only. Legacy callers lacking it explicitly use the issued seven-day
    mean as a constant over the remaining future decisions.
    """
    if not isinstance(terminal, dict):
        raise ValueError('terminal requires target, remaining_days and forecast')
    try:
        target = _vectors(terminal['target'], 'Terminal target', shape)
        forecast = _vectors(terminal['forecast'], 'Terminal forecast', shape)
        raw_days = np.asarray(terminal['remaining_days'], dtype=np.float64)
    except (KeyError, TypeError, OverflowError) as exc:
        raise ValueError('terminal requires target, remaining_days and forecast') from exc
    if (forecast < 0.).any():
        raise ValueError('Terminal forecast inflow must be nonnegative')
    if raw_days.shape == (1,):
        raw_days = raw_days[0]
    if raw_days.shape not in ((), shape[:1]):
        raise ValueError('Terminal remaining_days must be scalar or match the batch')
    if (not np.isfinite(raw_days).all() or (raw_days < 1).any() or
            not np.equal(raw_days, np.floor(raw_days)).all()):
        raise ValueError('Terminal scheduling requires positive integer remaining days')
    days = np.broadcast_to(raw_days, shape[:1]).astype(np.int64)
    if 'future_external_volume_m3' in terminal:
        future = _vectors(terminal['future_external_volume_m3'], 'Future external volume', shape)
        if (future < 0.).any():
            raise ValueError('Future external volume must be nonnegative')
        if ((days == 1)[:, None] & (future != 0.)).any():
            raise ValueError('The final decision has no future external volume')
        model = 'provided_future_external_volume'
    else:
        future = dt*(days-1)[:, None]*forecast
        model = 'constant_next7d_legacy_fallback'
    return target, days, forecast, future, model


def _backward(storage, external, storage_lower, storage_upper, base_lower, base_upper, dt):
    """Project all imposed outlet ranges upstream, retaining conflict locations."""
    lower, upper = base_lower.copy(), base_upper.copy()
    conflicts = lower > upper+TERMINAL_TOLERANCE
    for i in range(3, -1, -1):
        lower[:, i] = np.maximum(lower[:, i], lower[:, i+1]-external[:, i+1]+
                                (storage_lower[:, i+1]-storage[:, i+1])/dt)
        upper[:, i] = np.minimum(upper[:, i], upper[:, i+1]-external[:, i+1]+
                                (storage_upper[:, i+1]-storage[:, i+1])/dt)
        conflicts[:, i] |= lower[:, i] > upper[:, i]+TERMINAL_TOLERANCE
    # Reservoir zero's own storage interval completes the joint feasibility
    # test; subsequent reservoirs have compatible choices by construction.
    first_lower = np.maximum(lower[:, 0], external[:, 0]+(storage[:, 0]-storage_upper[:, 0])/dt)
    first_upper = np.minimum(upper[:, 0], external[:, 0]+(storage[:, 0]-storage_lower[:, 0])/dt)
    conflicts[:, 0] |= first_lower > first_upper+TERMINAL_TOLERANCE
    return lower, upper, conflicts


def dispatch_release(storage, external, actions, storage_lower, storage_upper,
                     release_cap, dt, terminal=None):
    """Return release dispatch for independent rows, each with five reservoirs.

    Inputs are ``(N,5)`` arrays; release_cap may also be a five-vector. Terminal
    vectors accept either shape and remaining_days is scalar or ``(N,)``.
    All returned arrays retain the leading row dimension, including diagnostics.
    No input is modified. Empty batches are supported.
    """
    storage = _matrix(storage, 'Storage')
    shape = storage.shape
    external = _matrix(external, 'External inflow', shape)
    actions = _matrix(actions, 'Normalized actions', shape)
    storage_lower = _matrix(storage_lower, 'Lower storage bounds', shape)
    storage_upper = _matrix(storage_upper, 'Upper storage bounds', shape)
    cap = _vectors(release_cap, 'Release capacity', shape)
    if not np.isscalar(dt) or not np.isfinite(dt) or dt <= 0.:
        raise ValueError('The dispatch time step must be finite and positive')
    if ((external < 0.).any() or (actions < 0.).any() or (actions > 1.).any() or
            (cap <= 0.).any() or (storage_lower > storage_upper).any()):
        raise ValueError('Invalid forcing, actions, capacity or storage bounds')
    physical_lower, physical_upper, physical_conflict = _backward(
        storage, external, storage_lower, storage_upper, np.zeros(shape), cap, dt)
    if physical_conflict.any():
        raise InfeasibleCascade('Empty physically compatible release interval')
    lower, upper = physical_lower, physical_upper
    rejected = fallback = target = days = forecast = future = model = None
    window_low = window_high = plan_conflict = None
    if terminal is not None:
        target, days, forecast, future, model = terminal_arrays(terminal, shape, dt)
        # Sum the complete water budget above each outlet. Upstream target
        # restoration is included in storage-target, so upstream water is neither
        # omitted nor counted twice as an independently forecast total inflow.
        window_high = (np.cumsum(storage-target, axis=1)/dt+
                       np.cumsum(external, axis=1)+np.cumsum(future, axis=1)/dt)
        window_low = window_high-(days-1)[:, None]*cap
        planned_lower, planned_upper, plan_conflict = _backward(
            storage, external, storage_lower, storage_upper,
            np.maximum(0., window_low), np.minimum(cap, window_high), dt)
        rejected = plan_conflict.any(axis=1)
        fallback = rejected & (days == 1)
        lower = np.where(rejected[:, None], physical_lower, planned_lower)
        upper = np.where(rejected[:, None], physical_upper, planned_upper)
    inflow, release, low, high = [np.empty(shape) for _ in range(4)]
    physical_low_all, physical_high_all = np.empty(shape), np.empty(shape)
    controls = np.ones(shape, dtype=np.float32)
    executed = actions.copy()
    binding, infeasible = np.zeros(shape, dtype=bool), np.zeros(shape, dtype=bool)
    reason = np.zeros(shape, dtype=np.int8)
    for i in range(5):
        inflow[:, i] = external[:, i]+(release[:, i-1] if i else 0.)
        low[:, i] = np.maximum(lower[:, i], inflow[:, i]+(storage[:, i]-storage_upper[:, i])/dt)
        high[:, i] = np.minimum(upper[:, i], inflow[:, i]+(storage[:, i]-storage_lower[:, i])/dt)
        if (low[:, i] > high[:, i]+TERMINAL_TOLERANCE).any():
            raise InfeasibleCascade('Empty feasible release interval at reservoir '+str(i))
        collapsed = high[:, i]-low[:, i] <= TERMINAL_TOLERANCE
        physical_low = np.maximum(physical_lower[:, i], inflow[:, i]+
                                  (storage[:, i]-storage_upper[:, i])/dt)
        physical_high = np.minimum(physical_upper[:, i], inflow[:, i]+
                                   (storage[:, i]-storage_lower[:, i])/dt)
        physical_low_all[:, i], physical_high_all[:, i] = physical_low, physical_high
        if terminal is not None:
            above = window_low[:, i] > physical_high+TERMINAL_TOLERANCE
            below = window_high[:, i] < physical_low-TERMINAL_TOLERANCE
            infeasible[:, i] = plan_conflict[:, i] | above | below
            reason[:, i] = np.where(infeasible[:, i], np.where(above, 1, np.where(below, 2, 3)), 0)
            binding[:, i] = (~rejected & ((low[:, i] > physical_low+TERMINAL_TOLERANCE) |
                                          (high[:, i] < physical_high-TERMINAL_TOLERANCE)))
            uncontrolled = collapsed | fallback
        else:
            uncontrolled = collapsed
        controls[uncontrolled, i] = 0.
        executed[uncontrolled, i] = .5
        midpoint = np.minimum(cap[:, i], np.maximum(0., .5*(low[:, i]+high[:, i])))
        release[:, i] = np.where(collapsed, midpoint,
                                 low[:, i]+executed[:, i]**2*(high[:, i]-low[:, i]))
        if terminal is not None:
            # Only an unreachable FINAL target bypasses the actor. Earlier
            # forecast rejections preserve the full physical command mapping.
            desired = np.clip(window_high[:, i], low[:, i], high[:, i])
            desired = np.where(collapsed, midpoint, desired)
            release[:, i] = np.where(fallback, desired, release[:, i])
    result = dict(inflow=inflow, release=release, low=low, high=high,
                  controls=controls, actions=executed,
                  physical_low_m3s=physical_low_all, physical_high_m3s=physical_high_all)
    if terminal is not None:
        result.update(terminal_target=target.copy(), terminal_remaining_days=days.copy(),
            terminal_forecast=forecast.copy(), terminal_future_external_volume_m3=future.copy(),
            terminal_window_low_m3s=window_low, terminal_window_high_m3s=window_high,
            terminal_window_binding=binding, terminal_window_infeasible=infeasible,
            terminal_window_reason=reason, terminal_plan_rejected=rejected, terminal_fallback=fallback,
            terminal_mechanism=np.full(shape[0], TERMINAL_MECHANISM),
            terminal_forecast_model=np.full(shape[0], model),
            terminal=dict(target=target.copy(), remaining_days=days.copy(), forecast=forecast.copy(),
                          future_external_volume_m3=future.copy()))
    return result
