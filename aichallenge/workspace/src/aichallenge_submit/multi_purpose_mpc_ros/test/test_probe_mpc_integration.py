"""Real path mutation and real OSQP problem construction, without ROS startup."""
from contextlib import redirect_stdout
import io
import pickle
from types import MethodType, SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from .probe_support import configured_mpc
from .test_overtake_session import controller_method
from multi_purpose_mpc_ros.overtake_session import OvertakeSession, ShadowProbe


def path_state(path):
    return pickle.dumps((path.target_lane_idx, path.is_overtaking, path.waypoints,
                         path.border_cells, path.last_constraint_bounds,
                         path.unsafe_static_fallback_wp_ids, path.path_constraints))


@pytest.fixture(scope='module', params=['center', 'race'])
def mpc(request):
    return configured_mpc(request.param)


@pytest.mark.parametrize('fail', [False, True])
def test_shadow_solve_and_exception_leave_live_path_unchanged(mpc, fail):
    path = mpc.model.reference_path
    before = path_state(path)
    probe = controller_method('_probe_corridor')
    class ProbeFailure(Exception):
        pass
    try:
        with probe(None, mpc.model, 0) as snapshot:
            wp = snapshot.get_waypoint(30)
            mpc.model.update_states(wp.x, wp.y, wp.psi)
            with redirect_stdout(io.StringIO()):
                mpc.get_control()
            assert snapshot.last_constraint_bounds is not None
            assert path_state(path) == before
            if fail:
                raise ProbeFailure()
    except ProbeFailure:
        assert fail
    assert mpc.model.reference_path is path
    assert mpc.model.current_waypoint is path.get_waypoint(mpc.model.wp_id)
    assert path_state(path) == before


@pytest.mark.parametrize('lane', [0, 2])
def test_artificial_blend_preserves_hard_bounds(mpc, lane):
    probe = controller_method('_probe_corridor')
    snapshots = []
    for weight in (0.0, 0.5, 1.0):
        with probe(None, mpc.model, lane) as path:
            wp = path.get_waypoint(30)
            mpc.model.update_states(wp.x, wp.y, wp.psi)
            mpc.set_lane_transition_weights(np.full(mpc.N, weight))
            with redirect_stdout(io.StringIO()):
                mpc._init_problem(mpc.N, 0.0)
            snapshots.append(path.last_constraint_bounds)
    mpc.set_lane_transition_weights()
    for snapshot in snapshots[1:]:
        np.testing.assert_allclose(snapshot.hard_lb, snapshots[0].hard_lb)
        np.testing.assert_allclose(snapshot.hard_ub, snapshots[0].hard_ub)
        assert np.all(snapshot.final_lb >= snapshot.hard_lb - 1e-9)
        assert np.all(snapshot.final_ub <= snapshot.hard_ub + 1e-9)
    if lane == 0:
        assert np.all(snapshots[0].lane_ub >= snapshots[1].lane_ub)
        assert np.all(snapshots[1].lane_ub >= snapshots[2].lane_ub)
        assert np.any(snapshots[0].lane_ub > snapshots[2].lane_ub)
    else:
        assert np.all(snapshots[0].lane_lb <= snapshots[1].lane_lb)
        assert np.all(snapshots[1].lane_lb <= snapshots[2].lane_lb)
        assert np.any(snapshots[0].lane_lb < snapshots[2].lane_lb)


def test_hybrid_objective_overrides_lane_only_with_matching_boundary_weights(mpc):
    with controller_method('_probe_corridor')(None, mpc.model, 2):
        wp = mpc.model.reference_path.get_waypoint(30)
        mpc.model.update_states(wp.x, wp.y, wp.psi)
        targets = np.linspace(-0.2, 0.4, mpc.N + 1)
        mpc.set_soft_lateral_reference(lane_idx=2, alpha=1.0, lateral_targets=targets)
        mpc.set_lane_transition_weights(np.linspace(0.0, 1.0, mpc.N))
        with redirect_stdout(io.StringIO()):
            mpc._init_problem(mpc.N, 0.0)
        q = mpc.optimizer._derivative_cache['q']
        cost = np.concatenate([np.tile(mpc.Q.diagonal(), mpc.N), mpc.QN.diagonal()])
        np.testing.assert_allclose(q[:mpc.nx*(mpc.N+1):mpc.nx], -cost[::mpc.nx]*targets)
        # A normal hard lane must still win over unrelated soft objectives.
        mpc.set_lane_transition_weights()
        with redirect_stdout(io.StringIO()):
            mpc._init_problem(mpc.N, 0.0)
        hard_targets = [mpc._compute_lane_center(mpc.model.wp_id+n, 2) for n in range(mpc.N+1)]
        q = mpc.optimizer._derivative_cache['q']
        np.testing.assert_allclose(q[:mpc.nx*(mpc.N+1):mpc.nx], -cost[::mpc.nx]*hard_targets)
        mpc.set_soft_lateral_reference()


@pytest.mark.parametrize('kind', ['overtake', 'race'])
@pytest.mark.parametrize('fail', [False, True])
def test_real_probe_call_sites_are_isolated_and_clear_hybrid_weights(mpc, kind, fail, monkeypatch):
    path = mpc.model.reference_path
    c = SimpleNamespace(
        _overtake=OvertakeSession(probe=ShadowProbe(vehicle_id='d2', lane_idx=0)),
        _mpc_safety_recovery_active=False, _post_reverse_full_width_recovery_active=False,
        _reference_path=path, _reference_pathN_center=path,
        _mpcN_overtake_commit_probe=mpc, _carN_overtake_commit_probe=mpc.model,
        _mpcN_center=SimpleNamespace(previous_steering=0.0),
        _update_l2_target_objective_offsets=Mock(),
        _candidate_lane_has_valid_width_ahead=Mock(return_value=(True, None)),
        _overtake_commit_max_relaxation=0.2, _overtake_commit_probe_required_success_cycles=2,
        _race_rejoin_handoff_active=True, _race_rejoin_handoff_guidance_ready=True,
        _race_rejoin_probe_started_at=10.0, _race_rejoin_probe_timeout_sec=3.0,
        _race_rejoin_probe_confirmed=False, _race_rejoin_probe_success_cycles=0,
        _race_rejoin_probe_required_success_cycles=2,
        _mpcN_race=mpc, _carN_race=mpc.model,
        get_logger=Mock(return_value=Mock()))
    c._probe_corridor = MethodType(controller_method('_probe_corridor'), c)
    before = path_state(path)
    solve = mpc.get_control
    class ProbeFailure(Exception):
        pass
    def checked_solve():
        assert mpc.model.reference_path is not path
        assert mpc.lane_transition_weights is None
        result = solve()
        assert path_state(path) == before
        if fail:
            raise ProbeFailure()
        return result
    monkeypatch.setattr(mpc, 'get_control', checked_solve)
    mpc.set_lane_transition_weights(np.ones(mpc.N))
    wp = path.get_waypoint(30)
    pose = SimpleNamespace(x=wp.x, y=wp.y, theta=wp.psi)
    with redirect_stdout(io.StringIO()):
        try:
            if kind == 'overtake':
                controller_method('_run_overtake_commit_probe')(c, pose, False)
            else:
                controller_method('_run_race_rejoin_probe')(c, pose, False, 10.1, True, False)
        except ProbeFailure:
            assert fail
    assert mpc.model.reference_path is path
    assert path_state(path) == before
