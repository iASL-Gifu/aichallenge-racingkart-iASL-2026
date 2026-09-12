"""Actual map/constraint regressions for passage=True with a blocked horizon."""
from contextlib import redirect_stdout
import io
import math
from types import SimpleNamespace as NS
from unittest.mock import Mock

import numpy as np
import pytest

from multi_purpose_mpc_ros.core import passage_constraints as pc
from multi_purpose_mpc_ros.core.map import Obstacle
from multi_purpose_mpc_ros.core.reference_path import obstacle_center_bounds
from multi_purpose_mpc_ros.core.traffic_work import TrafficWork
from multi_purpose_mpc_ros.v2x_vehicle_tracker import evaluate_lane_width_samples
from .probe_support import configured_mpc
from .test_probe_mpc_integration import path_state
from .test_overtake_session import controller_method


@pytest.fixture(scope='module')
def model():
    return configured_mpc()


@pytest.fixture
def mpc(model):
    path = model.model.reference_path
    path.map.reset_map()
    path.reset_dynamic_constraints()
    # Compare obstacle-preserving MPC bounds; production comparison mode is separate.
    path.unsafe_static_fallback_on_narrow = False
    path.target_lane_idx = 0
    path.is_overtaking = True
    model.set_lane_transition_weights()
    model.set_soft_lateral_reference()
    model._constraint_collapse_detail = None
    wp = path.get_waypoint(47)
    model.model.update_states(wp.x, wp.y, wp.psi)
    model.model.wp_id = 47
    return model


def block_lane(path, wp_id, lane):
    wp = path.get_waypoint(wp_id)
    upper, lower = path.get_lane_bounds(wp_id)[lane]
    lateral = .5 * (upper + lower)
    path.map.add_obstacles([Obstacle(
        wp.x - math.sin(wp.psi) * lateral,
        wp.y + math.cos(wp.psi) * lateral, .725)])


def preview(mpc, lane):
    car = mpc.model
    state = car.temporal_state
    return pc.preview_lane_corridor(
        car.reference_path, car.wp_id, (state.x, state.y, state.psi),
        mpc.N, car.length, car.width, lane,
        guard=mpc.prediction_outer_boundary_guard,
        connection_points=mpc.lane_constraint_connection_points)


@pytest.mark.parametrize('start,blocked_wp,lane', [(47, 62, 0), (47, 62, 2), (187, 196, 0)])
def test_passage_and_actual_mpc_reject_same_blocked_corridor(mpc, start, blocked_wp, lane):
    path = mpc.model.reference_path
    path.target_lane_idx = lane
    wp = path.get_waypoint(start)
    mpc.model.update_states(wp.x, wp.y, wp.psi)
    mpc.model.wp_id = start
    assert preview(mpc, lane) is None
    block_lane(path, blocked_wp, lane)
    before = path_state(path)
    failure = preview(mpc, lane)
    assert failure is not None
    assert path_state(path) == before  # scratch geometry never changes live state
    assert preview(mpc, 2 - lane) is None
    with redirect_stdout(io.StringIO()):
        mpc._init_problem(mpc.N, 0.)
    assert mpc._constraint_collapse_detected
    assert failure['wp'] == mpc._constraint_collapse_detail['wp']


def controller(mpc):
    car = mpc.model
    path = car.reference_path
    target = path.get_waypoint(55)
    tracker = NS(_samples={'d2': [(0., target.x, target.y)]}, velocity=lambda _: (0., 0.))
    return NS(_traffic_work=TrafficWork(), _reference_pathN_center=path,
              _reference_path=path, _carN_center=car, _mpcN_center=mpc,
              _cfg=NS(bicycle_model=NS(length=car.length, width=car.width)),
              _v2x_tracker=tracker, _v2x_vehicle_radius=.725, _passage_clearance=.3,
              _prepass_lane_fallback_prediction_sec=1., _v2x_t_samples=[0., .2, .375],
              _passage_lane_width_tolerance=.05, _passage_lane_width_tolerance_points=2,
              get_logger=Mock(return_value=Mock()))


def test_vehicle_passage_checks_other_traffic_and_invalidates_map_cache(mpc, monkeypatch):
    c = controller(mpc)
    call = controller_method('_vehicle_passage')
    call.__globals__['evaluate_lane_width_samples'] = evaluate_lane_width_samples
    pose = NS(x=mpc.model.temporal_state.x, y=mpc.model.temporal_state.y, theta=0.)
    assert call(c, 'd2', pose)[0][0]
    # The tracked target and static lane widths stay identical. A different
    # vehicle's prediction on the map blocks the final MPC horizon sample.
    block_lane(c._reference_pathN_center, 62, 0)
    with monkeypatch.context() as old:
        old.setattr(pc, 'candidate_corridor_failure', lambda *args: None)
        c._traffic_work.begin_cycle()
        assert call(c, 'd2', pose)[0][0]  # original distance/static-width gates
    c._traffic_work.begin_cycle()
    assert not call(c, 'd2', pose)[0][0]
    c._reference_pathN_center.map.reset_map()
    assert call(c, 'd2', pose)[0][0]  # same cycle, changed occupancy
    block_lane(c._reference_pathN_center, 62, 0)
    assert not call(c, 'd2', pose)[0][0]


def test_corridor_reused_across_targets_but_not_changed_state(mpc, monkeypatch):
    c = controller(mpc)
    pose = NS(x=0., y=0., theta=0.)
    build = Mock(wraps=pc.preview_lane_corridor)
    monkeypatch.setattr(pc, 'preview_lane_corridor', build)
    assert pc.candidate_corridor_failure(c, pose, 0) is None
    assert pc.candidate_corridor_failure(c, pose, 0) is None
    assert build.call_count == 1
    # Active MPC uses the delay-predicted state, not the measured pose here.
    assert build.call_args.args[1] == 47
    assert build.call_args.args[2][0] == mpc.model.temporal_state.x
    mpc.model.temporal_state.x += .01
    pc.candidate_corridor_failure(c, pose, 0)
    assert build.call_count == 2
    c._traffic_work.begin_cycle()
    pc.candidate_corridor_failure(c, pose, 0)
    assert build.call_count == 3


def test_preview_never_erases_blockage_even_if_comparison_mode_is_enabled(mpc):
    path = mpc.model.reference_path
    block_lane(path, 62, 0)
    path.unsafe_static_fallback_on_narrow = True
    try:
        assert preview(mpc, 0) is not None
        assert path.unsafe_static_fallback_on_narrow
        assert path.unsafe_static_fallback_wp_ids == []
    finally:
        path.unsafe_static_fallback_on_narrow = False


def test_added_boundary_invalidates_cached_corridor_without_replacing_map(mpc):
    c = controller(mpc)
    pose = NS(x=0., y=0., theta=0.)
    path = c._reference_pathN_center
    assert pc.candidate_corridor_failure(c, pose, 0) is None
    data = path.map.data
    revision = path.map.revision
    wp = path.get_waypoint(62)
    path.map.add_boundary([wp.static_border_cells])
    assert path.map.data is data
    assert path.map.revision > revision
    assert pc.candidate_corridor_failure(c, pose, 0) is not None


@pytest.mark.parametrize('lane', [0, 1, 2])
def test_obstacle_half_width_is_kept_without_shrinking_artificial_lanes(lane):
    # Course lb/ub already contain the outside body margin. Only the obstacle
    # at -1.0 gets another .8 m; the L0/L1/L2 partition does not affect this.
    assert obstacle_center_bounds(-4., -1., 1.6, lane, .8,
        lower_occupied=False, upper_occupied=True) == pytest.approx((-4., -1.8))
    assert obstacle_center_bounds(1., 4., 1.6, lane, .8,
        lower_occupied=True, upper_occupied=False) == pytest.approx((1.8, 4.))
    assert obstacle_center_bounds(-3.95, 3.95, 1.6, lane, .8,
        lower_occupied=False, upper_occupied=False) == pytest.approx((-3.95, 3.95))


def test_full_width_does_not_add_obstacle_margin_twice():
    assert obstacle_center_bounds(-2., 2., 1.6, None, .8,
        lower_occupied=True, upper_occupied=True) == pytest.approx((-1.2, 1.2))


@pytest.mark.parametrize('lane', [None, 1])
def test_two_metre_free_space_keeps_center_interval_for_one_point_six_metre_body(mpc, monkeypatch, lane):
    path = mpc.model.reference_path
    path.target_lane_idx = lane
    wp = path.get_waypoint(62)
    center = sum(path.get_lane_bounds(62)[1]) / 2.
    cells = tuple((wp.x - math.sin(wp.psi) * lateral,
                   wp.y + math.cos(wp.psi) * lateral)
                  for lateral in (center + 1., center - 1.))
    path.map.add_obstacles([Obstacle(x, y, .15) for x, y in cells])
    monkeypatch.setattr(path, '_compute_free_segments', lambda *a, **k: [cells])
    upper, lower, _ = path.update_path_constraints(
        62, (wp.x, wp.y, wp.psi), 1, mpc.model.length, 1.6, .8,
        connect_lane_from_current_pose=False)
    # Two occupied edges leave .4 m of valid reference-point travel. Do not
    # demand a further 1.6 m inside that interval, even for full-width mode.
    assert upper[0] - lower[0] == pytest.approx(.4)
