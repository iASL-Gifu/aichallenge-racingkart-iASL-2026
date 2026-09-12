"""PP uses the same lateral objective as the active MPC, including transitions."""
import ast
import math
from contextlib import redirect_stdout
from functools import lru_cache
import io
from pathlib import Path
from types import MethodType, SimpleNamespace as NS

import numpy as np
import pytest

from .probe_support import configured_mpc


@lru_cache(maxsize=None)
def controller_method(name):
    source = Path(__file__).parents[1] / 'multi_purpose_mpc_ros/mpc_controller.py'
    tree = ast.parse(source.read_text())
    node = next(n for n in ast.walk(tree)
                if isinstance(n, ast.FunctionDef) and n.name == name)
    scope = dict(math=math, np=np, Pose2D=NS)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), 'exec'), scope)
    return scope[name]


def controller(lane):
    c = NS(_reference_path=NS(target_lane_idx=lane, n_waypoints=100,
                              get_waypoint=lambda i: NS(x=float(i), y=0., psi=0.)),
           _car=NS(wp_id=98),
           _mpc=NS(_compute_lane_center=lambda i, lane: (-2., 0., 2.)[lane]))
    c._fallback_target_xy = MethodType(controller_method('_fallback_target_xy'), c)
    return c


@pytest.mark.parametrize('lane,lateral', [(0, -2.), (1, 0.), (2, 2.), (None, 0.)])
def test_lane_goal_and_lap_wrap(lane, lateral):
    assert controller(lane)._fallback_target_xy(2) == (2., lateral)


def test_l2_offsets_and_world_rotation():
    c = controller(2)
    c._mpc.target_lane_lateral_offsets = [0., -.3]
    c._reference_path.get_waypoint = lambda i: NS(x=10., y=20., psi=math.pi / 2)
    assert c._fallback_target_xy(2) == pytest.approx((8.3, 20.))


def test_soft_rejoin_and_hybrid_follow_the_blended_goal():
    c = controller(None)
    c._mpc.soft_target_lane_idx = 2
    c._mpc.soft_target_lateral_offset = 0.
    c._mpc.soft_target_start_e_y = -2.
    c._mpc.soft_target_alpha = .25
    assert c._fallback_target_xy(2) == (2., -1.)
    c._reference_path.target_lane_idx = 2
    c._mpc.lane_transition_weights = [0., 1.]
    c._mpc.soft_lateral_targets = [-2., -1., 0., 1., 1.5, 2.]
    c._mpc.soft_target_alpha = 1.
    assert c._fallback_target_xy(2) == (2., 1.5)
    # A soft goal without a synchronized Hybrid transition cannot override L2.
    c._mpc.lane_transition_weights = None
    assert c._fallback_target_xy(2) == (2., 2.)


def test_snapshot_requires_current_waypoint_and_lane_and_holds_terminal_goal():
    c = controller(2)
    c._mpc._fallback_lateral_reference = (98, 2, np.array([.1, .2, .3]))
    assert c._fallback_target_xy(99) == (99., .2)
    assert c._fallback_target_xy(2) == (2., .3)
    c._car.wp_id = 99
    assert c._fallback_target_xy(2) == (2., 2.)
    c._car.wp_id = 98
    c._reference_path.target_lane_idx = 0
    assert c._fallback_target_xy(2) == (2., -2.)


@pytest.mark.parametrize('lane,sign', [(0, -1), (1, 0), (2, 1)])
def test_actual_pp_command_turns_toward_the_selected_lateral_goal(lane, sign):
    c = controller(lane)
    c._car.wp_id = 0
    c._car.get_closest_waypoint = lambda x, y: 0
    c._cfg = NS(bicycle_model=NS(length=1.))
    c._steering_fallback_lookahead_gain = 0.
    c._steering_fallback_min_distance = 4.
    c._steering_fallback_max_distance = 4.
    c._limit_fallback_steering = lambda requested: requested
    angle, _, wp, reason = controller_method('_active_path_pure_pursuit_feedback')(
        c, NS(x=0., y=0., theta=0.), 0.)
    assert np.sign(angle) == sign
    assert wp == 4 and reason == 'ok'


def test_full_width_mpc_objective_is_shared_with_pp_without_changing_it():
    mpc = configured_mpc()
    path = mpc.model.reference_path
    wp = path.get_waypoint(47)
    mpc.model.update_states(wp.x, wp.y, wp.psi)
    mpc.model.wp_id = 47
    path.target_lane_idx = None
    mpc.set_full_width_l0_offset_limits(np.full(mpc.N + 1, .3))
    with redirect_stdout(io.StringIO()):
        mpc._init_problem(mpc.N, 0.)
    c = NS(_mpc=mpc, _car=mpc.model, _reference_path=path)
    reference = mpc._fallback_lateral_reference[2]
    # Compare against the actual QP objective, not a second lane calculation.
    q = mpc.optimizer._derivative_cache['q']
    weights = mpc.optimizer._derivative_cache['P'].diagonal()[:mpc.nx_N]
    np.testing.assert_allclose(reference, (-q[:len(weights)] / weights)[::mpc.nx])
    x, y = controller_method('_fallback_target_xy')(c, 50)
    target = path.get_waypoint(50)
    assert (x, y) == pytest.approx((target.x - reference[3] * math.sin(target.psi),
                                       target.y + reference[3] * math.cos(target.psi)))
