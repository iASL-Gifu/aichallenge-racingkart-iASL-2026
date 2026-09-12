import ast
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from multi_purpose_mpc_ros.core.wall_constraints import wall_center_bounds
from multi_purpose_mpc_ros.core.reference_path import OUTER_COURSE_MARGIN
from .probe_support import configured_mpc


def test_body_half_width_is_not_added_twice():
    # Physical road +/-3m, already inset by .8m, .8m vehicle half-width.
    assert wall_center_bounds(-2.2, 2.2, course_margin=.8, half_width=.8,
                              guard=.1) == pytest.approx((-2.1, 2.1))


@pytest.mark.parametrize('angle', [-.6, -.2, 0., .2, .6])
def test_all_body_corners_stay_inside_straight_wall(angle):
    lb, ub = wall_center_bounds(-2.2, 2.2, course_margin=.8, half_width=.8,
                               guard=.1, heading_error=angle, half_length=1.554)
    extent = .8 * abs(math.cos(angle)) + 1.554 * abs(math.sin(angle))
    assert lb - extent >= -2.9 - 1e-10
    assert ub + extent <= 2.9 + 1e-10


def test_guard_rows_exist_for_all_lanes_and_survive_lane_relaxation():
    mpc = configured_mpc()
    for lane in (None, 0, 1, 2):
        mpc.model.reference_path.target_lane_idx = lane
        mpc.osqp_initialized = False
        mpc._init_problem(mpc.N, 0., lane_relaxation=1.2)
        rows = mpc.A0[-2*mpc.N:].toarray()
        for n in range(1, mpc.N+1):
            assert rows[2*(n-1), n*mpc.nx] == 1.
            assert rows[2*(n-1), n*mpc.nx+1] == mpc.wall_body_center_offset - mpc.wall_body_half_length
            assert rows[2*(n-1)+1, n*mpc.nx+1] == mpc.wall_body_center_offset + mpc.wall_body_half_length
        # A centre inside the old scalar bound can still have a corner outside.
        candidate = np.zeros(rows.shape[1])
        wp = mpc.model.reference_path.get_waypoint(mpc.model.wp_id+1)
        _, upper = wall_center_bounds(wp.lb, wp.ub, course_margin=OUTER_COURSE_MARGIN,
                                      half_width=.5*mpc.model.width,
                                      guard=mpc.prediction_outer_boundary_guard)
        candidate[mpc.nx] = upper - .05
        candidate[mpc.nx+1] = .2
        assert (rows @ candidate)[1] > upper


def controller_method(name):
    source = Path(__file__).parents[1] / 'multi_purpose_mpc_ros/mpc_controller.py'
    tree = ast.parse(source.read_text())
    method = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == name)
    scope = {'math': math}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(source), 'exec'), scope)
    return scope[name]


def test_controller_and_mpc_use_same_wall_reference():
    cfg = SimpleNamespace(bicycle_model=SimpleNamespace(width=1.6),
                          mpc=SimpleNamespace(prediction_outer_boundary_guard=.1))
    wp = SimpleNamespace(x=0., y=0., psi=0., lb=-2.2, ub=2.2)
    c = SimpleNamespace(_cfg=cfg, _car=SimpleNamespace(get_closest_waypoint=lambda x,y: 0),
                        _reference_path=SimpleNamespace(get_waypoint=lambda _: wp))
    method = controller_method('_physical_corridor_state')
    for yaw in (0., .2):
        _, _, lb, ub = method(c, 0., 1., yaw)
        assert (lb,ub) == pytest.approx(wall_center_bounds(
            wp.lb,wp.ub,course_margin=OUTER_COURSE_MARGIN,half_width=.8,guard=.1,
            heading_error=yaw))


def test_final_wall_stop_survives_later_forward_override_but_not_reverse_recovery():
    source = Path(__file__).parents[1] / 'multi_purpose_mpc_ros/mpc_controller.py'
    tree = ast.parse(source.read_text())
    branch = next(n for n in ast.walk(tree) if isinstance(n, ast.If)
                  and ast.unparse(n.test).startswith('wall_fallback_stop and'))
    code = compile(ast.Module(body=[branch], type_ignores=[]), str(source), 'exec')
    for recovering in (False, True):
        scope = dict(wall_fallback_stop=True, recovering_from_stuck=recovering,
                     self=SimpleNamespace(_enable_control=True,_mpc_cfg=SimpleNamespace(a_min=-2.5)),
                     u=[7.5, .1], acc=2.5, bug_acc_enabled=True)
        exec(code, scope)
        assert scope['u'][0] == (7.5 if recovering else 0.)
        assert scope['acc'] == (2.5 if recovering else -2.5)
