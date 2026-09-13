"""Work reuse must preserve fresh safety checks and periodic steering choices."""
from types import SimpleNamespace as NS
from unittest.mock import Mock
import pytest
from multi_purpose_mpc_ros import collision_geometry as collision
from multi_purpose_mpc_ros.core.boundary_recovery import RecoveryAttempts
from .test_boundary_recovery import choose, static_map
from .test_overtake_session import controller_method


def test_static_evidence_reused_without_aliasing_and_reset_each_tick():
    m = static_map()
    m.data_backup[50,50] = 0
    body = collision.BodyPose(0.,0.,0.,0.)
    geometry = collision.BodyGeometry()
    compute = Mock(wraps=m._compute_static_body_collision_detail)
    m._compute_static_body_collision_detail = compute
    m.begin_collision_cycle()
    expected = m.static_body_collision_detail(body, geometry, .05, include_cells=True)
    evidence = m.static_body_collision_detail(body, geometry, .05, include_cells=True)
    evidence['occupied_cells'].clear()
    assert m.static_body_collision_detail(body, geometry, .05, include_cells=True) == expected
    assert compute.call_count == 1
    m.static_body_collision_detail(body, geometry, .1, include_cells=True)
    m.static_body_collision_detail(collision.BodyPose(.1,0.,0.,0.), geometry, .05, include_cells=True)
    assert compute.call_count == 3
    m.data_backup[:] = 1
    m.begin_collision_cycle()
    assert m.static_body_collision_detail(body, geometry, .05, include_cells=True) is None
    assert compute.call_count == 4


def test_forecasts_share_only_exact_position_velocity_and_arrival_time():
    collision.begin_prediction_cycle()
    assert collision.linear_prediction_position(1.,2.,[3.,4.],.5) == (2.5,4.)
    assert collision.linear_prediction_position(1.,2.,(3.,4.),.5) == (2.5,4.)
    assert collision._linear_prediction_position.cache_info().hits == 1
    assert collision.linear_prediction_position(1.,2.,(3.,4.),1.) == (4.,6.)
    assert collision.linear_prediction_position(2.,2.,(3.,4.),.5) == (3.5,4.)
    assert collision.linear_prediction_position(1.,2.,(1.,4.),.5) == (1.5,4.)
    collision.begin_prediction_cycle()
    assert collision._linear_prediction_position.cache_info().currsize == 0


def test_retained_path_checked_every_call_and_hazard_triggers_comparison():
    clear = Mock(return_value=(True,'clear'))
    for x in (0.,.05,.1):
        motion, _ = choose(pose=(x,0.,0.), retained=(1,0.),
                           compare_retained_turns=False, clear=clear)
        assert motion.poses[0][0] == x
    assert clear.call_count == 3
    clear.side_effect = lambda path: (path[-1][1] > .1, 'vehicle_collision=car')
    motion, _ = choose(retained=(1,0.), compare_retained_turns=False, clear=clear,
                       target=(3.5,2.), target_heading=.7)
    assert motion is not None and motion.steering > 0.


def test_periodic_comparison_restores_turn_without_switching_for_tiny_difference():
    motion, _ = choose(target=(3.5,2.), target_heading=.7, retained=(1,0.),
                       compare_retained_turns=False, recompare=True)
    assert motion.steering > 0.
    motion, reason = choose(target=(3.5,.01), target_heading=.01, retained=(1,0.),
                            compare_retained_turns=False, recompare=True)
    assert motion.steering == 0. and reason.startswith('retained:')


def test_controller_comparison_interval_and_rollback():
    points = [NS(x=float(i),y=float(i)*.5,psi=.7) for i in range(8)]
    attempts = RecoveryAttempts()
    attempts.improving_key=(1,0)
    attempts.improving_until=10.
    clear = Mock(return_value=(True,'clear'))
    c=NS(_reference_path=NS(n_waypoints=8,circular=False,get_waypoint=lambda i:points[i]),
         _car=NS(get_closest_waypoint=lambda *a:0),
         _cfg=NS(bicycle_model=NS(length=1.087)),_mpc_cfg=NS(delta_max=.314),
         _mpc=NS(max_steering_rate=2.),_velocity_report=NS(longitudinal_velocity=.2),
         _last_u=[.2,0.],_straight_reentry_probe_distance=2.,_straight_reentry_speed=1.,
         _straight_reentry_direction=1,_reentry_steering=0.,_reentry_hold_until=0.,
         _reentry_overlap=lambda p:0.,_reentry_path_is_clear=clear,_recovery_attempts=attempts,
         _reentry_comparison_at=1.)
    c._reentry_comparison_key=((1,0),id(c._reference_path))
    select=controller_method('_select_reentry_motion')
    assert select(c,NS(x=0.,y=0.,theta=0.),1.01)[0].steering == 0.
    assert clear.call_count == 1
    assert select(c,NS(x=.03,y=0.,theta=0.),1.1)[0].steering == 0.
    assert clear.call_count == 2
    assert select(c,NS(x=.06,y=0.,theta=0.),1.26)[0].steering > 0.
    assert c._reentry_comparison_at == 1.26
    assert select(c,NS(x=.06,y=0.,theta=0.),.5)[0].steering > 0.
    assert c._reentry_comparison_at == .5


def test_wall_bounds_share_exact_geometry_but_not_changed_heading_or_width():
    from multi_purpose_mpc_ros.core.wall_constraints import wall_center_bounds
    collision.begin_prediction_cycle()
    options=dict(course_margin=.1,half_width=.725,guard=.1,heading_error=.2)
    first=wall_center_bounds(-2.,2.,**options)
    assert wall_center_bounds(-2.,2.,**options) == first
    assert wall_center_bounds.cache_info().hits == 1
    assert wall_center_bounds(-2.,2.,**dict(options,heading_error=.3)) != first
    assert wall_center_bounds(-2.,2.,**dict(options,half_width=.8)) != first
    assert wall_center_bounds(-2.,1.,**options) != first
    collision.begin_prediction_cycle()
    assert wall_center_bounds.cache_info().currsize == 0
