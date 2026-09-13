from types import SimpleNamespace as NS, MethodType
from unittest.mock import Mock
import math
import pytest
from multi_purpose_mpc_ros.core.path_check_work import PathCheckWork,prepare_bodies
from multi_purpose_mpc_ros.core.boundary_recovery import Motion,rollout
from multi_purpose_mpc_ros.core.recovery_diagnostics import log_event,routine_log_due
from .test_boundary_recovery import choose,recovery_controller,static_map
from .test_overtake_session import controller_method
from .test_slow_pass_spacing_release import controller
from multi_purpose_mpc_ros import collision_geometry as cg


def test_preparing_short_left_is_retained_after_timer_without_long_path():
    c=recovery_controller()
    points=[NS(x=float(i),y=.5*i,psi=.7) for i in range(8)]
    c._reference_path=NS(n_waypoints=8,circular=False,get_waypoint=lambda i:points[i])
    c._car=NS(get_closest_waypoint=lambda *a:0)
    c._cfg=NS(bicycle_model=NS(length=1.087));c._mpc_cfg.delta_max=.314
    c._straight_reentry_probe_distance=2.;c._reentry_overlap=lambda p:0.
    c._reentry_phase='preparing';c._straight_reentry_direction=1;c._reentry_steering=.314
    c._reentry_hold_until=0.;c._last_u=[0.,.314]
    path=rollout((0.,0.,0.),1,.314,.15,1.087)
    c._reentry_approved_motion=Motion(1,.314,path,.1,speed_limit=.17)
    c._reentry_path_is_clear=lambda p,**kw:(sum(math.hypot(b[0]-a[0],b[1]-a[1]) for a,b in zip(p,p[1:]))<=.151,'new_wall_contact_at_step=4')
    motion,reason=controller_method('_select_reentry_motion')(c,NS(x=0.,y=0.,theta=0.),10.)
    assert motion.direction==1 and motion.steering==.314 and reason.startswith('retained:')
    c._reentry_path_is_clear=lambda *a,**kw:(False,'vehicle_collision=new_car')
    motion,_=controller_method('_select_reentry_motion')(c,NS(x=0.,y=0.,theta=0.),10.1)
    assert motion is None and not c._recovery_attempts.wall_failures


@pytest.mark.parametrize('hazard,overlap,swept,expected',[(False,False,True,True),(True,False,True,False),(False,True,True,False),(False,False,False,False)])
def test_longitudinal_release_requires_current_and_future_separation(hazard,overlap,swept,expected):
    c=controller();c._dynamic_gap_swept_clear=Mock(return_value=swept)
    result=controller_method('_fresh_prediction_releases_dynamic_gap')(c,'d2',dict(rectangles_overlap=overlap,lateral_gap=.8),dict(collision=hazard))
    assert result is expected


def test_actual_sweep_catches_collision_between_prediction_points(monkeypatch):
    c=controller();m=c._mpc;c._collision_path_pose=(0.,0.,0.)
    m.collision_prediction_context=(m.current_prediction,)
    c._delay_prediction_enabled=False
    # Target d2 is at x=8; neither endpoint is in contact.
    monkeypatch.setattr('multi_purpose_mpc_ros.core.path_check_work.prepare_mpc_path',lambda *a:(((0.,0.,0.),(16.,0.,0.)),(0.,2.)))
    assert not controller_method('_dynamic_gap_swept_clear')(c,'d2')


def test_points_reused_across_shortened_paths(monkeypatch):
    c=NS(_path_check_work=PathCheckWork(),_collision_now=1.)
    convert=Mock(side_effect=lambda c,p:cg.BodyPose(p.x,p.y,p.theta,1.))
    monkeypatch.setattr(cg,'ego_body',convert)
    path=tuple((i*.05,0.,0.) for i in range(20))
    prepare_bodies(c,path);prepare_bodies(c,path[:10]);assert convert.call_count==20
    c._collision_now=2.;prepare_bodies(c,path[:10]);assert convert.call_count==30


def test_prefix_rejection_reused_but_terminal_rejection_is_not():
    m=static_map();m.begin_collision_cycle()
    bodies=tuple(cg.BodyPose(i*.05,0.,0.,0.) for i in range(8));g=cg.BodyGeometry()
    compute=Mock(return_value=(False,'new_wall_contact_at_step=2'))
    m._compute_static_recovery_samples_are_clear=compute
    m._static_recovery_samples_are_clear(bodies,g,padding=.1)
    m._static_recovery_samples_are_clear(bodies[:4],g,padding=.1)
    assert compute.call_count==1
    m._static_recovery_samples_are_clear(bodies[:4],g,padding=.2);assert compute.call_count==2
    m.begin_collision_cycle();compute.return_value=(False,'wall_overlap_not_reduced')
    m._static_recovery_samples_are_clear(bodies[:4],g,padding=.1)
    m._static_recovery_samples_are_clear(bodies,g,padding=.1);assert compute.call_count==4


def test_minimal_logging_skips_factory_and_throttles_before_formatting():
    c=NS(_cfg=NS(mpc=NS(minimal_logging=True)))
    factory=Mock(side_effect=AssertionError('expensive details generated'))
    log_event(c,'enter',extra_factory=factory);factory.assert_not_called()
    assert routine_log_due(c,'same')
    assert not routine_log_due(c,'same')
