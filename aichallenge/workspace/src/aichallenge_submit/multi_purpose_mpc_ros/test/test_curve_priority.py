"""Curve preference must share selection/hold ownership without bypassing safety."""
from types import SimpleNamespace as NS
from unittest.mock import Mock
import pytest
from multi_purpose_mpc_ros.core.curve_priority import priority_at, select
from .test_overtake_session import controller_method
from .test_l0_preparation import priority_block

ZONES=[(30,60,0),(80,105,2),(185,215,0)]

@pytest.mark.parametrize('wp,lane',[(29,0),(30,0),(60,0),(61,2),(80,2),(105,2),
                                   (106,None),(154,None),(155,0),(185,0),(215,0),(216,None)])
def test_current_curve_wins_over_next_curve_approach(wp,lane):
    assert priority_at(wp,ZONES,[1.]*354,30.) == lane


def test_nearest_zone_and_meter_distance_and_invalid_geometry():
    assert priority_at(65,ZONES,[2.]*354,30.) == 2
    assert priority_at(64,ZONES,[2.]*354,30.) is None
    assert priority_at(350,[(5,10,2)],[1.]*354,9.) == 2
    assert priority_at(1,ZONES,[float('nan')]*354,30.) is None
    assert priority_at(60,ZONES,[1.]*354,100.) == 0


@pytest.mark.parametrize('preferred',[0,2])
@pytest.mark.parametrize('case,expected', [('follow','preferred'),('side',1),('narrow',1),
    ('slow_clear','opposite'),('slow_blocked','preferred'),('ordinary_clear','preferred')])
def test_inner_follow_and_feasible_slow_overtake_exception(preferred,case,expected):
    opposite=2-preferred
    c=NS(_l2_restricted_slow_override=lambda _:case.startswith('slow'),
         _lane_horizon_has_vehicle_width=lambda _:case!='narrow')
    conflicts={preferred:{'front':['lead']},opposite:{}}
    if case=='side':conflicts[preferred]['side']=['other']
    if case=='slow_blocked':conflicts[opposite]['rear']=['other']
    result=select(c,preferred,opposite,'lead',{preferred:False,opposite:True},conflicts)
    assert result == {'preferred':preferred,'opposite':opposite}.get(expected,expected)


@pytest.mark.parametrize('preferred,wp',[(0,40),(2,90)])
@pytest.mark.parametrize('phase', ['ordinary', 'pending', 'committed'])
@pytest.mark.parametrize('owner',[None,'recovery_active','_mpc_safety_recovery_active',
    '_parallel_abort_active','startup_overtake_suppressed','_follow_escape_active'])
def test_final_request_clears_opposing_owners_and_preserves_recovery(preferred,wp,owner,phase):
    opposite=2-preferred
    path=NS(segment_lengths=[1.]*354)
    c=NS(_curve_lane_priority_zones=ZONES,_curve_lane_priority_prepare_distance=30.,
         _reference_path=path,_reference_pathN_center=path,
         _follow_only=False,_follow_escape_active=False,_mpc_safety_recovery_active=False,
         _post_reverse_full_width_recovery_active=False,_prepass_fallback_recovery_active=False,
         _prepass_fallback_follow_active=False,_l1_safety_recovery_active=False,
         _l1_rejoin_backoff_active=False,_parallel_abort_active=False,
         _l0_priority_zone_active=lambda _:preferred==0,
         _overtake=NS(requested_lane=opposite,probe=NS(lane_idx=opposite)),
         _prepass_fallback_lane_idx=opposite,_prepass_fallback_commit_lane_idx=opposite,
         _prepass_fallback_commit_pending=True,_prepass_fallback_commit_success_since=1.,
         _lane_horizon_has_vehicle_width=lambda _:True,_relative_lane_vehicle_samples=lambda *a:[],
         _prepass_lane_fallback_front_distance=8.,_prepass_lane_fallback_side_distance=2.,
         _prepass_lane_fallback_rear_distance=5.,_l0_priority_prepare_distance=50.,
         _l2_restricted_slow_override=lambda _:False,
         _reset_overtake_commit_probe=Mock(),_clear_prepass_soft_guidance=Mock(),
         _mpc=NS(osqp_initialized=True),get_logger=lambda:Mock())
    c._apply_l2_restricted_zone_policy=lambda candidate,**kw: controller_method(
        '_apply_l2_restricted_zone_policy')(c,candidate,**kw)
    ns=dict(self=c,recovery_active=False,startup_overtake_suppressed=False,
        initial_start_lateral_hold_active=False,l0_entry_prohibited_active=False,
        outer_lane_mpc_problem_active=False,center_wp_temp=wp,prev_lane_idx=opposite,
        new_target_lane_idx=opposite,opponent_vehicle_id=None,opponent_ahead_detected=False,
        pose=object(),v=8.,classify_lane_conflicts=lambda *a,**k:{})
    if phase == "ordinary":
        c._prepass_fallback_lane_idx = None
        c._prepass_fallback_commit_lane_idx = None
        c._prepass_fallback_commit_pending = False
    elif phase == "committed":
        c._prepass_fallback_commit_lane_idx = None
        c._prepass_fallback_commit_pending = False
    protected = bool(owner or phase != "ordinary")
    if owner:
        if owner.startswith('_'):setattr(c,owner,True)
        else:ns[owner]=True
    exec(priority_block(),ns)
    assert ns['new_target_lane_idx'] == (opposite if protected else preferred)
    assert c._overtake.requested_lane == (opposite if protected else preferred)
    assert c._prepass_fallback_lane_idx == (None if phase == 'ordinary' else opposite)
    assert c._reset_overtake_commit_probe.call_count == (0 if protected else 1)
    assert c._clear_prepass_soft_guidance.call_count == (0 if protected else 1)


def test_l2_curve_suppresses_l0_preparation_and_legacy_end_zone_survives():
    from types import MethodType
    c=NS(_curve_lane_priority_zones=ZONES,_curve_lane_priority_prepare_distance=30.,
         _l2_entry_restricted_zones=[(325,6)],_l0_priority_prepare_distance=50.,
         _reference_pathN_center=NS(segment_lengths=[1.]*354))
    c._waypoint_in_configured_zones=MethodType(controller_method('_waypoint_in_configured_zones'),c)
    gate=controller_method('_l0_priority_zone_active')
    assert gate(c,40)
    assert not gate(c,90)
    assert gate(c,190)
    assert gate(c,300)
    assert gate(c,350)


@pytest.mark.parametrize('wp,expected', [(154,None),(155,None),(174,None),(175,0),(184,0),(185,0),(215,0),(216,None)])
def test_zone_specific_preparation_delays_only_wp185_approach(wp,expected):
    assert priority_at(wp,ZONES,[1.]*354,30.,{(185,215,0):10.}) == expected


def test_zone_specific_distance_preserves_other_curves_and_shared_controller_policy():
    from multi_purpose_mpc_ros.core.curve_priority import controller_priority
    override={(185,215,0):10.}
    for wp in range(0,140):
        assert priority_at(wp,ZONES,[1.]*354,30.,override)==priority_at(wp,ZONES,[1.]*354,30.)
    c=NS(_curve_lane_priority_zones=ZONES,_curve_lane_priority_prepare_distance=30.,
        _curve_lane_priority_prepare_distances=override,
        _reference_pathN_center=NS(segment_lengths=[1.]*354))
    assert controller_priority(c,160) is None
    assert controller_priority(c,175)==0


def test_zero_preview_starts_at_entry_and_invalid_preview_is_not_used():
    override={(185,215,0):0.}
    assert priority_at(184,ZONES,[1.]*354,30.,override) is None
    assert priority_at(185,ZONES,[1.]*354,30.,override)==0
    assert priority_at(184,ZONES,[1.]*354,30.,{(185,215,0):float('nan')}) is None
