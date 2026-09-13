from types import SimpleNamespace as NS
from unittest.mock import Mock
import pytest
from multi_purpose_mpc_ros.core.follow_zone import active, exception, final_request
from multi_purpose_mpc_ros.core.curve_priority import retained_priority


def controller(wp=240, speed=6., lane=0):
    path=NS(segment_lengths=[1.]*354,target_lane_idx=lane)
    return NS(_normal_follow_zones=((216,270),),_normal_follow_prepare_distance=0.,
        _reference_pathN_center=path,_reference_path=path,_carN_center=NS(wp_id=wp),
        _ultra_slow_early_commit_speed=1.5,
        _overtake=NS(target_id='d2',requested_lane=lane,committed=False),
        _v2x_tracker=NS(has_velocity_estimate=lambda t:True,velocity=lambda t:(speed,0.)),
        _committed_corridor=None,_l1_rejoin_traffic_is_clear=Mock(return_value=(True,{})))


@pytest.mark.parametrize('wp,expected',[(194,False),(195,False),(214,False),(215,False),(216,True),(240,True),(270,True),(271,False)])
def test_zone_and_preparation(wp,expected):
    assert active(controller(wp)) is expected


@pytest.mark.parametrize('speed,allowed',[(0.,True),(1.5,True),(1.51,False),(6.,False),(float('nan'),False)])
def test_new_pass_only_ultra_slow(speed,allowed):
    assert exception(controller(speed=speed),'d2') is allowed


def test_existing_pass_hysteresis_does_not_apply_to_new_target():
    c=controller(speed=1.7);c._overtake.committed=True
    assert exception(c,'d2')
    assert not exception(c,'d3')
    c._v2x_tracker.velocity=lambda t:(1.81,0.)
    assert not exception(c,'d2')


def test_ordinary_and_empty_road_request_l1():
    c=controller()
    assert final_request(c,0,None,5.)==(1,True)
    c._overtake.target_id=None
    assert final_request(c,None,None,5.)==(1,True)


def test_occupied_merge_keeps_applied_outer_lane():
    c=controller();c._committed_corridor=NS(lane=2)
    c._l1_rejoin_traffic_is_clear=Mock(return_value=(False,{'side':['d3']}))
    assert final_request(c,1,None,5.)==(2,False)


def test_safe_merge_requests_candidate_only():
    c=controller();c._committed_corridor=NS(lane=0)
    assert final_request(c,0,None,5.)==(1,True)
    assert c._committed_corridor.lane==0


@pytest.mark.parametrize('owner',['_mpc_safety_recovery_active','_prepass_fallback_recovery_active','_parallel_abort_active','_straight_reentry_active'])
def test_recovery_owners_not_overwritten(owner):
    c=controller();setattr(c,owner,True)
    assert final_request(c,None,None,5.)==(None,False)


def test_slow_exception_and_outside_zone_keep_selection():
    assert final_request(controller(speed=1.),2,None,5.)==(2,False)
    assert final_request(controller(wp=271),0,None,5.)==(0,False)


def test_inner_priority_cannot_hold_in_follow_zone():
    c=controller(216);c._curve_lane_priority_zones=[(185,216,0)]
    assert retained_priority(c) is None


def test_previous_curve_does_not_start_l1_return():
    c=controller(215)
    assert final_request(c,0,None,5.)==(0,False)
    c._carN_center.wp_id=216
    assert final_request(c,0,None,5.)==(1,True)


def test_actual_lane_selector_blocks_priority_override_for_normal_car():
    from .test_overtake_session import controller_method
    c=controller()
    method=controller_method('_apply_l2_restricted_zone_policy')
    for lane in (0,2):
        assert method(c,lane,target_vehicle_id='d2',physical_passage={0:True,2:True},
                      conflicts_by_lane={},center_wp=240)==1
    c._v2x_tracker.velocity=lambda t:(1.,0.)
    assert method(c,2,target_vehicle_id='d2',physical_passage={0:True,2:True},
                  conflicts_by_lane={},center_wp=240)==2


def test_race_handoff_cannot_replace_l1_in_zone():
    from .test_overtake_session import controller_method
    c=controller()
    assert controller_method('_start_race_rejoin_handoff')(c,2.,10.,240) is None
    assert not hasattr(c,'_race_rejoin_handoff_active')
