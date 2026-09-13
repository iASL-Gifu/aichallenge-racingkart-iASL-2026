from types import SimpleNamespace as NS, MethodType
import pytest
from multi_purpose_mpc_ros.core.l0_preparation import approaching_zone
from .test_overtake_session import controller_method


@pytest.mark.parametrize('wp,expected', [(294, False),(295, True),(324,True),(325,True),(326,False)])
def test_preview_uses_meters_not_waypoint_count(wp, expected):
    assert approaching_zone(wp, [(325,6)], [1.]*354, 30.) is expected
    assert approaching_zone(wp, [(325,6)], [2.]*354, 60.) is expected


def test_wrap_and_invalid_geometry():
    assert approaching_zone(95, [(5,10)], [1.]*100, 10.)
    assert not approaching_zone(94, [(5,10)], [1.]*100, 10.)
    assert not approaching_zone(95, [(5,10)], [float('nan')]*100, 10.)
    assert not approaching_zone(95, [(5,10)], [1.]*100, 0.)


def test_controller_gate_covers_preparation_and_existing_wrapped_zone():
    c=NS(_l2_entry_restricted_zones=[(325,6)],
         _reference_pathN_center=NS(segment_lengths=[1.]*354),
         _l0_priority_prepare_distance=30.)
    c._waypoint_in_configured_zones=MethodType(controller_method('_waypoint_in_configured_zones'), c)
    gate=controller_method('_l0_priority_zone_active')
    for wp,expected in [(294,False),(295,True),(325,True),(353,True),(0,True),(6,True),(7,False)]:
        assert gate(c,wp) is expected


@pytest.mark.parametrize('target,slow,passage,conflicts,expected', [
    ('d2',True,True,{},True),
    ('d2',False,True,{},False),
    ('d2',True,False,{},False),
    (None,True,True,{},False),
    ('d2',True,True,{'front':['d3']},False),
    ('d2',True,True,{'side':['d3']},False),
    ('d2',True,True,{'rear':['d3']},False),
])
def test_l2_exception_needs_feasible_slow_target(target,slow,passage,conflicts,expected):
    from multi_purpose_mpc_ros.core.l0_preparation import l2_exception
    assert l2_exception(target,slow,passage,conflicts) is expected


def priority_block():
    import ast
    from pathlib import Path
    source = (Path(__file__).parents[1]/'multi_purpose_mpc_ros/mpc_controller.py').read_text()
    start = source.index('        l2_entry_restricted_override_active = False', source.index('# Final L2 geographic policy.'))
    end = source.index('        preserve_hybrid_request =', start)
    import textwrap
    return compile(ast.parse(textwrap.dedent(source[start:end])), '<actual priority block>', 'exec')


@pytest.mark.parametrize('owner', [None, 'recovery_active', '_mpc_safety_recovery_active',
    '_prepass_fallback_follow_active', '_parallel_abort_active', 'startup_overtake_suppressed'])
def test_no_lead_l2_hold_releases_but_other_owners_win(owner):
    from unittest.mock import Mock
    path=object()
    c=NS(_reference_path=path, _reference_pathN_center=path,
         _follow_only=False, _follow_escape_active=False,
         _mpc_safety_recovery_active=False, _post_reverse_full_width_recovery_active=False,
         _prepass_fallback_recovery_active=False, _prepass_fallback_follow_active=False,
         _l1_safety_recovery_active=False, _l1_rejoin_backoff_active=False,
         _parallel_abort_active=False, _l0_priority_zone_active=lambda _:True,
         _overtake=NS(requested_lane=2,probe=NS(lane_idx=2)),
         _prepass_fallback_lane_idx=None, _prepass_fallback_commit_lane_idx=None,
         _prepass_fallback_commit_pending=False, _prepass_fallback_commit_success_since=None,
         _lane_horizon_has_vehicle_width=lambda _:True,
         _relative_lane_vehicle_samples=lambda *a:[],
         _prepass_lane_fallback_front_distance=8., _prepass_lane_fallback_side_distance=2.,
         _prepass_lane_fallback_rear_distance=5., _l0_priority_prepare_distance=30.,
         _apply_l2_restricted_zone_policy=Mock(return_value=0),
         _reset_overtake_commit_probe=Mock(), _clear_prepass_soft_guidance=Mock(),
         _mpc=NS(osqp_initialized=True), get_logger=lambda:Mock())
    ns=dict(self=c, recovery_active=False, startup_overtake_suppressed=False,
            initial_start_lateral_hold_active=False, l0_entry_prohibited_active=False,
            outer_lane_mpc_problem_active=False, center_wp_temp=300,
            prev_lane_idx=2,new_target_lane_idx=2,opponent_vehicle_id=None,
            opponent_ahead_detected=False, pose=object(),v=8.,
            classify_lane_conflicts=lambda *a,**k:{})
    if owner:
        if owner.startswith('_'):setattr(c,owner,True)
        else:ns[owner]=True
    exec(priority_block(),ns)
    assert ns['new_target_lane_idx'] == (2 if owner else 0)
    assert c._overtake.requested_lane == (2 if owner else 0)
    assert c._reset_overtake_commit_probe.call_count == (0 if owner else 1)
