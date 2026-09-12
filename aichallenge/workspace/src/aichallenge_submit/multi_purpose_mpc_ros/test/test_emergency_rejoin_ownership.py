"""Emergency target survives normal L1 rejoin until follow/reverse resolves."""
import ast
import copy
from types import SimpleNamespace as NS, MethodType
from unittest.mock import Mock
from multi_purpose_mpc_ros.overtake_session import OvertakeSession
from multi_purpose_mpc_ros.v2x_vehicle_tracker import prepass_recovery_owns_lane_selection
from .test_four_vehicle_deadlock import method
from .test_hybrid_integration import control_tree


def controller():
    c=NS(_overtake=OvertakeSession(),_follow_last_released_vehicle_id=None,
         _follow_last_released_at=None,_follow_emergency_reacquire_sec=1.,
         _stuck_recovery_until=None,_prepass_retry_after_reverse=False,
         _prepass_fallback_follow_active=False,_prepass_fallback_recovery_active=False,
         _prepass_fallback_commit_pending=False,_post_reverse_full_width_recovery_active=False,
         _follow_escape_active=False,_center_lane_rejoin_active=True,
         _l1_safety_reprobe_pending=True,_l1_rejoin_backoff_active=False,
         _l1_probe_active=True,_l1_probe_context='rejoin',
         _l1_safety_recovery_context='rejoin',
         _vehicle_passage=lambda *a:({0:True,2:False},3.6),
         _relative_lane_vehicle_samples=lambda *a:[],
         _prepass_lane_fallback_front_distance=8.,_prepass_lane_fallback_side_distance=4.,
         _prepass_lane_fallback_rear_distance=5.,_reset_l1_rejoin_backoff=Mock(),
         _reset_overtake_state_for_target_change=Mock(),
         get_clock=lambda:NS(now=lambda:NS(nanoseconds=10_000_000_000)),
         get_logger=lambda:Mock())
    c._cancel_normal_l1_rejoin_for_prepass=MethodType(method('_cancel_normal_l1_rejoin_for_prepass'),c)
    return c


def selection(c):
    tree=control_tree()
    assignments=[next(n for n in ast.walk(tree) if isinstance(n,ast.Assign)
                     and any(isinstance(t,ast.Name) and t.id==name for t in n.targets))
                 for name in ('prepass_selection_exclusive','exclusive_l1_rejoin')]
    release=next(n for n in ast.walk(tree) if isinstance(n,ast.If)
                 and ast.unparse(n.test)=='exclusive_l1_rejoin')
    env=dict(self=c,recovery_active=False,
             prepass_recovery_owns_lane_selection=prepass_recovery_owns_lane_selection)
    exec(compile(ast.Module(body=copy.deepcopy(assignments+[release]),type_ignores=[]),'<selection>','exec'),env)
    return env


def test_repeated_emergency_ticks_keep_one_target_and_reverse_progress():
    c=controller();arm=method('_arm_emergency_blocker_recovery')
    arm(c,'d4',NS(),0.)
    assert not c._center_lane_rejoin_active
    assert c._l1_safety_reprobe_pending  # physical safety is not cancelled
    c._prepass_reverse_distance=.12
    for _ in range(20):
        assert not selection(c)['exclusive_l1_rejoin']
        arm(c,'d4',NS(),0.)
        assert c._overtake.target_id=='d4'
        assert c._prepass_retry_lane_idx==0
        assert c._prepass_reverse_distance==.12
    c._reset_overtake_state_for_target_change.assert_called_once()


def test_follow_target_is_also_preserved_when_no_outer_pass_is_available():
    c=controller();c._overtake.target_id='d4';c._prepass_fallback_follow_active=True
    assert not selection(c)['exclusive_l1_rejoin']
    assert c._overtake.target_id=='d4'


def test_normal_completed_pass_still_hands_back_to_l1():
    c=controller();c._overtake.target_id='old';c._prepass_fallback_lane_idx=None
    assert selection(c)['exclusive_l1_rejoin']
    assert c._overtake.target_id is None

