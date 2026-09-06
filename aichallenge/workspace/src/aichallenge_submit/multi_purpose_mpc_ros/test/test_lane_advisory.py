"""Traffic recommendations cannot reset an accepted lateral manoeuvre."""
import ast
import copy
from types import SimpleNamespace as NS
from unittest.mock import Mock

import pytest

from multi_purpose_mpc_ros.lane_evaluation import LaneObstruction, propose_lane
from multi_purpose_mpc_ros.overtake_session import LaneDecision
from .test_overtake_session import active_session, controller_method
from .test_hybrid_integration import control_tree


CLEAR = {0: {}, 2: {}}
PASSAGE = {0: True, 2: True}


def test_three_opponents_rank_prefix_without_requiring_one_lane_for_everyone():
    rows = (LaneObstruction('d2', 8., ()), LaneObstruction('d3', 25., (2,)),
            LaneObstruction('d4', 12., (0,)))
    assert propose_lane(0, PASSAGE, CLEAR, rows) == 2
    assert propose_lane(2, PASSAGE, CLEAR, rows, active_lane=0) == 0
    assert propose_lane(0, PASSAGE, {0: {'side': ['d4']}, 2: {}}, rows) == 2
    assert propose_lane(0, PASSAGE, {0: {'side': ['d4']}, 2: {'front': ['d3']}}, rows) is None


def handoff_controller():
    s = active_session()
    s.hybrid.travelled = 3.
    return NS(_overtake=s, _lane_decision=LaneDecision('d2',2,2,'hybrid_lane_transition'),
              _constraint_transition_until=18., _follow_latched_cache={'vehicle_id':'d2'},
              _prepass_dynamic_conflict_speed_limit=0.,
              _reset_outer_lane_progress=Mock(), _clear_consecutive_overtake_handoff=Mock(),
              _clear_urgent_overtake_switch_candidate=Mock(), get_logger=lambda:Mock(),
              _reference_path=NS(target_lane_idx=2,is_overtaking=True),
              _reference_pathN=NS(target_lane_idx=2,is_overtaking=True),
              _mpcN_center=Mock(), _hybrid_overtake_transition_timeout=8.,
              _hybrid_overtake_enabled=True, _l1_probe_active=False)


def test_target_handoff_and_actual_lane_writer_keep_geometry_and_deadline():
    c=handoff_controller()
    anchor=c._overtake.hybrid
    for successor in ('d3', 'd4', 'd2'):
        assert controller_method('_accept_same_lane_target_handoff')(c,successor,2)
        controller_method('_apply_lane_decision')(
            c,requested_lane=2,now_sec=12.,l0_prohibited=False,full_width_recovery=False)
        assert c._overtake.hybrid is anchor
        assert (anchor.start_wp,anchor.started_at,anchor.travelled)==(255,10.,3.)
        assert c._constraint_transition_until==18.
        assert c._reference_path.target_lane_idx==c._reference_pathN.target_lane_idx==2
        assert c._overtake.verification.vehicle_id==successor
    c._mpcN_center.set_lane_transition_weights.assert_not_called()
    assert c._prepass_dynamic_conflict_speed_limit is None
    assert c._follow_latched_cache is None


def test_session_cannot_inherit_geometry_across_side_change_or_completed_pass():
    s=active_session()
    original=copy.deepcopy(s)
    assert not s.handoff_target('d3',0)
    assert s==original
    s.hybrid.completed=True
    assert not s.handoff_target('d3',2)


@pytest.mark.parametrize('problem', ['none','width','traffic','overlap','unknown','recovery'])
def test_handoff_admission_checks_all_three_opponents(problem):
    c=handoff_controller()
    c._stuck_recovery_until=None
    c._parallel_abort_active=problem=='recovery'
    c._vehicle_passage=lambda *a:(PASSAGE,10.)
    c._lane_horizon_has_vehicle_width=lambda *a:problem!='width'
    c._follow_escape_lane_traffic_is_clear=lambda *a:problem!='traffic'
    c._v2x_tracker=NS(active_vehicle_ids=lambda:['d2','d3','d4'],
                      has_velocity_estimate=lambda vid:not(problem=='unknown' and vid=='d4'))
    c._committed_target_body_overlap=lambda pose,vid:problem=='overlap' and vid=='d4'
    before=copy.deepcopy(c._overtake)
    assert controller_method('_same_lane_target_handoff_available')(c,'d3',NS(),1.) == (problem=='none')
    assert c._overtake==before


def test_pending_successor_probe_does_not_release_or_rewrite_live_state():
    c=handoff_controller()
    c._same_lane_target_handoff_available=lambda *a:True
    c._overtake_commit_probe_is_fresh=lambda *a:False
    c._prepare_overtake_commit_probe=Mock()
    node=next(n for n in ast.walk(control_tree()) if isinstance(n,ast.If)
              and '_same_lane_target_handoff_available' in ast.unparse(n.test)
              and 'target_switch_confirmed' in ast.unparse(n.test))
    ns=dict(self=c,target_switch_confirmed=True,opponent_vehicle_id='d3',
            pose=NS(),v=1.,now_sec=12.,same_lane_handoff=False)
    before=copy.deepcopy(c._overtake)
    exec(compile(ast.Module(body=[node],type_ignores=[]),'<pending-handoff>','exec'),ns)
    assert not ns['target_switch_confirmed']
    assert c._overtake==before and c._constraint_transition_until==18.
    assert c._reference_path.target_lane_idx==2
    c._prepare_overtake_commit_probe.assert_called_once_with('d3',2)


def test_controller_advisory_with_no_candidate_is_read_only():
    c=handoff_controller()
    c._overtake.committed=False
    c._overtake.hybrid.verified_start=False
    c._v2x_tracker=NS(active_vehicle_ids=lambda:[])
    before=copy.deepcopy(c._overtake)
    assert controller_method('_propose_traffic_lane')(c,2,{},CLEAR,'d2',NS()) is None
    assert c._overtake==before
    assert c._reference_path.target_lane_idx==2
