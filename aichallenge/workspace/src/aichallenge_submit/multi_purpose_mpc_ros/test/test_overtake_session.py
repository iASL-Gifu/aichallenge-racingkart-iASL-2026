"""Session handoff and real controller application tests without ROS startup."""
import ast
import copy
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
import math
import numpy as np
from functools import lru_cache
from multi_purpose_mpc_ros.core.MPC import spatial_lane_transition_reference, blend_previous_lateral_prediction
from multi_purpose_mpc_ros.v2x_vehicle_tracker import overtake_shadow_solution_acceptable
from unittest.mock import Mock

import pytest

from multi_purpose_mpc_ros.overtake_session import (
    HybridTransition, OvertakeSession, ShadowVerification, decide_lane,
)


def decision(target="d2", lane=2, **kwargs):
    values = dict(target_id=target, requested_lane=lane, l0_prohibited=False,
                  full_width_recovery=False, transition_active=True,
                  hybrid_active=True)
    values.update(kwargs)
    return decide_lane(**values)


def active_session():
    session = OvertakeSession(target_id="d2", requested_lane=2, committed=True)
    session.apply(decision())
    session.hybrid = HybridTransition(
        vehicle_id="d2", lane_idx=2, start_wp=255, start_e_y=-0.9,
        started_at=10.0, length=8.0)
    session.verification = ShadowVerification("d2", 2)
    return session


@pytest.mark.parametrize("target,lane", [("d2", 0), ("d3", 2)])
def test_side_or_target_change_discards_previous_spatial_anchor_and_proof(target, lane):
    session = active_session()
    assert session.apply(decision(target, lane))
    assert session.accepted_key == (target, lane)
    assert session.hybrid == HybridTransition()
    assert session.verification == ShadowVerification()


def test_temporary_full_width_recovery_preserves_same_pass_for_resume():
    session = active_session()
    hybrid = session.hybrid
    assert not session.apply(decision(full_width_recovery=True), preserve_manoeuvre=True)
    assert session.hybrid is hybrid
    assert not session.apply(decision())
    assert session.hybrid.start_wp == 255


def test_recovery_can_request_full_width_without_losing_the_latched_side():
    session = active_session()
    assert not session.apply(decision(lane=None, full_width_recovery=True), preserve_manoeuvre=True)
    assert session.accepted_key == ("d2", 2)
    assert session.hybrid.start_wp == 255
    assert session.verification == ShadowVerification()


def test_recommit_discards_pending_probe_for_previous_side():
    session = active_session()
    session.probe.vehicle_id = "d2"
    session.probe.lane_idx = 2
    session.probe.confirmed = True
    session.apply(decision(lane=0))
    assert not session.probe.confirmed
    assert session.probe.vehicle_id is None


def test_new_verified_side_keeps_only_its_own_proof():
    session = active_session()
    session.verification = ShadowVerification("d2", 0)
    session.apply(decision(lane=0))
    assert session.verification == ShadowVerification("d2", 0)
    assert session.hybrid.vehicle_id is None


@pytest.mark.parametrize("recovery,prohibited,lane,expected,mode", [
    (True, True, 0, None, "full_width_recovery"),
    (False, True, 0, 1, "l0_prohibited"),
    (False, True, 2, 2, "l0_prohibited"),
    (False, False, 0, 0, "hybrid_lane_transition"),
])
def test_hybrid_never_overrides_geographic_or_recovery_decisions(
    recovery, prohibited, lane, expected, mode,
):
    result = decision(lane=lane, full_width_recovery=recovery,
                      l0_prohibited=prohibited)
    assert (result.applied_lane, result.mode) == (expected, mode)


@lru_cache(maxsize=None)
def controller_method(name):
    # Execute the real method body, not a second implementation; ROS message
    # packages are not required for the state/constraint integration contract.
    source = Path(__file__).parents[1] / "multi_purpose_mpc_ros/mpc_controller.py"
    tree = ast.parse(source.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef)
               and n.name == "MPCController")
    method = copy.deepcopy(next(n for n in cls.body if isinstance(n, ast.FunctionDef)
                                and n.name == name))
    from multi_purpose_mpc_ros.overtake_lane_hold import dynamic_longitudinal_conflict_unsafe
    from multi_purpose_mpc_ros.v2x_vehicle_tracker import slow_lead_commit_distance
    from multi_purpose_mpc_ros.v2x_vehicle_tracker import hybrid_lateral_escape_creep_allowed, classify_lane_conflicts
    namespace = dict(hybrid_lateral_escape_creep_allowed=hybrid_lateral_escape_creep_allowed,
                     classify_lane_conflicts=classify_lane_conflicts, dynamic_longitudinal_conflict_unsafe=dynamic_longitudinal_conflict_unsafe,
                     slow_lead_commit_distance=slow_lead_commit_distance, decide_lane=decide_lane, copy=copy, contextmanager=contextmanager,
                     np=np, math=math, spatial_lane_transition_reference=spatial_lane_transition_reference,
                     blend_previous_lateral_prediction=blend_previous_lateral_prediction,
                     overtake_shadow_solution_acceptable=overtake_shadow_solution_acceptable)
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(source), "exec"), namespace)
    return namespace[name]


def test_controller_applies_new_lane_to_both_paths_and_invalidates_old_hybrid():
    session = active_session()
    controller = SimpleNamespace(
        _overtake=session, _lane_decision=decision(),
        _hybrid_overtake_transition_timeout=8.0, _hybrid_overtake_enabled=True,
        _reference_path=SimpleNamespace(target_lane_idx=2, is_overtaking=True),
        _reference_pathN=SimpleNamespace(target_lane_idx=2, is_overtaking=True),
        _mpcN_center=Mock(), _reset_outer_lane_progress=Mock(),
        _l1_probe_active=False, get_logger=Mock(return_value=Mock()),
    )
    result = controller_method("_apply_lane_decision")(
        controller, requested_lane=0, now_sec=12.0,
        l0_prohibited=False, full_width_recovery=False)
    assert result == (0, True, False)
    assert controller._reference_path.target_lane_idx == 0
    assert controller._reference_pathN.target_lane_idx == 0
    assert session.accepted_key == ("d2", 0)
    assert session.hybrid.start_wp is None
    assert session.verification.vehicle_id is None


def test_probe_cannot_mutate_live_lane_or_waypoint_even_on_failure():
    path = SimpleNamespace(
        target_lane_idx=2, is_overtaking=True,
        waypoints=[SimpleNamespace(ub=3.0)], border_cells=SimpleNamespace(),
        unsafe_static_fallback_wp_ids=[])
    model = SimpleNamespace(reference_path=path)
    with pytest.raises(RuntimeError):
        with controller_method("_probe_corridor")(None, model, 0) as snapshot:
            assert path.target_lane_idx == 2
            assert model.reference_path.target_lane_idx == 0
            snapshot.waypoints[0].ub = 1.0
            raise RuntimeError("probe failed")
    assert model.reference_path is path
    assert path.waypoints[0].ub == 3.0
