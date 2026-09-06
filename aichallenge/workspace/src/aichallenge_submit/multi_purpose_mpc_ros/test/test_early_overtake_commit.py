"""Execute actual distance/Shadow gates to guard the widened entry range."""
import ast
import copy
from pathlib import Path
from types import MethodType, SimpleNamespace
from unittest.mock import Mock

import pytest
import yaml

from multi_purpose_mpc_ros.v2x_vehicle_tracker import (
    evaluate_overtake_commit_gate, is_follow_retry_within_distance,
)
from .test_hybrid_integration import control_tree
from .test_overtake_session import controller_method


def controller():
    config = yaml.safe_load((Path(__file__).parents[1]/'config/config.yaml').read_text())
    switch = config['trajectory_switch']
    c = SimpleNamespace(
        _overtake=SimpleNamespace(requested_lane=None, probe=SimpleNamespace(success_cycles=0)),
        _l2_entry_restricted_zones=[], _waypoint_in_configured_zones=lambda *a: False,
        _ultra_slow_early_commit_speed=switch['ultra_slow_early_commit_speed_kmh']/3.6,
        _overtake_commit_min_distance=switch['overtake_commit_min_distance'],
        _overtake_outside_curvature_threshold=switch['overtake_outside_curvature_threshold'],
        _overtake_commit_curvature_preview=lambda wp: [0.],
        _overtake_commit_probe_required_success_cycles=2,
        _reset_overtake_commit_probe=Mock(), _prepare_overtake_commit_probe=Mock(),
        get_logger=Mock(return_value=Mock()))
    for key in ('slow_lead_overtake_prepare_distance', 'slow_lead_overtake_commit_distance',
                'late_defense_slow_lead_overtake_commit_distance', 'overtake_latch_max_distance'):
        setattr(c, '_'+key, switch[key])
    c._slow_lead_commit_distance_at = MethodType(controller_method('_slow_lead_commit_distance_at'), c)
    return c


def run_gate(c, *, speed, distance, fresh):
    # Select the actual contiguous distance + strict Shadow block, stopping
    # immediately before the latch writer. No replica of that logic is used.
    bodies = (value for parent in ast.walk(control_tree())
              for _, value in ast.iter_fields(parent) if isinstance(value, list))
    for body in bodies:
        start = next((i for i,n in enumerate(body) if isinstance(n, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == 'existing_outer_latch' for t in n.targets)), None)
        if start is not None:
            end = next(i for i in range(start,len(body)) if isinstance(body[i], ast.If))
            nodes = copy.deepcopy(body[start:end+1])
            break
    else:
        raise AssertionError('actual commit block not found')
    c._overtake_commit_probe_is_fresh = Mock(return_value=fresh)
    values = dict(self=c, center_wp_temp=0, opponent_v_lead=speed,
        lead_is_stationary=speed<=.1, lead_is_special_slow=.1<speed<=3.,
        opponent_arc_distance=distance, opponent_vehicle_id='target',
        new_target_lane_idx=2, now=SimpleNamespace(nanoseconds=10_000_000_000),
        is_follow_retry_within_distance=is_follow_retry_within_distance,
        evaluate_overtake_commit_gate=evaluate_overtake_commit_gate)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), '<actual-entry-gates>', 'exec'), values)
    return values


@pytest.mark.parametrize('speed,distance,fresh,expected_lane', [
    (0.,34.9,False,1), (0.,34.9,True,2),
    (1.5,34.9,False,1), (1.5,34.9,True,2),
    (1.501,34.9,True,1), (2.,27.9,True,2), (2.,27.9,False,1),
    (0.,35.,True,1), (2.,28.,True,1), (0.,35.1,True,1),
])
def test_actual_entry_distance_never_bypasses_strict_shadow(speed,distance,fresh,expected_lane):
    result = run_gate(controller(), speed=speed, distance=distance, fresh=fresh)
    assert result['latch_candidate_lane_idx'] == expected_lane
    assert result['latch_candidate_vehicle_id'] == ('target' if expected_lane==2 else None)


@pytest.mark.parametrize('speed,expected', [(None,28.), (float('nan'),28.), (0.,35.), (1.5,35.), (1.501,28.)])
def test_actual_distance_helper_unknown_speed_and_threshold(speed,expected):
    assert controller()._slow_lead_commit_distance_at(0, lead_speed=speed) == expected


def test_follow_retry_uses_same_speed_sensitive_distance():
    assignment = next(n for n in ast.walk(control_tree()) if isinstance(n,ast.Assign)
        and any(isinstance(t,ast.Name) and t.id=='follow_retry_commit_distance' for t in n.targets))
    for speed, expected in [(0.,35.), (1.5,35.), (2.,28.), (None,28.)]:
        values = dict(self=controller(), center_wp_temp=0, follow_retry_state={'speed':speed})
        exec(compile(ast.Module(body=[copy.deepcopy(assignment)],type_ignores=[]), '<actual-retry>', 'exec'),values)
        assert values['follow_retry_commit_distance']==expected
